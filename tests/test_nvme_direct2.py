#!/usr/bin/env python3
"""test_nvme_direct2.py — host-side unit tests for the NVMe-direct offload
spec (overlay/kvoffload/nvme_direct2.py). Runs against the vllm package in
the serve image or a dev checkout; needs no GPU.

Covers the contract the boot depends on:
  * fence digest sensitivity: any field that changes KV meaning -> new digest
  * revision resolution: config > env > HF cache > 'unresolved' sentinel
  * sidecar protocol: adopt equal, refuse disagreeing, refuse data-without-sidecar
  * path layout: shard dir, group suffix
  * worker plan: positional pairing, group from path, counts validated
  * store durability: short write must not truncate
  * manager: HIT_PENDING, stale-positive recheck (TTL sweeper), reset_cache
    clears memory only
"""
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from typing import Any

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
sys.path.insert(0, _root)
_spec = importlib.util.spec_from_file_location(
    "nvme_direct2", os.path.join(_root, "overlay", "kvoffload", "nvme_direct2.py")
)
nd2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nd2)


def make_config(**over) -> SimpleNamespace:
    base: dict[str, Any] = dict(
        name="Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw",
        dtype="fp8",
        tokens_per_hash=64,
        blocks_per_chunk=1,
        tp_size=2,
        pp_size=1,
        pcp_size=1,
        dcp_size=1,
        worker_kv_bytes_per_block=4096,
        revision="rev-a",
        groups=((3584, ("layer.a", "layer.b")), (3584, ("layer.c",))),
        extra={},
    )
    base.update(over)
    groups = tuple(
        SimpleNamespace(tokens_per_block=t, layer_names=l)
        for t, l in base["groups"]
    )
    return SimpleNamespace(
        model=SimpleNamespace(name=base["name"], dtype=base["dtype"]),
        worker_kv_bytes_per_block=base["worker_kv_bytes_per_block"],
        cache=SimpleNamespace(
            tokens_per_hash=base["tokens_per_hash"],
            blocks_per_chunk=base["blocks_per_chunk"],
        ),
        parallel=SimpleNamespace(
            tp_size=base["tp_size"], pp_size=base["pp_size"],
            pcp_size=base["pcp_size"], dcp_size=base["dcp_size"],
            rank=0,
        ),
        groups=groups,
        extra_config=base["extra"],
        enable_kv_cache_events=False,
        engine_id="test-engine",
        _revision=base["revision"],
    )


def digest_of(**over) -> str:
    return nd2._fence_digest(
        nd2._fence_fields(make_config(**over), over.get("revision", "rev-a"))
    )


class TestFence(unittest.TestCase):
    def test_digest_deterministic(self):
        self.assertEqual(digest_of(), digest_of())

    def test_digest_sensitive_to_every_field(self):
        d = digest_of()
        for kw in (
            dict(name="other/Model"),
            dict(tokens_per_hash=32),
            dict(tp_size=4),
            dict(pp_size=2),
            dict(groups=((3584, ("layer.a",)), (3584, ("layer.c",)))),
            dict(groups=((1024, ("layer.a", "layer.b")), (3584, ("layer.c",)))),
            dict(revision="rev-b"),
        ):
            rev = kw.get("revision", "rev-a")
            fields = nd2._fence_fields(make_config(**kw), rev)
            self.assertNotEqual(d, nd2._fence_digest(fields), f"insensitive to {kw}")

    def test_layer_order_within_group_irrelevant(self):
        a = nd2._fence_digest(
            nd2._fence_fields(make_config(groups=((3584, ("x.a", "x.b")),)), "rev-a")
        )
        b = nd2._fence_digest(
            nd2._fence_fields(make_config(groups=((3584, ("x.b", "x.a")),)), "rev-a")
        )
        self.assertEqual(a, b)

    def test_revision_priority(self):
        os.environ.pop("MODEL_REVISION", None)
        self.assertEqual(
            nd2.resolve_model_revision("m", "from-config"), "from-config"
        )
        os.environ["MODEL_REVISION"] = " from-env "
        try:
            self.assertEqual(nd2.resolve_model_revision("m", None), "from-env")
        finally:
            del os.environ["MODEL_REVISION"]
        # unresolvable id, offline -> sentinel
        self.assertEqual(
            nd2.resolve_model_revision("no/such-model-xyz", None), "unresolved"
        )

    def test_sidecar_adopt_and_refuse(self):
        with tempfile.TemporaryDirectory() as td:
            f = nd2.Fence(td, nd2._fence_fields(make_config(), "rev-a"))
            f.adopt_or_create("t")
            self.assertTrue(os.path.exists(f.sidecar_path))
            with open(f.sidecar_path) as fh:
                self.assertEqual(json.load(fh)["digest"], f.digest)
            # identical fields -> silent adopt (second boot)
            nd2.Fence(td, f.fields).adopt_or_create("t")
            # tamper the on-disk digest -> refuse (collision read-back path)
            with open(f.sidecar_path) as fh:
                doc = json.load(fh)
            doc["digest"] = "0" * 64
            with open(f.sidecar_path, "w") as fh:
                json.dump(doc, fh)
            with self.assertRaises(RuntimeError):
                nd2.Fence(td, f.fields).adopt_or_create("t")

    def test_sidecar_missing_with_data_refuses(self):
        with tempfile.TemporaryDirectory() as td:
            f = nd2.Fence(td, nd2._fence_fields(make_config(), "rev-a"))
            os.makedirs(os.path.join(f.base_dir, "r0", "abc"))
            with open(
                os.path.join(f.base_dir, "r0", "abc", "deadbeef_g0.bin"), "w"
            ) as fh:
                fh.write("x")
            with self.assertRaises(RuntimeError):
                f.adopt_or_create("t")

    def test_probe_narrows_to_own_rank(self):
        # Shared filesystem, our rank's subtree empty but another rank's is not
        # (stale tree from a different deployment): must NOT refuse; we adopt
        # and co-write under one sidecar.
        with tempfile.TemporaryDirectory() as td:
            f = nd2.Fence(td, nd2._fence_fields(make_config(), "rev-a"))
            other = os.path.join(f.base_dir, "r0", "abc")
            os.makedirs(other)
            with open(os.path.join(other, "deadbeef_g0.bin"), "wb") as fh:
                fh.write(b"x")
            f.adopt_or_create("t", rank=1)  # probes r1 only -> clean
            self.assertTrue(os.path.exists(f.sidecar_path))
            # the reverse: OUR subtree holds data with no sidecar -> refuse
            mine = os.path.join(f.base_dir, "r1", "abc")
            os.makedirs(mine)
            with open(os.path.join(mine, "deadbeef_g0.bin"), "wb") as fh:
                fh.write(b"x")
            os.unlink(f.sidecar_path)
            g = nd2.Fence(td, f.fields)
            with self.assertRaises(RuntimeError):
                g.adopt_or_create("t", rank=1)

    def test_dir_name_shape(self):
        fields = nd2._fence_fields(make_config(), "rev-a")
        f = nd2.Fence("/root", fields)
        d = nd2._fence_digest(fields)
        self.assertEqual(
            f.base_dir,
            os.path.join("/root", f"Mia-AiLab_GLM-5.3-Flash-EXL3-TR3-4bpw_{d[:16]}"),
        )


class TestPaths(unittest.TestCase):
    def test_layout(self):
        key = nd2.make_offload_key(bytes.fromhex("aa" * 32), 2)
        rp = nd2._relpath(key)
        self.assertEqual(rp, os.path.join("aaa", f"{'aa' * 32}_g2.bin"))

    def _fake_worker(self, group_bytes):
        w = object.__new__(nd2.NvmeDirectWorker2)
        w._dir = "/x"
        w._group_bytes = group_bytes
        return w

    def test_group_idx_from_path_not_order(self):
        w = self._fake_worker([10, 20])
        paths = [
            nd2._relpath(nd2.make_offload_key(bytes.fromhex("bb" * 32), 1)),
            nd2._relpath(nd2.make_offload_key(bytes.fromhex("cc" * 32), 0)),
        ]
        rows, total = w._plan(
            SimpleNamespace(block_ids=[5, 6], group_sizes=None), paths
        )
        self.assertEqual([g for _, g, _ in rows], [1, 0])
        self.assertEqual([b for _, _, b in rows], [5, 6])
        self.assertEqual(total, 30)

    def test_group_sizes_crosscheck(self):
        w = self._fake_worker([10, 20])
        p0 = nd2._relpath(nd2.make_offload_key(bytes.fromhex("bb" * 32), 0))
        p1 = nd2._relpath(nd2.make_offload_key(bytes.fromhex("cc" * 32), 1))
        rows, total = w._plan(
            SimpleNamespace(block_ids=[1, 2], group_sizes=(1, 1)), [p0, p1]
        )
        self.assertEqual(total, 30)
        with self.assertRaises(ValueError):  # sizes sum != counts
            w._plan(SimpleNamespace(block_ids=[1, 2], group_sizes=(2, 2)), [p0, p1])

    def test_plan_validation(self):
        w = self._fake_worker([10])
        with self.assertRaises(ValueError):  # file/block count mismatch
            w._plan(SimpleNamespace(block_ids=[1, 2], group_sizes=None), ["a_g0.bin"])
        with self.assertRaises(ValueError):  # missing group suffix
            w._plan(SimpleNamespace(block_ids=[1], group_sizes=None), ["a.bin"])
        with self.assertRaises(ValueError):  # group out of range
            w._plan(SimpleNamespace(block_ids=[1], group_sizes=None), ["a_g9.bin"])


class TestFinishedContract(unittest.TestCase):
    class _F:
        def __init__(self, done=True, exc=None, result=(7, 0.5)):
            self._d, self._e, self._r = done, exc, result
        def done(self):
            return self._d
        def exception(self):
            return self._e
        def result(self):
            if self._e:
                raise self._e
            return self._r

    def _w(self):
        w = object.__new__(nd2.NvmeDirectWorker2)
        w._jobs = {}
        return w

    def test_metrics_never_none_time(self):
        # worker.py records transfer bytes only when BOTH size and time are
        # present; a None transfer_time silently kills the gate counters.
        w = self._w()
        w._jobs[1] = (self._F(), 0, False)
        (r,) = w.get_finished()
        self.assertTrue(r.success)
        self.assertEqual((r.transfer_size, r.transfer_time), (7, 0.5))
        self.assertEqual(w._jobs, {})

    def test_store_failure_keeps_result_contract(self):
        # submit's bool and result.success are asserted by the connector; a
        # failed store must log, report success with zero bytes, and raise
        # nothing.
        w = self._w()
        w._jobs[2] = (self._F(exc=OSError("disk gone")), 0, False)
        (r,) = w.get_finished()
        self.assertTrue(r.success)
        self.assertEqual((r.transfer_size, r.transfer_time), (0, 0.0))

    def test_load_failure_raises(self):
        # committed restore cannot degrade silently; crash loudly.
        w = self._w()
        w._jobs[3] = (self._F(exc=OSError("short read")), 0, True)
        with self.assertRaises(OSError):
            w.get_finished()


class TestStoreWrite(unittest.TestCase):
    def test_write_all_loops_short_writes(self):
        captured = bytearray()
        lens = []

        def fake_write(fd, data):
            lens.append(len(data))
            captured.extend(bytes(data[:1]))  # always writes 1 byte
            return 1

        real_write = os.write
        os.write = fake_write
        try:
            nd2._write_all(3, memoryview(b"hello"))
        finally:
            os.write = real_write
        self.assertEqual(bytes(captured), b"hello")
        self.assertEqual(lens, [5, 4, 3, 2, 1])


class TestManager(unittest.TestCase):
    def _mgr(self, td):
        return nd2.NvmeDirectManager2(os.path.join(td, "r0"))

    def _key(self, h, g=0):
        return nd2.make_offload_key(bytes.fromhex(h * 32), g)

    def _touch(self, m, k):
        p = m._path(k)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as fh:
            fh.write(b"x")
        return p

    def test_pending_then_hit_then_sweeper(self):
        with tempfile.TemporaryDirectory() as td:
            m = self._mgr(td)
            k = self._key("a")
            self.assertIs(m.lookup(k, None), nd2.LookupResult.MISS)
            out = m.prepare_store([k], None)
            self.assertIsNotNone(out)
            self.assertIs(m.lookup(k, None), nd2.LookupResult.HIT_PENDING)
            p = self._touch(m, k)
            m.complete_store(list(out.keys_to_store), None, success=True)
            self.assertIs(m.lookup(k, None), nd2.LookupResult.HIT)
            # TTL sweeper removes the file: lookup must NOT report HIT
            os.unlink(p)
            self.assertIs(m.lookup(k, None), nd2.LookupResult.MISS)

    def test_failed_store_leaves_miss(self):
        with tempfile.TemporaryDirectory() as td:
            m = self._mgr(td)
            k = self._key("b")
            out = m.prepare_store([k], None)
            # store job failed: never touch the file
            m.complete_store(list(out.keys_to_store), None, success=False)
            self.assertIs(m.lookup(k, None), nd2.LookupResult.MISS)

    def test_dedupe_existing_file(self):
        with tempfile.TemporaryDirectory() as td:
            m = self._mgr(td)
            k = self._key("c")
            self._touch(m, k)
            self.assertIsNone(m.prepare_store([k], None))  # already there
            self.assertIs(m.lookup(k, None), nd2.LookupResult.HIT)

    def test_prepare_load_order(self):
        with tempfile.TemporaryDirectory() as td:
            m = self._mgr(td)
            ks = [self._key(h) for h in "abc"]
            spec = m.prepare_load(ks, None)
            self.assertEqual(spec.relpaths, [nd2._relpath(k) for k in ks])

    def test_reset_clears_memory_only(self):
        with tempfile.TemporaryDirectory() as td:
            m = self._mgr(td)
            k = self._key("d")
            p = self._touch(m, k)
            m.lookup(k, None)
            m.reset_cache()
            self.assertEqual(len(m._exists), 0)  # base get_stats -> None
            self.assertTrue(os.path.exists(p))  # files survive
            self.assertIs(m.lookup(k, None), nd2.LookupResult.HIT)  # repopulate


class TestSpecGuards(unittest.TestCase):
    def test_blocks_per_chunk_guard(self):
        cfg = make_config(blocks_per_chunk=2, extra={"root_dir": "/x"})
        with self.assertRaises(ValueError):
            nd2.NvmeDirectOffloadingSpec2(cfg)

    def test_root_dir_required(self):
        cfg = make_config(extra={})
        with self.assertRaises(ValueError):
            nd2.NvmeDirectOffloadingSpec2(cfg)

    def test_root_dir_absolute(self):
        cfg = make_config(extra={"root_dir": "relative/path"})
        with self.assertRaises(ValueError):
            nd2.NvmeDirectOffloadingSpec2(cfg)


if __name__ == "__main__":
    unittest.main(verbosity=2)
