"""CPU-only draft-page tests; no torch import, model, or serving process.

Set GLM53_VLLM_SRC to pristine vLLM 487ecf187d3dfe74d2cf6119a92881dba403c219
sources to include real grouping, tensor allocation, and backend-selection tests.
Only core/kv_cache_utils.py, worker/utils.py, and kv_cache_interface.py under
vllm/v1 are needed. The fixtures are hash-checked and copied before patching.
"""
from __future__ import annotations

import ast
import collections
import copy
import dataclasses
import enum
import hashlib
import importlib.util
import logging
import math
import os
from pathlib import Path
import sys
import types
import typing

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "draft_patch", ROOT / "overlay/patch_glm5_drafter_group.py"
)
patch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(patch)
ENV = "GLM53_DRAFT_KV_COMPACT"
PINS = {
    "core/kv_cache_utils.py": "f1c553daa9f214e03126fe30de9c89e764b0a5168464e66fd9f739e168ffffc7",
    "worker/utils.py": "3dcd6ad34ee1d1db2875f7f7dd51d90ee0e64041ab282180687770a38b26acb1",
    "kv_cache_interface.py": "54a761dd60907945c8f3bfc450b1264033013b83b204320581961f887b03e5b0",
}


def test_compact_block_preserves_alignment_and_fits_page(monkeypatch):
    monkeypatch.setenv(ENV, "1")
    ns = {"os": os}
    exec(patch.COMPACT_BLOCK_HELPER, ns)
    choose = ns["_glm53_draft_block_size"]
    # Sweep non-power-of-two MLA pages and different TP-local draft widths.
    for mla_block in range(64, 8193, 64):
        for bytes_per_token in (512, 1024, 2048, 3072, 4096):
            page = mla_block * 656
            if page < 64 * bytes_per_token:
                with pytest.raises(ValueError):
                    choose(mla_block, page, bytes_per_token)
                continue
            block = choose(mla_block, page, bytes_per_token)
            assert block % 64 == 0
            assert math.lcm(mla_block, block) == mla_block
            assert block * bytes_per_token <= page
            legal = [b for b in range(64, mla_block + 1, 64)
                     if mla_block % b == 0 and b * bytes_per_token <= page]
            assert block == max(legal)
    assert choose(3584, 3584 * 656, 2048) == 896
    assert choose(4608, 4608 * 656, 2048) == 1152


def test_opt_out_and_invalid_configuration(monkeypatch):
    ns = {"os": os}
    exec(patch.COMPACT_BLOCK_HELPER, ns)
    choose = ns["_glm53_draft_block_size"]
    monkeypatch.delenv(ENV, raising=False)
    assert choose(3584, 3584 * 656, 2048) == 64
    for bad in ("", "auto", "true", "01"):
        monkeypatch.setenv(ENV, bad)
        with pytest.raises(ValueError):
            choose(3584, 3584 * 656, 2048)
    monkeypatch.setenv(ENV, "1")
    for geometry in ((65, 65536, 512), (0, 65536, 512), (64, 65536, 0)):
        with pytest.raises(ValueError):
            choose(*geometry)


@pytest.fixture
def sources(tmp_path):
    root = os.environ.get("GLM53_VLLM_SRC")
    if not root:
        pytest.skip("set GLM53_VLLM_SRC for pinned-source CPU integration")
    for rel, sha in PINS.items():
        data = (Path(root) / "vllm/v1" / rel).read_bytes()
        assert hashlib.sha256(data).hexdigest() == sha, f"source drift: {rel}"
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return tmp_path


def definitions(path, ns, names=None):
    nodes = [n for n in ast.parse(path.read_text()).body
             if isinstance(n, (ast.ClassDef, ast.FunctionDef))
             and (names is None or n.name in names)]
    tree = ast.Module(body=[ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    ), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), ns)


@pytest.fixture
def allocator(sources, monkeypatch):
    patch.patch_file(str(sources / "core/kv_cache_utils.py"))
    module = types.ModuleType("draft_cache_cpu_fixture")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    ns = module.__dict__
    ns.update({
        "dataclass": dataclasses.dataclass, "fields": dataclasses.fields,
        "replace": dataclasses.replace, "Enum": enum.Enum, "IntEnum": enum.IntEnum,
        "Counter": collections.Counter, "defaultdict": collections.defaultdict,
        "prod": math.prod, "math": math, "copy": copy, "os": os,
        "cast": typing.cast, "get_dtype_size": lambda dtype: dtype,
        "cdiv": lambda a, b: (a + b - 1) // b,
        "round_up": lambda a, b: (a + b - 1) // b * b,
        "MambaAttentionBackendEnum": types.SimpleNamespace(MAMBA2="mamba2"),
        "logger": logging.getLogger(__name__),
        "MultipleOf": dataclasses.make_dataclass("MultipleOf", [("base", int)]),
    })
    definitions(sources / "kv_cache_interface.py", ns)
    ns["KVCacheSpecRegistry"] = types.SimpleNamespace(
        get_uniform_type_base_spec=lambda spec: type(spec)
    )
    definitions(sources / "core/kv_cache_utils.py", ns, {
        "_glm53_draft_block_size", "_get_kv_cache_groups_glm5_next",
        "_pp_balanced_mamba_group_count", "create_kv_cache_group_specs",
        "_glm5_next_tensor_layout", "_pool_bytes_per_block",
        "get_kv_cache_config_from_groups", "may_override_num_blocks",
    })
    definitions(sources / "worker/utils.py", ns, {
        "select_common_block_size", "prepare_kernel_block_sizes",
    })
    return ns


def make_groups(ns, block=3584, draft_heads=4):
    simple = types.SimpleNamespace
    config = simple(
        parallel_config=simple(pipeline_parallel_size=1),
        cache_config=simple(num_gpu_blocks_override=None),
    )
    specs = {}
    for i in range(11):
        specs[f"mla.{i}"] = ns["MLAAttentionSpec"](
            block_size=block, num_kv_heads=1, head_size=576,
            dtype=1, cache_dtype_str="fp8_ds_mla",
        )
        specs[f"indexer.{i}"] = ns["MLAAttentionSpec"](
            block_size=block, num_kv_heads=1, head_size=128,
            dtype=2, compress_ratio=4,
        )
        specs[f"tail.{i}"] = ns["KpoolTailSpec"](
            block_size=4, num_kv_heads=1, head_size=128,
            dtype=2, sliding_window=4,
        )
    for i in range(34):
        specs[f"mamba.{i}"] = ns["MambaSpec"](
            block_size=block, shapes=((1024,),), dtypes=(2,),
        )
    for i in range(5):
        specs[f"draft.{i}"] = ns["SlidingWindowSpec"](
            block_size=64, num_kv_heads=draft_heads, head_size=128,
            dtype=2, sliding_window=2048,
        )
    groups = ns["_get_kv_cache_groups_glm5_next"](config, specs)
    cache = ns["get_kv_cache_config_from_groups"](config, groups, 16 * 1024**3)
    draft = next(iter(groups[-1].kv_cache_spec.kv_cache_specs.values()))
    return groups, cache, draft


def test_reservation_reduction_without_extra_allocation(allocator, monkeypatch):
    monkeypatch.setenv(ENV, "0")
    old_groups, old_cache, old_draft = make_groups(allocator)
    monkeypatch.setenv(ENV, "1")
    groups, cache, draft = make_groups(allocator)
    assert old_draft.max_admission_blocks_per_request(2048, 262144) == 65
    assert draft.max_admission_blocks_per_request(2048, 262144) == 6
    assert cache.num_blocks == old_cache.num_blocks
    assert cache.kv_cache_tensors == old_cache.kv_cache_tensors
    assert groups[:-1] == old_groups[:-1]
    assert math.lcm(3584, draft.block_size) == 3584
    # Reject a page that fits in bytes but increases prefix-cache alignment.
    malformed = dataclasses.replace(draft, block_size=1024)
    groups[-1] = allocator["KVCacheGroupSpec"](
        ["draft.0"], allocator["UniformTypeKVCacheSpecs"].from_specs({"draft.0": malformed})
    )
    assert allocator["_glm5_next_tensor_layout"](groups) is None


def test_backend_split_guard_and_unpadded_exception(allocator, monkeypatch):
    monkeypatch.setenv(ENV, "1")
    groups, cache, draft = make_groups(allocator)
    cache.kv_cache_groups = [groups[-1]]

    def select(supported):
        backend = type("Backend", (), {
            "get_supported_kernel_block_sizes": staticmethod(lambda: supported)
        })
        return allocator["prepare_kernel_block_sizes"](
            cache, [[types.SimpleNamespace(backend=backend)]]
        )

    assert select([allocator["MultipleOf"](16)]) == [896]
    with pytest.raises(ValueError, match="cannot be split"):
        select([64])
    # Exact-fit pages are contiguous and can still be virtually split.
    unpadded = dataclasses.replace(draft, page_size_padded=None)
    cache.kv_cache_groups = [allocator["KVCacheGroupSpec"](["draft.0"], unpadded)]
    assert select([64]) == [64]
    # Even zero extra padding selects the strided reshape when the field is set.
    explicit_stride = dataclasses.replace(
        draft, page_size_padded=draft.real_page_size_bytes
    )
    cache.kv_cache_groups = [
        allocator["KVCacheGroupSpec"](["draft.0"], explicit_stride)
    ]
    with pytest.raises(ValueError, match="cannot be split"):
        select([64])


def test_patch_is_idempotent_and_preflights_both_files(sources):
    kv = sources / "core/kv_cache_utils.py"
    worker = sources / "worker/utils.py"
    original = (kv.read_bytes(), worker.read_bytes())
    patch.patch_file(str(kv), dry_run=True)
    assert (kv.read_bytes(), worker.read_bytes()) == original
    worker.write_text(worker.read_text().replace(
        "selected_kernel_size = select_common_block_size(",
        "selected_kernel_size = changed_selector(",
    ))
    drifted = worker.read_bytes()
    with pytest.raises(AssertionError):
        patch.patch_file(str(kv))
    assert kv.read_bytes() == original[0]
    assert worker.read_bytes() == drifted
    worker.write_bytes(original[1])
    patch.patch_file(str(kv))
    applied = (kv.read_bytes(), worker.read_bytes())
    patch.patch_file(str(kv))
    assert (kv.read_bytes(), worker.read_bytes()) == applied
