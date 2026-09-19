# NVMe-direct KV prefix cache with a model-keyed fence
`overlay/kvoffload/nvme_direct2.py` + `overlay/patch_kv_offload_groups.py` +
the `OFFLOAD_NVME=1` arm in `start.sh`. Lab boot on the 2× DGX Spark pair
(GB10, 128 GB UMA); a fenced rewrite of upstream #232 on top of #230
(`lab/cold-load` a0c13b8).

## The shape
One file per KV block (`<ns>/r<rank>/<h[:3]>/<h>.bin`), `blocks_per_chunk=1`, a
chunk-sized pinned bounce buffer per IO thread, event-fenced D2H → `fsync` →
atomic rename. No CPU tier — GPU↔file directly, because GB10 has no spare host
tier (#57's crash). Files outlive the boot: the point is restore-after-restart.
Measured here on a 105,222-token prompt: store 6.2 GB ≈ 43 s (~150 MB/s
effective), restore 1.68 GB in **5.5 s** vs 130 s recompute (~23×), bit-exact
marker ids. A 6×168k flood reached 76.5 GB stored with min head-avail 4.6 GB
and zero capacity stalls.

## The fence (the part #232 did not have)
Keys are `hash(token_ids)` — model-blind. Without a namespace gate, a boot of
different weights that sees a stored prompt replays *another model's KV bytes*
(#57 failure-mode 4: confidently wrong output). So the root partitions by
sha256 over everything that changes the meaning of a stored block:

    { v, model, revision, kv_bytes_per_block, tokens_per_hash, blocks_per_file,
      tp/pp/pcp/dcp_size, groups:[{tokens_per_block, layer_names}],
      engine (vllm.__version__), stamp (GLM53_RECIPE_STAMP), format }

and writes `config.json` beside the data — **read back at init**: a
sidecar-less data dir or a disagreeing sidecar refuses to boot (fail closed).
Upstream's `FileMapper` has the same layout idea but nothing ever reads its
config.json; this is that half, finished.

Field notes, learned on this fork:

- **`revision`** comes from `kv_connector_extra_config.model_revision` (the
  launcher passes `$MODEL_REVISION`, overridable per boot with
  `OFFLOAD_NVME_REVISION`); fallback is an offline HF-cache resolve
  (`try_to_load_from_cache`); no answer ⇒ the `unresolved` namespace — cold,
  never merged.
- **`model.dtype` is NOT in the fence.** It reports the *params* dtype and is
  process-dependent on this fork (`fp8_ds_mla` in the EngineCore, `fp8` in the
  workers): one boot split into two namespaces until replaced by
  `kv_bytes_per_block`, derived from the broadcast `kv_cache_config` and
  identical in every process. It pins KV dtype + geometry, which is what the
  stored bytes actually mean.
- **`stamp` fired for real**: after an image rebuild baked a new recipe stamp,
  the next boot opened a fresh namespace and recomputed (load 0, ids still
  match) instead of replaying stale-KV-shaped bytes. That's the engine-side
  axis the field exists for.
- Cross-boot key stability additionally needs `PYTHONHASHSEED=0` (the
  `NONE_HASH` chain root is seed-random otherwise) and `expandable_segments`
  off — both enforced by the launcher arm. `GLM53_APC_NO_STORE` (the box
  default) is compatible by design: it suppresses the *GPU* prefix-hash
  insertion only, the connector's `block_hashes` still arrive intact on the
  store path. The arm's original conflict check against it was a false
  positive; removed.

## Contracts (crash-shaped lessons)
- `get_stats()` must return `OffloadingConnectorStats` or `None` — a dict
  crashes the first engine step (`AttributeError: is_empty`). Inherit the base.
- `TransferResult.transfer_time` must never be `None` — the connector records
  metrics only when size *and* time are set, and the gate counters depend on
  that.
- The worker `df` line keeps its "Avail" header column — pick numerics, not
  fields.

## Gates run (2026-09-19, this pair)
| gate | result |
|---|---|
| unit (fence sensitivity per field, sidecar protocol, short-write, HIT/recheck) | 24 pass in-image |
| in-boot restore | 1.68 GB / 5.5 s, ids match |
| cross-boot persistence (boot 1 stores, boot 2 restores, `store_delta 0`) | pass |
| fence gate (`OFFLOAD_NVME_REVISION=deadbeef…`) | new ns beside real one, load 0, full recompute, ids match — #57 mode 4 closed |
| stamp gate (image rebuild → new stamp) | cold recompute, old tree untouched |
| flood headroom | 76.5 GB stored, min avail 4.6 GB head / 6.6 GB worker, no stalls |

## Running it
`OFFLOAD_NVME=1` in `.env` (commented by default), then `./start.sh`. The arm
stages the spec on both nodes (fs + free-space gates), bind-mounts the root,
sets the env laws, and injects the connector JSON via `EXTRA_ARGS`. Capacity
`NVME_CAPACITY` (default 500 GiB). To go back: comment the line, `./start.sh
stop && ./start.sh start`.

Related: upstream #230 (cold-load, taken), #231 (benches), #232 (the worker
this rewrite is based on), PR #58's CPU-tier ladder in docs/apc-retention-
qualification.md.
