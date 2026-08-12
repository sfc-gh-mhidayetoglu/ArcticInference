# REBASE_TEST.md — ArcticInference vLLM 0.24.0 → 0.26.0 (GPU validation runbook)

Branch: `rebase/vllm_v26` (based off `rebase/vllm_v24`). Target pin: `vllm==0.26.0`.

**Status:** GPU validation (Phase 6 of the skill) **complete** on an 8×H200 node
(env `<venv>`, vllm 0.26.0, torch 2.11.0+cu130, CUDA 13.0). Phases
1–5, 7, 9, 10 **PASS** at runtime; Phase 6 (SwiftKV) is **deprecated / unsupported on
0.26** — unmaintained since vLLM 0.14.1; runtime-tested here and found to **produce
incorrect output even in eager mode** (custom bulk KV kernel not ported to the new packed
layout) plus a separate cudagraph-capture blocker (see Phase 6); Phase 8 (weight-sync)
API/import validated, full runtime deferred (needs HF-downloadable models + 2-proc RL harness).
The static no-GPU scan was the floor — the runtime loop below surfaced **three new
runtime-only breaks** (all fixed) past engine init, into first prefill/decode/step.

### New runtime-only breaks found + fixed (0.24→0.26, this loop)
| Where | Symptom (runtime) | Fix | reference.md |
|-------|-------------------|-----|--------------|
| `vllm/args.py` `create_engine_config` | `AttributeError: 'EngineArgs' object has no attribute 'ulysses_sequence_parallel_size'` at first engine build via `LLM(...)` | plugin-load timing: first `EngineArgs(...)` built before patches apply, so `__new__` never upgraded it to `ArcticEngineArgs`. Added `_ensure_arctic_fields(self)` (backfill `ArcticArgs` defaults) at top of both sync/async `create_engine_config`. | pattern #11 (new) |
| `vllm/ulysses.py` `graph_capture` (module fn) | `TypeError: graph_capture() got an unexpected keyword argument 'graph_capture_context'` at engine-init cudagraph mem profiling | pattern #1: base free fn gained `graph_capture_context=None`; added kwarg + `context = graph_capture_context or GraphCaptureContext(...)`. | pattern #1 (extended) |
| `vllm/ulysses.py` `step_with_batch_queue` | `AttributeError: 'EngineCoreProc' object has no attribute 'log_iteration_details'` at first engine step | re-derived the mirrored method from base 0.26: `capture_iteration_details(...) as x` + `_attach_iteration_details(...)`, `schedule(self._should_throttle_prefills())`, `log_error_detail` around execute, `check_for_draft_tokens`+`if` (was `use_spec_decode`+assert), new non-block return. | pattern #12 (new) |
| `vllm/swiftkv/llama_swiftkv.py` (SwiftKV, **deprecated since 0.14.1**) | runtime-tested with Llama-3.1-SwiftKV-8B: (a) OOB rope index from uninitialized `positions` under 0.26 `profile_cudagraph_memory`; (b) `ForwardContext.virtual_engine` removed; (c) bulk op CPU→GPU copy illegal during capture-in-profiling; (d) **wrong output even in eager mode** — `reshape_and_cache_flash_bulk.cu` index math is for the old KV layout | (a) `positions` `torch.empty`→`torch.zeros`; (b) `attn.kv_cache[virtual_engine]`→`attn.kv_cache`; plus the drift #8 FA KV split (both branches). **(c)+(d) left unfixed — (d) needs a CUDA bulk-kernel rewrite for the 0.26 packed/interleaved layout; SwiftKV unsupported on 0.26.** | drift #8 (extended) |

## Setup on the GPU node (env `<venv>`, actually used)
```bash
source <venv>/bin/activate          # venv on the dev conda base
export ARCTIC_INFERENCE_ENABLED=1
# vllm==0.26.0, torch 2.11.0+cu130 preinstalled. Extra deps this env needed:
pip install 'nanobind==2.9.2' pytest ray          # build dep + test deps
# CUDA 13 build fix (committed): csrc/custom_ops/CMakeLists.txt TORCH_CUDA_ARCH_LIST
#   dropped sm_70 (nvcc 13 rejects compute_70): "7.5 8.0 8.6 8.9 9.0+PTX".
ARCTIC_INFERENCE_PRECOMPILED_OPS=1 ARCTIC_MINIMAL_BUILD=1 \
  pip install -e . --no-build-isolation --no-deps   # builds suffix_decoding + custom_ops (cp312)
```
Notes: the pin already matches (`vllm==0.26.0`) so **no** `ARCTIC_INFERENCE_SKIP_VERSION_CHECK`
is needed. Remove any stale `arctic_inference.egg-info` and stale cp310 `.so` files first.
No-GPU gate in the GPU env:
```bash
python -c "import vllm; vllm.plugins.load_general_plugins()"   # plugin binds ✓
pytest tests/unit_tests --collect-only                          # 44 collected, clean ✓
```

## Code changes in this rebase (map change → test)
| # | File | Change | Targeted test / what to watch |
|---|------|--------|-------------------------------|
| 1 | `pyproject.toml` | `vllm==0.24.0 → 0.26.0` (torch 2.11.0 / protobuf 5.29.6 unchanged) | version check passes with `vllm==0.26.0`; import smoke |
| 2 | `vllm/model_runner.py` | `_capture_cudagraphs` gained `profiler=None`, forwarded to both `_orig_capture_cudagraphs` calls (base caller now passes `profiler=` kw) | **cudagraph capture at engine init** with `-O '{"cudagraph_mode":...}'`; base + shift capture must complete, no `TypeError: unexpected keyword 'profiler'` |
| 3 | `vllm/attention/flash_attn_forest_cascade.py` | **KV-cache layout port (drift #8)**: 0.26 packs KV as `(num_blocks, num_kv_heads, block_size, 2*head_size)`. Ported backend `get_kv_cache_shape`/`get_kv_cache_stride_order`, and forward split `kv_cache.unbind(1)` → `kv_cache.transpose(1,2).split(head_size, dim=-1)` + `canonicalize_singleton_dim_strides`; added the import. FCA Impl is the **always-active** attention path. | **first prefill + first decode step** on any model: no `too many values to unpack` / shape/stride errors from the KV split; correctness (coherent output). Run **with FCA off (default)** AND **with `--forest-cascade-attn-configs '{}'`** so both the plain and cascade paths exercise the new split. **PASS** (Phases 1 + 7). |
| 4 | `csrc/custom_ops/CMakeLists.txt` | CUDA-13 build fix: `TORCH_CUDA_ARCH_LIST` dropped `7.0` (nvcc 13 rejects `compute_70`/Volta) → `"7.5 8.0 8.6 8.9 9.0+PTX"`. | `custom_ops` compiles for cp312; `try_load_torch_library()` → True; `test_sum_lstm`/`test_speculator_ln`/`test_reshape_and_cache_flash_nvfp4` pass. **PASS**. |
| 5 | `vllm/args.py`, `vllm/ulysses.py` (×2), `vllm/swiftkv/llama_swiftkv.py` | Three runtime-only breaks + SwiftKV KV port — see the "New runtime-only breaks" table above. | Phases 1/5/7/9/10 pass; **SwiftKV deprecated/unsupported on 0.26** (unmaintained since 0.14.1; one bulk-op blocker left unfixed). |

## Ordered GPU validation phases (smoke → features) — RESULTS
Validated with `Qwen3-1.7B` (dynamic `--quantization fp8`; on the V2 whitelist, so it
also confirms the `VLLM_USE_V2_MODEL_RUNNER=0` force). Arctic args (ulysses/shift/fca)
passed via `LLM(...)` require an explicit `vllm.plugins.load_general_plugins()` first
(pre-patch `EngineArgs` rejects them); the real user path is `vllm serve` (Phase 10).
1. **PASS — Smoke / engine init** (FP8, TP=1, cudagraphs FULL_AND_PIECEWISE): engine
   init + first prefill + first decode, coherent "Paris…". Watch #2 (profiler=) ✓,
   watch #3/KV split ✓, `VLLM_USE_V2_MODEL_RUNNER=0` ✓ (log says "V1 LLM engine").
   *(surfaced the 3 runtime breaks above)*
2. **PASS — CPU unit tests**: `test_patching`, `test_scheduler_prefix_routing`.
3. **PASS — GPU unit tests**: `test_fp32_lm_head`, `test_reshape_and_cache_flash_nvfp4`,
   `test_speculator_ln`, `test_sum_lstm` (36 passed), **plus** `test_custom_ops.py::
   test_reshape_and_cache_flash_bulk` (now passes). The latter had a stale **2D** cache
   fixture that 0.26's `reshape_and_cache_flash` rejects (reads `stride(2)`); rewrote it to
   the classic 4D `[num_blocks, block_size, num_heads, head_size]` layout (distinct slots via
   `randperm`). This validates the bulk kernel in isolation — both the vLLM reference op and
   Arctic's kernel agree on that contiguous layout. (Independent of SwiftKV BLOCKER-B, which
   is SwiftKV feeding the kernel 0.26's *packed* FA views — see Phase 6.)
4. **PASS — Spec decode**: `suffix`, `arctic` (Arctic-LSTM-Speculator-Qwen3-30B ckpt on
   the 2048-hidden Qwen3-1.7B base), and hybrid — all coherent. Watch first spec step
   (fp8 LM-head lazy build) ✓ no `fp8_linear is None`/config-not-set errors.
5. **PASS — Ulysses SP** (`ulysses_sequence_parallel_size=2`, TP=1, 2 GPUs): coherent
   output; exercises the `graph_capture(graph_capture_context=)` fix under real SP.
6. **UNSUPPORTED ON 0.26 (deprecated) — SwiftKV**: SwiftKV has been **unmaintained since
   vLLM 0.14.1** — commit `19c8eac` "Rebase to vLLM 0.14.1 (#242)" (2026-02-23) is the last
   commit that touches `arctic_inference/vllm/swiftkv/`; `main` then bumped 0.14.1→0.18.0
   (RL-integration merge #264) and 0.18→0.24→0.26 all carried SwiftKV forward untouched.
   Runtime-tested this loop with `Snowflake/Llama-3.1-SwiftKV-8B-Instruct` (BF16, TP=1) and
   the ~4-version gap surfaced four independent 0.26 breaks (2 fixed, 2 left as blockers):
   - **(fixed)** `decode_runner.inputs["positions"]` was `torch.empty` (uninitialized);
     0.26's new `profile_cudagraph_memory()` runs the decode forward while
     `attn_metadata is None`, so `swiftkv_select` returns the raw buffer and its garbage
     int64 positions index the rotary cache (size `max_position_embeddings`=131072) out of
     bounds → device-side assert. Fixed by zero-initializing the buffer (`torch.zeros`).
   - **(fixed)** `swiftkv_select` used `attn.kv_cache[forward_context.virtual_engine]`, but
     `ForwardContext.virtual_engine` was **removed in 0.26** (`attn.kv_cache` is now a single
     tensor per layer). Fixed → `kv_cache = attn.kv_cache` (drift #1/#7); dropped the now-
     unused `get_forward_context()` call.
   - **(BLOCKER-A, not fixed — cudagraph capture)** the bulk custom op
     `reshape_and_cache_flash_bulk` runs during 0.26's new capture-in-profiling `_dummy_run`
     (attn_metadata is non-None there) and does an internal `torch.from_blob(...).to(device)`
     CPU→GPU copy → `RuntimeError: Cannot copy between CPU and CUDA tensors during CUDA graph
     capture`. Needs a capture-safe (pinned + async-copy-on-stream / device-built) pointer
     array in the `.cu`, or a capture-skip guard in `swiftkv_select`.
   - **(BLOCKER-B, not fixed — WRONG OUTPUT, the hard one)** with `enforce_eager=True`
     (bypasses BLOCKER-A) the model **runs to completion but generates garbage** ("…Corm Corm
     Corm…"). Root cause is the KV-layout drift (#8) reaching into the **custom CUDA kernel**:
     0.26 packs KV as `(num_blocks, num_kv_heads, block_size, 2*head_size)` (head-major within
     a block, **K/V interleaved** in the last dim), but `reshape_and_cache_flash_bulk.cu` still
     computes `tgt_key_value_idx = block_idx*block_stride + block_offset*num_heads*head_size +
     head_idx*head_size + head_offset`, i.e. the **old** `(num_blocks, block_size, num_kv_heads,
     head_size)` separate-K/V layout. The Python-side `transpose(1,2).split(...)` gives correct
     *logical views* but the kernel does raw pointer math and writes KV into the wrong physical
     slots → attention reads corrupted KV → gibberish. **Fix requires rewriting the bulk
     kernel's index math (and re-validating the FA metadata fixups) for the new interleaved,
     head-major layout — a real CUDA change, ~½–1 day; full cudagraph parity ~1–2 days with
     tail risk.** Not pursued — SwiftKV is deprecated.
   The FA KV-layout port (0.26 packed split, both bulk + per-layer branches) plus the two
   init fixes above are kept for compile/import parity only. **SwiftKV runtime is unsupported
   and produces incorrect output on 0.26** — even in eager mode. Revive from the 0.14.1
   baseline (last maintained there) if it's ever needed again.
7. **PASS — Forest-cascade / FP8** (`forest_cascade_attn_configs="{}"`): FCA ENABLED,
   cudagraph_mode force-downgraded FULL_AND_PIECEWISE→PIECEWISE, PIECEWISE capture +
   coherent output — validates the ported KV split in the cascade path.
8. **PARTIAL — Weight sync**: `arctic_inference.server.weight_sync` imports clean on
   0.26; `WeightSyncExtension` API intact (`sync_spec_weights`/`sync_weights`/IPC). Full
   runtime deferred (test defaults to 70B via `snapshot_download` + 2-proc TP=2 harness).
9. **PASS — Shift parallel** (`enable_shift_parallel=True`, SP=2, cudagraphs): base +
   **shift** both captured in **PIECEWISE and FULL**, no "inappropriate time" error;
   TP path (small batch) and SP path (large batch) both coherent.
10. **PASS — Server** (`vllm serve`, CLI `from_cli_args`→`ArcticEngineArgs` path):
    boots, `/v1/models` + `/v1/completions` coherent ("Paris…").

## Latent gaps — status after GPU validation (still need sign-off to implement)
These remain config-gated / diverged; **not** force-ported. None were hit by the phases
above (all ran with DCP=1, DP=1, no sliding-window model). Get sign-off before porting.
- **`_dummy_run` — DCP (decode context parallel)**: still a mirror with no DCP code.
  Not exercised (all runs `decode_context_parallel_size=1`, a no-op). **Only port if
  DCP+Arctic is ever required.**
- **`sample_tokens` — DP>1 drafter hang guard**: **NOT tested** — the offline `LLM(...)`
  path hard-blocks `data_parallel_size>1` (single-process), and spec-decode phases ran
  DP=1. Still unguarded in Arctic's diverged `sample_tokens`. **Test spec-decode under
  DP>1 via the multi-proc launcher**; if it hangs, port the drafter dummy-run.
- **New 0.26 attention features not in the FCA fork** (`supports_sliding_window`, RSWA
  `_make_rswa_mask_mod`, `_maybe_symmetrize_window`): not hit — Qwen3-1.7B uses full
  attention. **If a model needs sliding-window/RSWA under the always-patched FCA
  backend, re-diff and port** (would surface as wrong outputs / mask errors at decode).
- **Drift #10 (V2 runner) — CONFIRMED intact at runtime**: `plugin.py` forces
  `VLLM_USE_V2_MODEL_RUNNER=0`; validated on `Qwen3ForCausalLM` (a V2-whitelist arch) —
  logs show "V1 LLM engine" and `os.environ['VLLM_USE_V2_MODEL_RUNNER']=='0'`. The
  carried-forward TODO to port Arctic patches to the V2 runner still stands.
- **SwiftKV runtime (Phase 6)**: **deprecated — unsupported on 0.26** (unmaintained since
  vLLM 0.14.1). Runtime-tested; two breaks fixed, one bulk-op capture blocker left unfixed
  by decision. Do **not** ship as working; revive from the 0.14.1 baseline if needed.
- **Weight-sync runtime (Phase 8)**: code/API validated but not run end-to-end offline —
  see phase notes. Run before shipping.

## Adjacent artifacts (bump separately if used)
- `benchmark/rollout/*.patch` (`sampling_params.patch`, `parallel_sampling.patch`): re-diff
  their target files against `/tmp/vllm_new` and regenerate if you use rollout replay.
