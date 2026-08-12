# ArcticInference: rebase plugin from vLLM 0.18.0 to 0.24.0

> Companion to [REBASE_REPORT.md](REBASE_REPORT.md). This is the actionable plan;
> the report is the underlying analysis (import matrix, signature diffs, churn table).

Branch `rebase/vllm_v24` is checked out and clean. All work happens in `ArcticInference/`.
vLLM `v0.24.0` is fetched in the local `vllm/` clone for side-by-side diffing.

**Scope:** Full plugin rebase (model_runner, scheduler/worker patches, args, config, ulysses,
Forest Cascade attention, spec-dec, swiftkv, structured output, fp32 lm-head, stats, server +
weight-sync, examples, tests).

**Stopping point for this pass:** all modules import and `apply_arctic_patches()` runs clean.
GPU/functional validation (offline examples, pytest, `bench_fca.py`) is a follow-up.

**Key upfront finding:** at v0.24.0 **every import path/symbol the plugin uses still resolves**
(report §2), so this is mostly a signature + behavioral-parity rebase, not an import-chasing
rebase. The dominant effort is `gpu_model_runner.py` (~2417 changed lines) and
`fp8.py`/`kernels/linear` (~1266 combined).

## Task checklist

- [ ] **pins** — Bump pyproject.toml: `vllm==0.24.0`, `torch==2.11.0`, refresh protobuf comment, decide embedding extra; set `ARCTIC_INFERENCE_SKIP_VERSION_CHECK=1` for dev
- [ ] **imports** — Run `apply_arctic_patches()` one-liner and clear any import/attribute errors (expected light)
- [ ] **sig-bookkeeping** — Fix model_runner `_bookkeeping_sync`: drop `spec_decode_metadata` param, re-source it via self stash from `execute_model`, handle new 7-tuple return
- [ ] **sig-kvcache** — Fix model_runner `initialize_kv_cache`: add and forward `is_profiling`; verify `AsyncGPUModelRunnerOutput` `routed_experts` doesn't break async spec path
- [ ] **behavior-runner** — Re-derive model_runner overrides (`execute_model`, `_dummy_run`, `propose_draft_token_ids`, `sample_tokens`, `_capture_cudagraphs`, `profile_run`, `load_model`, `reload_weights`) against 2417-line churn
- [ ] **behavior-sched-worker** — Re-verify patches.py `AsyncSchedulerPatch` and `WorkerPatch` sleep/wake against scheduler + gpu_worker churn
- [ ] **behavior-args-config** — Reconcile args.py and config.py against arg_utils/parallel/speculative churn
- [ ] **behavior-ulysses** — Reconcile ulysses.py against model/parallel_state/compilation/engine.core/attention churn
- [ ] **behavior-attn-fp8** — Reconcile Forest Cascade attention and spec_dec/fp8.py against flash_attn/backend/kv_cache_interface and fp8/kernels-linear churn
- [ ] **lower-risk** — Re-verify stats, structured_output, fp32_lm_head, swiftkv, arctic_proposer/speculator, vocab_parallel_embedding, logits_processor_opt
- [ ] **server-tests** — Check server/worker.py bootstrap, server metrics/weight_sync imports, and update projects examples + tests/unit_tests for changed signatures
- [ ] **validate** — apply_arctic_patches() clean, import-smoke all modules, `pytest --collect-only`, re-enable version check; document GPU follow-ups in PR

---

## Phase 0 - Pins and dev bypass
In [ArcticInference/pyproject.toml](ArcticInference/pyproject.toml):
- `vllm==0.18.0` -> `vllm==0.24.0` (line 53).
- `torch==2.10.0` -> `torch==2.11.0` (line 11) - required by `v0.24.0:requirements/common.txt` (verified). Re-confirm with `uv pip install vllm==0.24.0 --dry-run 2>&1 | grep '^ + torch'`.
- Refresh the `protobuf==` comment lines (8, 71) if the grpc-tools floor moved.
- Decide the `embedding` extra `vllm==0.9.2` (line 70): leave pinned (separate path) unless it must move too - recommend leaving as-is and noting it.
- Export `ARCTIC_INFERENCE_SKIP_VERSION_CHECK=1` for the whole dev loop; re-enable at the end.

## Phase 1 - Import/attribute triage (fast)
Iterate to a clean load:
```bash
ARCTIC_INFERENCE_ENABLED=1 python -c "import arctic_inference.vllm.patches as p; p.apply_arctic_patches()"
```
Expected to be light since paths resolve. Prefer real fixes over `getattr(..., default)` shims. Watch the known recurring patterns (report §5): pydantic `@config`/`world_size`, `EngineArgs.from_cli_args.__func__`, FlashInfer optional import, `CudaCommunicator` graph-capture.

## Phase 2 - Confirmed signature breaks
File: [ArcticInference/arctic_inference/vllm/model_runner.py](ArcticInference/arctic_inference/vllm/model_runner.py)
- `GPUModelRunnerPatch._bookkeeping_sync` (~line 793): base v0.24.0 **dropped** the trailing `spec_decode_metadata` param and now returns a **7-tuple** (added a `list[str]`). The Arctic override both reads `spec_decode_metadata` (for the async arctic/mlp/suffix stash) and forwards it. Fix: drop the param from the override signature, re-source `spec_decode_metadata` (stash it on `self` from the `execute_model` override before it calls `_bookkeeping_sync`), and call `self._orig_bookkeeping_sync(scheduler_output, sampler_output, logits, hidden_states, num_scheduled_tokens)` with the new arity.
- `initialize_kv_cache` (~line 1758): add `is_profiling: bool = False` and forward it to `_orig_initialize_kv_cache`.
- `AsyncGPUModelRunnerOutput.__init__` gained `routed_experts`; confirm `AsyncSchedulerPatch` still reads `sampled_token_ids`/`req_id_to_index` and the Arctic-added `_actual_draft_lens` unchanged.

## Phase 3 - Behavioral re-derivation (high-churn, signatures stable)
For each, diff the upstream body v0.18.0 vs v0.24.0 and reconcile the override:
- model_runner.py overrides: `execute_model`, `_dummy_run`, `propose_draft_token_ids`, `sample_tokens`, `_capture_cudagraphs`, `profile_run`, `load_model`, `reload_weights` (vs the 2417-line churn). This is the bulk of the work.
- [patches.py](ArcticInference/arctic_inference/vllm/patches.py) `AsyncSchedulerPatch` (placeholder allocation, `scheduled_spec_decode_tokens`, `num_output_placeholders`) vs scheduler ~717; and `WorkerPatch.sleep/wake_up` sleep-level-2 reload flow vs gpu_worker ~434.
- [args.py](ArcticInference/arctic_inference/vllm/args.py) vs arg_utils ~621: ensure Arctic-added flags (`ulysses_sequence_parallel_size`, `enable_shift_parallel`, `forest_cascade_attn_configs`, `fp32_lm_head`) still register cleanly.
- [config.py](ArcticInference/arctic_inference/vllm/config.py) vs parallel/speculative churn.
- [ulysses.py](ArcticInference/arctic_inference/vllm/ulysses.py) vs model/parallel_state/compilation/engine.core/attention churn (second-biggest source).
- [attention/flash_attn_forest_cascade.py](ArcticInference/arctic_inference/vllm/attention/flash_attn_forest_cascade.py) vs flash_attn/backend/kv_cache_interface churn.
- [spec_dec/fp8.py](ArcticInference/arctic_inference/vllm/spec_dec/fp8.py) vs fp8.py (~562, mostly deletions) + kernels/linear (~704) - highest hidden-risk refactor.

## Phase 4 - Lower-risk patches
Quick re-verify: `stats.py`, `structured_output.py`, `fp32_lm_head.py`, `swiftkv/llama_swiftkv.py`, `spec_dec/{arctic_proposer,arctic_speculator,vocab_parallel_embedding,logits_processor_opt}.py`. New proposer symbols (`eagle_prepare_next_token_padded_kernel`, `triton_utils.triton`) confirmed present at v0.24.0.

## Phase 5 - Server, weight-sync, examples, tests
- [server/worker.py](ArcticInference/arctic_inference/server/worker.py) `load_general_plugins()` bootstrap (line ~138); `server/metrics.py`, `server/weight_sync/{utils,receiver}.py` import checks (all paths currently resolve at v0.24.0).
- Update `projects/{spec_dec,swiftkv,ulysses}/offline_inference_*.py` and `tests/unit_tests/*.py` for any changed patched-API signatures.

## Phase 6 - Validation (no-GPU stopping point)
- `apply_arctic_patches()` runs clean with `ARCTIC_INFERENCE_ENABLED=1`.
- Import-smoke every touched module; `pytest tests/unit_tests --collect-only` (import-time errors surface without a GPU).
- Re-enable the exact-match version check (unset the skip env).
- Mark as follow-up (need CUDA box + `vllm==0.24.0` venv): the three offline examples, `pytest tests/{unit_tests,benchmarks,weight_sync}`, and your `bench_fca.py`.

## Out of scope (call out in PR, not done here)
Deployment tail from Ye_skills Skill 3: forward-port internal->public->`snowflakedb/ArcticInference`, corvo `vllmd_mainstream` Dockerfile + flash-attn ABI + `vllm-releases.yaml`. Not part of this branch.
