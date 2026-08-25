# ArcticInference Rebase Report: vLLM `0.18.0` → `0.24.0`

Generated to support planning the rebase of the ArcticInference vLLM plugin.
All findings below were produced by diffing the two vLLM tags directly.

- **ArcticInference**: `snowflakedb/ArcticInference` @ `main` (`9d46438`), version `0.2.1.dev0`, currently pinning `vllm==0.18.0`.
- **vLLM (current pin)**: `v0.18.0` (local `vllm/` working tree).
- **vLLM (rebase target)**: `v0.24.0` (fetched; confirmed to exist upstream).

## TL;DR

The rebase is **much smaller than the version jump suggests.** Every vLLM module path
and symbol the plugin imports still exists at `v0.24.0` — **zero import relocations.**
Only **three** signatures that the plugin actually overrides/wraps changed, and only
**one** of them is a genuine break. The real effort is **behavioral drift** inside a
handful of high-churn files (above all `gpu_model_runner.py`), where signatures are
identical but bodies changed substantially and the monkey-patch overrides re-implement
large chunks of upstream logic.

### Online availability

- vLLM `v0.24.0` exists upstream (tag confirmed).
- **No public ArcticInference branch targets 0.24.0** (or 0.18.0). The newest
  version-pinned public branch is `corvo_vllm_0.14.1`; `main` carries the 0.18.0 line.
  Per-version branches present: `0.8.4, 0.9.0, 0.9.2, 0.10.1, 0.10.2, 0.11.0, 0.14.1`.
  → The `0.18 → 0.24` rebase is net-new work; there is nothing public to cherry-pick from.
- (Internal remote `snowflake-eng/arcticinference-internal` not checked, per request.)

---

## 1. How the plugin hooks into vLLM (context for the rebase)

- Registered as a vLLM **general plugin** via entry point in `pyproject.toml`:
  `arctic_inference = "arctic_inference.vllm.plugin:arctic_inference_plugin"`.
- On load (`ARCTIC_INFERENCE_ENABLED=1`), `plugin.py` performs a **hard version check**:
  it reads the pinned version from its own package metadata (`vllm==<X>; extra == "vllm"`
  in `pyproject.toml`) and raises unless `vllm.__version__` matches exactly.
- It then calls `apply_arctic_patches()`, which applies a set of `ArcticPatch[Target]`
  monkey-patches. `ArcticPatch` (`arctic_inference/patching.py`) splices attributes/methods
  directly onto the live vLLM class/module, capturing the originals as `_orig_*` for
  delegation.

**Mandatory rebase steps (independent of any API change):**
- `pyproject.toml` → `[project.optional-dependencies] vllm = ['vllm==0.24.0']`.
- **Bump `torch` in lock-step** — `[build-system].requires` currently pins `torch==2.10.0`,
  but **vLLM `v0.24.0` requires `torch==2.11.0`** (verified from `v0.24.0`'s
  `requirements/common.txt`). This must move together with the vLLM pin. Confirm with:
  `uv pip install vllm==0.24.0 --dry-run 2>&1 | grep '^ + torch'`.
  A `torch` minor bump also changes the `flash-attn` wheel ABI tag (`cu12torch2.11…`) used
  by the downstream corvo image, and — if you touch `csrc/` — forces a fresh wheel build
  (the `[build-system]` `torch` pin exists precisely so the CUDA ops compile against the
  right ABI).
- Update the `protobuf==` comment line if vLLM's grpc-tools floor moved.
- The `embedding` extra also pins `vllm==0.9.2` — decide whether that path is still supported.
- Because the check is exact-match, nothing loads until the pin is bumped; you can iterate
  faster by setting `ARCTIC_INFERENCE_SKIP_VERSION_CHECK=1` during development (dev only —
  never in a `vllmd_mainstream` deployment, or you get unintelligible tracebacks deep inside
  patched code).

> This report is scoped to the **code rebase**. Per the internal rebase playbook (Ye_skills
> Skill 3), the full migration also has a deployment tail: forward-port internal→`public`→
> `snowflakedb/ArcticInference` main, then a coordinated corvo `vllmd_mainstream` Dockerfile +
> `vllm-releases.yaml` bump. Budget ~1 week of focused work for a minor-crossing rebase.

---

## 2. Import-surface stability matrix (verified)

Every path/symbol imported by `arctic_inference/vllm/**` was checked for existence at
both tags. **All present at `v0.24.0`.** No `from vllm...import` line needs a path change.

Notable points that looked risky but are FINE:
- `from vllm.model_executor.layers.attention import Attention` → still re-exported by the
  `attention/` package `__init__.py` in both tags.
- `from vllm.model_executor.kernels.linear import init_fp8_linear_kernel` → still exported
  by the `kernels/linear/` package `__init__.py` in both tags.
- `vllm.v1.attention.backend` (`CommonAttentionMetadata`, `AttentionType`),
  `vllm.utils.math_utils`, `vllm.utils.platform_utils`, `vllm.utils.argparse_utils`,
  `vllm.config.{compilation,cache,utils}` — all stable.

> Caveat: existence ≠ identical behavior. Presence of a symbol only rules out `ImportError`;
> semantic changes are covered in §3–§4.

---

## 3. Confirmed signature changes on patched/overridden methods

These are the only signature-level diffs among the exact upstream methods the plugin
re-binds (`_orig_*` targets). Ordered by risk.

### 3.1 `GPUModelRunner._bookkeeping_sync` — **BREAKING (HIGH)**
File: `vllm/v1/worker/gpu_model_runner.py`. Arctic overrides this in
`arctic_inference/vllm/model_runner.py` (`_orig_bookkeeping_sync`).

- **v0.18.0**: takes a trailing `spec_decode_metadata: SpecDecodeMetadata | None` param;
  returns a tuple starting `(dict[str,int], LogprobsLists|None, list[list[int]], dict[...], ...)`.
- **v0.24.0**: the `spec_decode_metadata` parameter is **removed**, and the **return tuple
  gained an extra `list[str]` field**.

Action: re-derive the Arctic `_bookkeeping_sync` override against the new signature and
return arity. This is the single most important correctness item.

### 3.2 `GPUModelRunner.initialize_kv_cache` — **BREAKING (LOW–MED)**
Arctic overrides it (`_orig_initialize_kv_cache`).

- **v0.18.0**: `initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None`
- **v0.24.0**: `initialize_kv_cache(self, kv_cache_config: KVCacheConfig, is_profiling: bool = False) -> None`

Action: update the Arctic override to accept and forward `is_profiling` to
`_orig_initialize_kv_cache`. Low risk because the new param is optional.

### 3.3 `AsyncGPUModelRunnerOutput.__init__` — **WATCH (LOW)**
Not directly patched, but the `AsyncSchedulerPatch` reads `model_runner_output`.

- **v0.24.0** adds `routed_experts: RoutedExpertsTensors | None = None` (new MoE routing
  output plumbing). No Arctic call constructs this object, so likely no code change — but
  verify the async spec-decode path still receives the fields Arctic reads
  (`sampled_token_ids`, `req_id_to_index`, and the Arctic-added `_actual_draft_lens`).

### Everything else = IDENTICAL signatures
The following overridden/wrapped methods have **byte-identical signatures** across tags
(bodies may still differ — see §4):
`GPUModelRunner.__init__` (`self, vllm_config, device`), `.profile_run`, `.load_model`,
`.propose_draft_token_ids`, `._dummy_run`, `._build_attention_metadata`, `.execute_model`,
`.sample_tokens`, `.reload_weights`, `._capture_cudagraphs`;
`EngineArgs.{add_cli_args, from_cli_args, create_engine_config}`;
`SpeculativeConfig.__post_init__`; `VllmConfig._set_cudagraph_sizes`;
`MLPSpeculatorConfig.__init__`;
`ModelConfig.{get_num_kv_heads, get_num_attention_heads, get_layers_start_end_indices}`;
`parallel_state.{initialize_model_parallel, graph_capture}`;
`CompilationConfig.post_init_cudagraph_sizes`;
`EngineCore.{post_step, step_with_batch_queue}`;
`SpecDecodingStats.observe_draft`; `XgrammarBackend.__post_init__`;
`LogitsProcessor.{_get_logits, get_top_tokens}`;
`AsyncScheduler._update_after_schedule`; `Scheduler.{update_from_output, _update_after_schedule}`;
`Worker.{sleep, wake_up}`; `WorkerBase.__init__`;
`Attention.{__init__, forward}`; `CudagraphDispatcher.initialize_cudagraph_keys`.

Scheduler/request attributes the `AsyncSchedulerPatch` depends on all still exist at
v0.24.0: `num_spec_tokens`, `scheduled_spec_decode_tokens`, `pending_structured_output_tokens`,
`num_output_placeholders`, `is_prefill_chunk`, `use_structured_output`, `spec_token_ids`.

---

## 4. Behavioral-drift risk by file (churn, `v0.18.0 → v0.24.0`)

Signature-stable but heavily rewritten upstream files. These are where a patch that
re-implements or wraps large chunks of upstream logic can silently diverge. Sorted by risk.

| vLLM file | Δ lines | Arctic patch(es) touching it | Risk |
|---|---:|---|---|
| `v1/worker/gpu_model_runner.py` | ~2417 | `GPUModelRunnerPatch` (12 overrides + many new methods) | **CRITICAL** |
| `model_executor/kernels/linear/__init__.py` | ~704 | `spec_dec/fp8.py` (`init_fp8_linear_kernel`) | HIGH |
| `v1/core/sched/scheduler.py` | ~717 | `AsyncSchedulerPatch` (calls `Scheduler.update_from_output`, `_update_after_schedule`) | HIGH |
| `engine/arg_utils.py` | ~621 | `EngineArgsPatch`, `AsyncEngineArgsPatch` | HIGH |
| `model_executor/layers/quantization/fp8.py` | ~562 (mostly deletions) | `spec_dec/fp8.py` (`Fp8MoEMethod`) | HIGH |
| `v1/kv_cache_interface.py` | ~527 | attention backends (`AttentionSpec`) | MED |
| `config/compilation.py` | ~447 | `UlyssesCompilationConfig.post_init_cudagraph_sizes` | MED |
| `config/speculative.py` | ~439 | `SpeculativeConfigPatch.__post_init__` | MED |
| `v1/engine/core.py` | ~497 | `UlyssesEngineCore.{post_step, step_with_batch_queue}` | MED |
| `config/parallel.py` | ~405 | `ParallelConfigPatch` | MED |
| `config/model.py` | ~333 | `UlyssesModelConfig` (kv/attn head counts, layer indices) | MED |
| `model_executor/layers/attention/attention.py` | ~333 | `UlyssesAttention.{__init__, forward}` | MED |
| `distributed/parallel_state.py` | ~372 | `UlyssesParallelState.{initialize_model_parallel, graph_capture}` | MED |
| `v1/attention/backends/flash_attn.py` | ~315 | Forest Cascade attention backend | MED |
| `v1/worker/gpu_worker.py` | ~434 | `WorkerPatch.{sleep, wake_up}` | MED |
| `v1/executor/multiproc_executor.py` | ~232 | `UlyssesWorkerProc.shutdown`, `UlyssesMultiprocExecutor` | LOW-MED |
| `v1/sample/rejection_sampler.py` | ~177 | `model_runner.py` (`MAX_SPEC_LEN`, `RejectionSampler`) | LOW-MED |
| `v1/attention/backend.py` | ~159 | Forest Cascade backend base classes | LOW-MED |
| `v1/spec_decode/metrics.py` | ~168 | `SpecDecodingStatsPatch`, `SpecDecodingLoggingPatch` | LOW-MED |
| `v1/worker/gpu_input_batch.py` | ~165 | `arctic_proposer.py` (`CachedRequestState`, `InputBatch`) | LOW-MED |
| `v1/outputs.py` | ~146 | reads `ModelRunnerOutput`, `SamplerOutput` | LOW |
| `config/cache.py` | ~128 | `CacheDType` | LOW |
| `v1/structured_output/utils.py` | ~100 | `apply_grammar_bitmask` | LOW |
| `v1/attention/ops/merge_attn_states.py` | ~80 | Forest Cascade | LOW |
| `v1/attention/backends/fa_utils.py` | ~61 | Forest Cascade | LOW |
| `v1/attention/ops/common.py` | ~51 | Forest Cascade (`cp_lse_ag_out_rs`) | LOW |
| `model_executor/models/llama.py` | ~36 | `swiftkv/llama_swiftkv.py` (subclasses Llama layers) | LOW |
| `v1/cudagraph_dispatcher.py` | ~20 | `UlyssesCudagraphDispatcher` | LOW |
| `config/__init__.py` | ~12 | re-exports | LOW |
| `v1/structured_output/backend_xgrammar.py` | ~13 | `XgrammarBackendPatch` | LOW |
| `model_executor/layers/logits_processor.py` | ~6 | `fp32_lm_head.py` | LOW |

---

## 5. Patch-by-patch punch-list

For each `ArcticPatch`, what it re-binds and what to re-validate against v0.24.0.

### `arctic_inference/vllm/patches.py`
- `AsyncSchedulerPatch` → `AsyncScheduler._update_after_schedule`, wraps
  `Scheduler.update_from_output`. Signatures identical; **re-verify scheduler body logic**
  (placeholder allocation, `scheduled_spec_decode_tokens`, `num_output_placeholders`) against
  the ~717-line scheduler churn.
- `WorkerBasePatch.__init__`, `WorkerPatch.{sleep, wake_up}` → signatures identical; verify
  the `reload_weights()` / CuMemAllocator sleep-level-2 flow still holds (gpu_worker churn ~434).

### `arctic_inference/vllm/model_runner.py` (`GPUModelRunnerPatch`) — **biggest item**
Re-binds 12 originals: `__init__, profile_run, load_model, propose_draft_token_ids,
_dummy_run, _build_attention_metadata, execute_model, _bookkeeping_sync, sample_tokens,
initialize_kv_cache, reload_weights, _capture_cudagraphs` plus many new methods
(suffix decoding, shift-parallel cudagraph tables, ulysses forward).
- **`_bookkeeping_sync`**: rewrite for new signature/return (see §3.1).
- **`initialize_kv_cache`**: add/forward `is_profiling` (see §3.2).
- All others signature-stable → **diff bodies** against the ~2417-line churn, especially
  `execute_model`, `_dummy_run`, `propose_draft_token_ids`, `sample_tokens`, `_capture_cudagraphs`.

### `arctic_inference/vllm/args.py`
- `EngineArgsPatch` / `AsyncEngineArgsPatch` → `add_cli_args, from_cli_args,
  create_engine_config, __post_init__`. Signatures identical; re-check new/renamed engine
  args added in 0.19–0.24 (arg_utils churn ~621) don't collide with Arctic-added flags
  (e.g. `ulysses_sequence_parallel_size`, `enable_shift_parallel`, `forest_cascade_attn_configs`,
  `fp32_lm_head`).

### `arctic_inference/vllm/config.py`
- `ParallelConfigPatch`, `SpeculativeConfigPatch.__post_init__`, `VllmConfigPatch.{__str__,
  __post_init__}`, `MLPSpeculatorConfigPatch.__init__`. Signatures identical; verify against
  `config/{parallel,speculative}.py` churn (esp. new speculative fields / validation).

### `arctic_inference/vllm/ulysses.py`
- Patches `ModelConfig`, `parallel_state`, `WorkerProc`, `MultiprocExecutor`, `Attention`,
  `CudagraphDispatcher`, `CompilationConfig`, `VllmConfig`, `EngineCore`. All signature-stable;
  medium body churn in `config/model.py`, `parallel_state.py`, `compilation.py`, `engine/core.py`,
  `attention.py` → re-validate cudagraph size generation and shift-parallel group setup.

### `arctic_inference/vllm/attention/` (Forest Cascade)
- Subclasses `FlashAttentionBackend` and uses `v1/attention/backend.py`, `fa_utils`,
  `ops/{common,merge_attn_states}`, `kv_cache_interface.AttentionSpec`. Signatures present;
  moderate churn — re-check the backend metadata-builder interface and `AttentionSpec` fields.

### `arctic_inference/vllm/spec_dec/fp8.py`
- Patches `Fp8MoEMethod` and uses `init_fp8_linear_kernel`. **Highest hidden risk after
  gpu_model_runner**: `fp8.py` had ~562 lines removed and `kernels/linear` ~704 changed
  upstream — likely a refactor of the fp8 MoE path. Expect real work here.

### `arctic_inference/vllm/stats.py`, `structured_output.py`, `fp32_lm_head.py`, `swiftkv/`
- Low churn in their targets; signature-stable. Quick re-verify only.

### `arctic_inference/server/worker.py` (outside `vllm/`, but in scope)
- Calls `vllm.plugins.load_general_plugins()` explicitly (line ~139) to bootstrap the plugin
  inside the server worker. The internal 0.18.0 rebase (PR #39) had to add/adjust this call
  when vLLM changed its plugin/worker bootstrap. Re-verify the bootstrap path still fires
  before any patched class is imported. (Note: the playbook also lists a `vllm/plugins.py`
  wrapper, but that file does **not** exist in this checkout — only `plugin.py`.)

### Do **not** touch these (version-independent)
Per Skill 3 §3.6, don't re-derive: `arctic_inference/utils.py::get_compatible_vllm_version`,
`arctic_inference/envs.py` (unless adding new env vars), `arctic_inference/patching.py`
(the `ArcticPatch[T]` machinery — editing it means you're overreaching), and the
`common/swiftkv/`, `dynasor/`, `suffix_decoding/`, `op_builder/` packages.

### Recurring upstream churn patterns to watch (from prior rebases)
These bit earlier rebases and are the usual cause of import/attribute errors — most were
already absorbed at the 0.18.0 baseline, but re-check any that resurface across 0.19–0.24:
- **`vllm.utils` module split** — utilities keep moving into `math_utils` / `network_utils` /
  `system_utils` / `argparse_utils` submodules.
- **Pydantic `@dataclass` → `@config(config=ConfigDict(extra="forbid"))`** for `ParallelConfig`
  / `SpeculativeConfig` (0.18+), which also forced `world_size` from a `@property` to a field
  set in `__post_init__`. A fresh `AttributeError: ... has no attribute 'world_size'` is this.
- **`EngineArgs.from_cli_args`** shape: plain classmethod now, invoke via `.__func__` (not the
  old `.__dict__[...].__wrapped__`).
- **V0→V1 attention moves** and `AttentionType`/`CommonAttentionMetadata` living in
  `vllm.v1.attention.backend`.
- **FlashInfer optional import** must catch both `ImportError` and `RuntimeError`.
- **CUDA-graph capture** via a group's `CudaCommunicator` rather than
  `parallel_state._TP.graph_capture()`.

---

## 6. Suggested rebase order (dependency-aware)

1. **Bump the pins** — `vllm==0.24.0` and `torch==2.11.0` in `pyproject.toml` (see §1);
   dev-set `ARCTIC_INFERENCE_SKIP_VERSION_CHECK=1`.
2. **Chase import/attribute errors first.** Surface them all at once:
   ```bash
   ARCTIC_INFERENCE_ENABLED=1 \
     python -c "import arctic_inference.vllm.patches as p; p.apply_arctic_patches()"
   ```
   Each failure points at the file to touch. Prefer real fixes over `getattr(..., default)`
   shims — those silently mask genuine API changes.
3. **`args.py` + `config.py`** — get the engine to build (arg/config plumbing gates everything).
4. **`model_runner.py`** — the critical path; fix `_bookkeeping_sync` (§3.1) and
   `initialize_kv_cache` (§3.2) first, then re-diff the signature-stable overrides.
5. **`patches.py` scheduler/worker** — re-validate scheduler + sleep/wake flows.
6. **`ulysses.py`** — shift/sequence parallel, once the core runner is green.
7. **`attention/` Forest Cascade** and **`spec_dec/fp8.py`** — most upstream refactor churn.
8. **stats / structured_output / fp32_lm_head / swiftkv / server/worker.py** — cleanups.
9. Re-enable the version check; run the validation checklist (§7).

## 7. Validation checklist (per Skill 3 §3.7)

- [ ] `pyproject.toml` `vllm==0.24.0` **and** `torch==2.11.0` updated and consistent.
- [ ] `ARCTIC_INFERENCE_ENABLED=1 python -c "import arctic_inference.vllm.patches as p; p.apply_arctic_patches()"` runs clean.
- [ ] Three offline smoke examples pass (this hits ~80% of the prod surface):
  ```bash
  python projects/spec_dec/offline_inference_spec_dec.py   # Arctic + suffix proposers, model_runner
  python projects/swiftkv/offline_inference_swiftkv.py      # swiftkv layers, model_runner
  python projects/ulysses/offline_inference_ulysses.py      # ulysses + parallel_state patches
  ```
- [ ] `pytest tests/unit_tests -x` green.
- [ ] `pytest tests/benchmarks -x` green (needs a CUDA box, real models).
- [ ] `pytest tests/weight_sync -x` green (matters for the RL path).
- [ ] Your local `bench_fca.py` runs (exercises FCA + Arctic spec-dec + ulysses SP together).
- [ ] No new `getattr(..., default)` shims masking a real API change.
- [ ] `csrc/` rebuild ok if any `.cu`/`.cuh` changed.

---

## 8. Reproduce / drill deeper

From `spec_dec/vllm/` (tag `v0.24.0` already fetched):

```bash
# Full diff of a patched file
git diff v0.18.0 v0.24.0 -- vllm/v1/worker/gpu_model_runner.py

# Signature of a specific method at each tag
git show v0.24.0:vllm/v1/worker/gpu_model_runner.py | grep -n "def _bookkeeping_sync" 

# Churn overview for any set of files
git diff --stat v0.18.0 v0.24.0 -- <paths...>
```

To scope a specific patch: cross-reference the `_orig_*` assignments in the corresponding
`arctic_inference/vllm/*.py` file against the upstream method body at `v0.24.0`.

---

## Sources

- Direct `git diff`/`git show` of vLLM tags `v0.18.0` ↔ `v0.24.0` (local `vllm/` clone).
- Import surface extracted from `arctic_inference/vllm/**` and `arctic_inference/server/worker.py`.
- Rebase process, "always-touched" file ranking, recurring-pattern list, deployment tail, and
  validation checklist cross-checked against internal `Ye_skills.md` **Skill 3** ("Upgrade
  ArcticInference to a new vLLM version"). `torch==2.11.0` requirement verified from
  `v0.24.0:requirements/common.txt`.
