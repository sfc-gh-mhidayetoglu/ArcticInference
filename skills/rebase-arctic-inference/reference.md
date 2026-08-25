# ArcticInference rebase — module map & drift patterns

Distilled from the 0.18.0 → 0.24.0 rebase. Use as the per-module punch-list.
The vLLM churn magnitude changes each bump, but **which** plugin files sit on top
of **which** vLLM surfaces is stable — that mapping is the durable value here.

## Plugin architecture (how the patching works)
- Entry point `vllm.general_plugins → arctic_inference.vllm.plugin:arctic_inference_plugin`.
  vLLM calls it during `load_general_plugins()` when `ARCTIC_INFERENCE_ENABLED=1`.
- `plugin.py` enforces an **exact** `vllm==<pin>` match (bypass: `ARCTIC_INFERENCE_SKIP_VERSION_CHECK=1`).
  The pin is read from package metadata via `utils.get_compatible_vllm_version()`,
  which parses the `vllm==` line in `pyproject.toml`.
- `patching.py::ArcticPatch[Target]` splices attributes/methods onto a live vLLM
  class **or module**. Overrides that wrap the original keep a class-level alias
  `_orig_<name> = Target.<name>` captured at class-definition time. **Every `_orig_*`
  alias is a hard dependency on the base still having that attribute.**

## Module → vLLM surface map (patch targets)
| Plugin file | Patches (ArcticPatch targets) | vLLM surfaces that drive churn |
|-------------|-------------------------------|--------------------------------|
| `vllm/model_runner.py` | `GPUModelRunner` | `v1/worker/gpu_model_runner.py` — **highest churn every bump** |
| `vllm/patches.py` | `Scheduler`/`AsyncScheduler`, `Worker` | `v1/core/sched/scheduler.py`, `v1/worker/gpu_worker.py` |
| `vllm/args.py` | `EngineArgs`, `AsyncEngineArgs` | `engine/arg_utils.py` |
| `vllm/config.py` | `ParallelConfig`, `SpeculativeConfig`, `VllmConfig`, `MLPSpeculatorConfig` | `config/*.py` (pydantic `@config` dataclasses) |
| `vllm/ulysses.py` | `parallel_state` (module), `CudagraphDispatcher`, `WorkerProc`, `VllmConfig`, `EngineCore` | `distributed/parallel_state.py`, `compilation/*`, `v1/engine/core.py` |
| `vllm/attention/flash_attn_forest_cascade.py` | `FlashAttentionBackend/Impl/MetadataBuilder` (fork) | `v1/attention/backends/flash_attn.py` + `fa_utils.py` |
| `vllm/spec_dec/fp8.py` | `Fp8Config`, FP8 linear/MoE methods | `model_executor/layers/quantization/fp8.py` + `kernels/` |
| `vllm/spec_dec/{arctic_proposer,arctic_speculator,vocab_parallel_embedding,logits_processor_opt}.py` | proposer/speculator/embedding/logits | `v1/spec_decode/*`, `model_executor/layers/*` |
| `vllm/swiftkv/llama_swiftkv.py` | Llama model rewiring | `model_executor/models/llama.py` |
| `vllm/{stats,structured_output,fp32_lm_head}.py` | loggers, grammar, lm-head | `v1/metrics/*`, `v1/structured_output/*`, logits processor |
| `server/*` | (no patches) imports only | `v1/engine/async_llm.py`, `distributed/*`, `_custom_ops`, metrics |

## Recurring drift patterns (watch these first)
These recurred across bumps and are the fastest wins:
1. **Method arg added upstream** → add it to the override with the base's default
   and forward it (e.g. `initialize_kv_cache` gained `is_profiling=False`;
   `initialize_cudagraph_keys` gained `uniform_decode_query_len=1`).
   **Recurred 0.24→0.26 (module-level patch):** the free function
   `distributed/parallel_state.py::graph_capture(device)` gained
   `graph_capture_context=None` (base `profile_cudagraph_memory` /
   cudagraph-capture callers now pass `graph_capture_context=cap_ctx`).
   Arctic's `ulysses.py` replaces this free function with an Ulysses-aware
   version — add the kwarg and honor it (`context = graph_capture_context or
   GraphCaptureContext(...)`). Runtime-only: surfaces at engine-init cudagraph
   memory profiling as `TypeError: graph_capture() got an unexpected keyword
   argument 'graph_capture_context'`.
2. **Method arg removed upstream** → drop it from the override; if the override
   still needs the value, stash it on `self` from an earlier override in the same
   call chain (e.g. `_bookkeeping_sync` lost `spec_decode_metadata`; it's now
   stashed in `sample_tokens` as `self._arctic_spec_decode_metadata`).
3. **Return-tuple arity change** → update unpacking at every call site. Also
   applies to base helpers Arctic *calls* from its overrides: `_get_cumsum_and_arange`
   went from returning `(cumsum, arange)` to just `cumsum` AND gained a required
   `arange_out` buffer arg (write-into, e.g. `self.query_pos.np`). Arctic's
   `_dummy_run` FULL/force-attention branch called the old form
   (`cum, _ = self._get_cumsum_and_arange(x)`) — only fires without FCA / in FULL
   cudagraph mode, so PIECEWISE+FCA runs never hit it. Diff the base definition and
   copy the base's exact call form (it also fills the padded `query_start_loc` tail).
4. **Symbol moved module** → e.g. `is_quantized_kv_cache` moved to
   `vllm.utils.torch_utils`; fix the import, not the usage.
5. **Symbol renamed** → e.g. `flash_attn_supports_fp8` → `flash_attn_supports_quant_query_input`.
6. **Helper deleted, replaced by env flag** → e.g. `vllm_is_batch_invariant()` →
   `envs.VLLM_BATCH_INVARIANT`.
7. **Signature change on a *called* vLLM class/function** → Arctic calls (not
   overrides) it, so import scans pass but the call breaks at runtime. (a) delegated
   constructor: `Fp8MoEMethod(self)` → `Fp8MoEMethod(self, layer)`. (b) free function
   gaining required args: `init_fp8_linear_kernel` now needs `input_dtype` +
   `weight_shape`, and since `weight_shape` isn't known in `__init__`, upstream moved
   the call into `create_weights` — so port the call site's *location*, not just its
   args. Diff the upstream file that defines the function to see where it's invoked now.
   **Gotcha (runtime-only):** relocating kernel construction to `create_weights`
   breaks Arctic paths that assign a quant method *manually* and never call
   `create_weights`. The speculator does exactly this for its FP8 LM head:
   `qhead.quant_method = OriginalFp8LinearMethod(...)` (arctic_speculator, two
   sites) — so `self.fp8_linear` stays `None` and fails at the first spec-decode
   step with `'NoneType' object has no attribute 'apply_weights'` (via
  `logits_processor_opt._get_logits` → `fp8.apply`). Fix: lazily build
  `self.fp8_linear` in `apply` when `None` (weight_shape only drives kernel
  *selection*; the kernel reads `layer.weight` at call time). Grep for direct
  `.quant_method = ` assignments to find every bypass.
  **Second-order gotcha:** building the kernel at *forward* time fails with
  `AssertionError: Current vLLM config is not set` — `init_fp8_linear_kernel`
  instantiates ops (`input_quant_fp8.Fp8LinearOp`) that call
  `get_current_vllm_config()`, which is only set during model init. Stash the
  config in `__init__` (`self.vllm_config = get_current_vllm_config()`) and wrap
  the lazy build in `with set_current_vllm_config(self.vllm_config):`.
  **Third-order gotcha:** after building the kernel you must also call
  `self.fp8_linear.process_weights_after_loading(layer)` once — upstream
  `Fp8LinearMethod.process_weights_after_loading` delegates to it, but Arctic's
  forked version doesn't and the manual LM-head path skips it. Without it,
  kernel-side state like `CutlassFP8ScaledMMLinearKernel.logical_output_size`
  stays `None` and `apply_scaled_mm` fails with `assert output_size is not None`.
8. **Forked-file drift** (`flash_attn_forest_cascade.py` is a fork of upstream
   `flash_attn.py`): re-diff the upstream base file old→new and port
   non-Arctic-specific changes (esp. `supports_kv_cache_dtype`, backend selection).
   Seen 0.18→0.24: **KV cache layout changed** — the k/v dimension moved from dim 0
   to dim 1. Upstream `get_kv_cache_shape` went `(2, num_blocks, block_size,
   num_kv_heads, head_size)` → `(num_blocks, 2, ...)`, the forward split went
   `kv_cache.unbind(0)` → `unbind(1)`, and the `include_num_layers_dimension`
   stride-order variants shifted (`(2,0,1,3,4,5)`→`(1,0,2,3,4,5)`,
   `(2,4,0,1,3,5)`→`(1,4,0,2,3,5)`). **Static/signature scans miss all of this** —
   it surfaces at first decode step as `ValueError: too many values to unpack
   (expected 2)` from `kv_cache.unbind(0)` (dim 0 is now num_blocks, not 2).
   Fix: port the layout change wholesale from upstream `flash_attn.py`.
   **Recurred 0.24→0.26 (again, differently):** the separate k/v `2` axis was
   *removed* and **packed into the content dim**: `get_kv_cache_shape` went
   `(num_blocks, 2, block_size, num_kv_heads, head_size)` →
   `(num_blocks, num_kv_heads, block_size, 2*head_size)`, and the forward split
   went `key_cache, value_cache = kv_cache.unbind(1)` →
   `kv_cache.transpose(1, 2).split(self.head_size, dim=-1)` (then
   `canonicalize_singleton_dim_strides` on each, from `vllm.utils.torch_utils`).
   Key insight that shrinks the port: the resulting key/value **views keep the
   same logical shape** `(num_blocks, block_size, num_kv_heads, head_size)` as the
   old `unbind`, so every downstream cascade path (reshape_and_cache, prefix/suffix
   FA, `key_cache.shape[-2]`/`[-3]` reads) is unaffected — only the single
   extraction point + `get_kv_cache_shape`/`get_kv_cache_stride_order` change.
   `do_kv_cache_update` uses the same split before `reshape_and_cache_flash`.
   Stride-order tuples also changed (4-dim now): NHD `(0,2,1,3)`, HND `(0,1,2,3)`,
   +num_layers NHD `(1,0,3,2,4)`, HND `(1,2,0,3,4)`.
   **Same layout assumption lives in a SECOND file — `swiftkv/llama_swiftkv.py`**
   (runtime-only, Phase-6 SwiftKV path): it still splits the FA KV cache with the
   old `[2, num_blocks, block_size, num_kv_heads, head_size]` shape via
   `key_caches.append(kv_cache[0])` / `value_caches.append(kv_cache[1])` and
   `k_cache, v_cache = kv_cache.unbind(0)` before `reshape_and_cache_flash_bulk`.
   Port to the 0.26 packed layout the same way as the FCA fork
   (`kv_cache.transpose(1,2).split(head_size, dim=-1)`), keeping the logical
   `(num_blocks, block_size, num_kv_heads, head_size)` views. Only fires with a
   SwiftKV model, so the Phase-1 smoke never hits it.
   **Stale unit test:** `tests/unit_tests/test_custom_ops.py::
   test_reshape_and_cache_flash_bulk` fails in its *reference* (not Arctic's op):
   it calls vLLM's `_C_cache_ops.reshape_and_cache_flash` with a synthetic **2D**
   `(num_tokens, hidden_size)` cache, which 0.26's op now rejects (reads `stride(2)`
   → `Dimension out of range`). The op requires the ≥3D packed cache now; update the
   test fixture's cache shape (and the bulk-op contract) alongside the SwiftKV port.
   The runbook Phase-3 GPU tests (`test_reshape_and_cache_flash_nvfp4`,
   `test_speculator_ln`, `test_sum_lstm`, `test_fp32_lm_head`) are unaffected. 0.26 also added
   `supports_sliding_window`, RSWA (`_make_rswa_mask_mod`), `_maybe_symmetrize_window`
   — not in the FCA fork; port only if a model needs them under the FCA backend.
   Separately 0.24→0.26: base `_capture_cudagraphs` gained a `profiler=None` kwarg
   (caller passes `profiler=`), so the mirror override must accept + forward it
   (pattern #1). fp8 surfaces (`init_fp8_linear_kernel`, `Fp8MoEMethod/Fp8LinearMethod`)
   did **not** drift 0.24→0.26.
9. **Instance-attribute drift** — base `__init__` stops setting an attribute the
   override reads after calling `_orig_init`. **Static import/signature scans do NOT
   catch this**; it only fails at runtime (`AttributeError` during worker init).
   Seen 0.18→0.24: base `GPUModelRunner` dropped `self.pin_memory`; v0.24 uses a
   module-level `PIN_MEMORY = is_pin_memory_available()` in `vllm.utils.torch_utils`.
   Variant — **attribute type/API drift**: the attribute still exists but its type
   changed, so the old accessor breaks. Seen 0.18→0.24: several `CpuGpuBuffer`s
   (`.gpu`/`.cpu`/`.np` + `.copy_to_gpu()`) became plain `torch` tensors — but only
   *some* of them. `self.positions`/`self.seq_lens` are now plain tensors (index
   directly; `seq_lens` is staged via a new `self.optimistic_seq_lens_cpu` +
   `.copy_(...)`), while `self.input_ids`/`inputs_embeds`/`query_start_loc`/
   `discard_request_mask`/`mrope_positions` stayed `_make_buffer` wrappers. Check
   each buffer individually against the target `__init__`; don't assume uniform.
   Fix: import and use the upstream constant, not `self.<attr>`. To find these
   proactively, grep the overrides for `self.<x>` reads and confirm each is still
   set (and still the same type) by the target base `__init__`: `rg 'self\.\w+' arctic_inference/vllm/model_runner.py`
   for attrs used right after `_orig_init`, cross-checked against the target
   `gpu_model_runner.py.__init__`.
10. **Parallel/alternate implementation of a patched class ("silent bypass")** —
    upstream ships a *second* implementation of something Arctic patches, in a
    **different module**, and routes to it for some configs. Arctic patches the
    old class; the new one is untouched, so patches silently no-op with **no
    error** — the run "works" but with none of Arctic's behavior. This is the
    worst failure mode: static scans, `apply_arctic_patches()`, and even a
    successful GPU run all pass. Seen 0.18→0.24: vLLM added the **V2 GPU model
    runner** `vllm.v1.worker.gpu.model_runner.GPUModelRunner` (a modular rewrite;
    V1 `vllm/v1/worker/gpu_model_runner.py` still exists). `GPUWorker` picks V2
    when `VllmConfig.use_v2_model_runner` is true, which defaults **on** for a
    whitelist of architectures (`Qwen3ForCausalLM`, `LlamaForCausalLM`,
    `MistralForCausalLM`, `Qwen2MoeForCausalLM`, `DeepseekV2ForCausalLM`,
    `GraniteMoeForCausalLM` — `config/vllm.py::DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES`),
    or via `VLLM_USE_V2_MODEL_RUNNER`. Arctic only patches V1. Symptom: patch
    binds but never runs — the capture bars/steps show stock vLLM behavior (all
    base capture sizes, no shift split, none of Arctic's `logger.info` capture
    lines, no per-step `bs=` cudagraph tag) even though
    `GPUModelRunner._capture_cudagraphs.__module__ == 'arctic_inference...'`.
    Diagnosis: print `type(self.model_runner)` / the bound method's `__module__`,
    or check `vllm_config.use_v2_model_runner`. **Current fix:** the plugin
    bootstrap `plugin.py::arctic_inference_plugin()` forces
    `VLLM_USE_V2_MODEL_RUNNER=0` whenever Arctic is enabled (warns if V2 was
    explicitly requested), before vLLM builds the engine config — so every Arctic
    run stays on V1 with no per-entrypoint env needed. Note V2 also does not yet
    support Arctic-shaped
    features anyway (sequence parallelism with TP>1, and spec methods outside
    `eagle/eagle3/mtp/dflash` — see `config/vllm.py::_get_v2_model_runner_unsupported_features`),
    so V1 is the only correct runner today. **TODO: port Arctic patches to the
    V2 runner** (patch `vllm.v1.worker.gpu.model_runner.GPUModelRunner` and its
    `gpu/*` helper modules) and drop the force. For each future bump, re-check
    whether more classes Arctic patches gained a V2/alternate implementation.

11. **Plugin-load timing: `EngineArgs` built before patches apply (runtime-only,
    `LLM()` python-API path).** Both 0.24 and 0.26 load general plugins from
    `EngineArgs.__post_init__` → the *first* `EngineArgs(...)` (e.g. `LLM.__init__`
    building it directly at `entrypoints/llm.py`) is constructed *before* Arctic's
    patches are applied, so `EngineArgsPatch.__new__` never upgrades it to
    `ArcticEngineArgs` and it lacks the `ArcticArgs` fields. The now-patched
    `create_engine_config` then runs on that plain instance →
    `AttributeError: 'EngineArgs' object has no attribute
    'ulysses_sequence_parallel_size'` at first engine build. **Static scans and a
    direct `EngineArgs(...)` after an explicit `load_general_plugins()` both pass**
    (the explicit pre-load patches `__new__` first). On this path the Arctic args
    can only be defaults anyway (a base `EngineArgs.__init__` would reject them as
    unknown kwargs), so the fix is `args.py::_ensure_arctic_fields(self)` at the top
    of both `create_engine_config` overrides: backfill missing `ArcticArgs` dataclass
    defaults. The CLI path (`from_cli_args` → `ArcticEngineArgs`, plugins already
    loaded via patched `add_cli_args`) is unaffected. `vllm serve` doesn't hit it;
    `vllm.LLM(...)` does — so exercise the python API, not just the server.
12. **Heavily-mirrored `EngineCore` methods drift as a block.**
    `ulysses.py::UlyssesEngineCore` mirrors `v1/engine/core.py::step_with_batch_queue`
    (only Arctic delta: a `self.iteration` counter + a commented-out timing print),
    so any upstream refactor of that method must be re-derived wholesale. 0.24→0.26
    changed several things at once: `log_iteration_details(...)` context manager →
    `capture_iteration_details(...) as iteration_details` + a post-hoc
    `self._attach_iteration_details(engine_core_outputs, iteration_details)`;
    `scheduler.schedule()` → `scheduler.schedule(self._should_throttle_prefills())`;
    `execute_model` wrapped in `log_error_detail`; deferred-sampling gate
    `use_spec_decode` + unconditional `assert draft_token_ids is not None` →
    `check_for_draft_tokens` + `if draft_token_ids is not None:`; and the
    non-block return became `return None, model_executed` gated on
    `len(batch_queue) < batch_queue_size and (model_executed or
    scheduler.has_requests())`. Runtime-only: first surfaces at the first engine
    step as `AttributeError: 'EngineCoreProc' object has no attribute
    'log_iteration_details'`. Re-copy the base body and re-add only the Arctic
    counter/timing lines.

## Config-gated incomplete-feature gaps (independent of any bump)
These only fire under specific runtime configs, so static scans AND a plain
`apply_arctic_patches()` smoke test miss them — they surface during GPU capture.
- **Shift-parallel cudagraph capture** (`enable_shift_parallel=True` + cudagraphs):
  `GPUModelRunnerPatch._capture_cudagraphs` referenced two names that were
  **defined/assigned nowhere**: the method `_register_shift_cudagraph_keys(...)`
  and the variable `compilation_cases_shift` passed to it — i.e. the whole
  shift-capture block was unfinished and had never executed. Resolved 0.18→0.24
  by implementing `_register_shift_cudagraph_keys(batch_descriptors, mode)` and
  fixing the call to pass `batch_descriptors_shift`. Contract: base
  `CudagraphDispatcher.initialize_cudagraph_keys` registers keys only for the
  base (scaled) sizes; the shift model captures a separate **unscaled** size
  table, so its keys must be added explicitly via `disp.add_cudagraph_key(mode, d)`.
  Must run inside `_use_shift_cudagraph_tables()` (the call site does) so
  `_create_padded_batch_descriptor` uses the shift `_bs_to_padded_graph_size` —
  matching what runtime `dispatch()` computes (the shift execute path dispatches
  under the same context). For PIECEWISE, relax the key with
  `replace(desc, num_reqs=None, uniform=False)` exactly like upstream's
  mixed-PIECEWISE branch.
  **Shift-model FULL capture (implemented later):** the original code downgraded
  the shift model to PIECEWISE in the FULL pass, so a uniform-decode shift batch
  routed to the shift model under the default `FULL_AND_PIECEWISE` would dispatch
  FULL, find no captured graph, and hit v0.24's `CUDA graph capturing detected at
  an inappropriate time`. Fixed by (a) a single descriptor source of truth,
  `_shift_capture_descriptors(mode)`, that builds the exact per-mode descriptors
  (PIECEWISE relaxed; FULL uniform-decode restricted to the decode-eligible subset
  `uniform_decode_query_len <= bs <= uniform_decode_query_len * max_num_seqs`,
  mirroring base `initialize_cudagraph_keys`), used for BOTH `add_cudagraph_key`
  and `_orig_capture_cudagraphs` so capture-key == registered-key == dispatch-key;
  and (b) removing the downgrade so the shift model captures the pass's mode. The
  outer `CUDAGraphWrapper(runtime_mode=FULL)` on `self.shift_model` and the runtime
  shift execute path (under `set_shift_parallel_mode(True)` + shift tables) already
  supported replay, so no runtime change was needed.
  **FCA gate:** FCA only fires under piecewise (its per-batch gate requires
  `not self.use_full_cuda_graph`), so when `_forest_cascade_attn_config is not None`
  the engine forces `compilation_config.cudagraph_mode` down to PIECEWISE at
  engine-config creation (`args.py._maybe_force_piecewise_for_fca`, both sync and
  async `create_engine_config`; only full modes are downgraded, NONE/PIECEWISE are
  preserved). This makes base+shift PIECEWISE together (so FCA fires, no FULL pass,
  no crash). FCA stays off by default (`forest_cascade_attn_configs=None`); do not
  coerce a missing config to `{}` in the args/config path (the server's `use_fca`
  toggle maps to `"{}"` separately — leave it). Still get maintainer sign-off
  before implementing feature gaps like this rather than treating them as drift.
- **Speculator has its own CUDA graph cache** (independent of vLLM warmup): the
  Arctic speculator (`arctic_speculator.py`, both `ArcticMLPSpeculator` and
  `ArcticLSTMSpeculator`) keeps a private `self.cuda_graphs` dict of raw
  `torch.cuda.CUDAGraph`s, keyed by **request-count** buckets (`padding_size`),
  captured **lazily** on first use inside `generate_proposals` via its own
  `graph_capture()` on the `_TP` group — NOT part of the base/shift capture, and
  NOT run during `_dummy_run` warmup (that drafter warmup is gated on
  `use_eagle()`, which is False for method `arctic`). Symptom: recurring
  `custom_all_reduce.py: Registering N cuda graph addresses` bursts + one-step
  `dt` spikes mid-run as the running batch size steps into new buckets (benign,
  not a rebase bug). To front-load them, `GPUModelRunnerPatch` provides
  `_capture_arctic_speculator_cudagraphs()` (called at the tail of
  `_capture_cudagraphs`, after base+shift, model restored, outside shift mode),
  which drives one dummy `generate_proposals` per distinct `padding_size` bucket
  (inputs sized from `model.static_cuda_buffers["previous_hidden_states"]` so it
  works for both speculator variants). Always on whenever an Arctic speculator
  is configured (method `arctic`/`mlp_speculator` with cudagraphs enabled).

## Dependency pins (pyproject.toml)
Bump together and keep consistent with the target vLLM's own pins:
- `[project.optional-dependencies] vllm = ['vllm==<target>']` (this is what the
  version check reads).
- build-system `torch==<target's torch>` and `protobuf>=<target's floor>`.
- Confirm with: `uv pip install vllm==<target> --dry-run 2>&1 | grep -E '^\s*\+\s*(torch|protobuf)'`.
- The `embedding` extra pins a **different** vLLM (separate path) — leave it unless
  explicitly rebasing embedding too.

## Rebase history (internal fork, ~Apr 2026 cutoff)
Every rebase landed as one feature branch in `arcticinference-internal` (PR
numbers/commits are that fork's; sanity-check against `git log`). Newest first:

| vLLM | PR | Commit | Notes |
|------|----|--------|-------|
| 0.18.0 | #39 | `f303dc1` | torch 2.10.0; pydantic-based config; FCA on-by-default |
| 0.14.1 | #242 | `19c8eac` | `vllmd_mainstream` baseline of that era |
| 0.11.0 | #216 | `0ea6a68` | added `csrc/custom_ops/` (`speculator_ln`, `sum_lstm`) |
| 0.10.1 | #162 | `e1d1ff2` | added `arctic_inference/envs.py` |
| 0.9.2 | #132 | `319edfc` | minor; mostly model_runner |
| 0.9.0.1 | #89 | `24e25ca` | renamed `shift_parallel.py` -> `ulysses.py` |
| 0.8.4 | #24 | `38eb780` | early; only 3 files |

Version-check infrastructure (not rebases, but relevant): PR #21 `2aa4642` add
runtime installed-vllm check; #22 `b676952` allow dev vllm to pass; #57 `07b323c`
warn+disable plugin on vLLM V0; #186 `e8d252f` `ARCTIC_INFERENCE_SKIP_VERSION_CHECK`.

Diff size grows with vLLM's V1 surface (budget ~a week for any minor cross):
`0.8.4` 3 files +153/-65 · `0.9.0.1` 12 +497/-339 · `0.9.2` 6 +306/-268 ·
`0.10.1` 9 +182/-315 · `0.11.0` 20 +2218/-771 · `0.14.1` 19 +2537/-963 ·
`0.18.0` 21 +617/-554.

## Historical import-path churn catalog (triage import errors fast)
Complements the runtime/behavioral drift patterns above — these are the *import
move / API-shape* churns that recur across bumps. Concrete before/after:
1. **`vllm.utils` module split** — `from vllm.utils import round_up, cdiv` ->
   `from vllm.utils.math_utils import round_up, cdiv`; also
   `vllm.utils.network_utils` (`get_distributed_init_method`, `get_open_port`,
   `get_loopback_ip`), `vllm.utils.system_utils` (`get_mp_context`),
   `vllm.utils.argparse_utils` (`FlexibleArgumentParser`).
2. **V0 -> V1 attention move** — `from vllm.attention.backends.abstract import
   AttentionType` + `vllm.v1.attention.backends.utils.CommonAttentionMetadata` ->
   `from vllm.v1.attention.backend import AttentionType, CommonAttentionMetadata`;
   `vllm.attention.layer.Attention` -> `vllm.model_executor.layers.attention.Attention`.
3. **`vllm.config` submodules** — `VllmConfig` still top-level, but
   `CompilationConfig`/`CUDAGraphMode` from `vllm.config.compilation`, and
   `config`/`is_init_field` from `vllm.config.utils` (0.18).
4. **Pydantic-based configs (0.18+)** — `ParallelConfig`/`SpeculativeConfig` no
   longer plain `@dataclass`: use `@config(config=ConfigDict(extra="forbid"))`
   from `vllm.config.utils`. Also drop `world_size` as a `@property` and set
   `self.world_size` in `__post_init__` (post-init now overwrites the field).
5. **`EngineArgs.from_cli_args` shape** — `<=0.14`: wrapped, via
   `EngineArgs.__dict__["from_cli_args"].__wrapped__`; `0.18`: plain classmethod,
   use `EngineArgs.from_cli_args.__func__(ArcticEngineArgs, args)`.
6. **Scheduler internals** — `Scheduler`/`AsyncScheduler._update_after_schedule`
   (patched in `patches.py`) and `request.is_prefill_chunk` /
   `scheduler_output.pending_structured_output_tokens` flip signatures repeatedly.
7. **`ParallelConfig` enum/flag renames** — e.g.
   `pass_config.enable_sequence_parallelism` -> `enable_sp` (0.14.1).
8. **Model-runner init/output fields** — `EMPTY_MODEL_RUNNER_OUTPUT`,
   `SamplerOutput`, `apply_grammar_bitmask` etc. move between `vllm.v1.outputs` /
   `vllm.v1.structured_output.utils` per release.
9. **FlashInfer optional import** — catch both (`swiftkv/llama_swiftkv.py`):
   `try: from vllm.v1.attention.backends.flashinfer import FlashInferMetadata
   except (ImportError, RuntimeError): FlashInferMetadata = None`.
10. **CUDA-graph capture context** — `parallel_state._TP.graph_capture()` was
    replaced by going through `CudaCommunicator` directly (0.18); expect
    "what's a group's communicator" churn whenever parallel-state internals move.

## Deployment tail (out of scope for the plugin rebase — call out in the PR)
Forward-port internal → public → `snowflakedb/ArcticInference`; corvo
`vllmd_mainstream` Dockerfile + flash-attn ABI + `vllm-releases.yaml`.
