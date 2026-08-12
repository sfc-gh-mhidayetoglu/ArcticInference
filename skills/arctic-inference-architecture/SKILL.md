---
name: arctic-inference-architecture
description: >-
  Orientation map for the ArcticInference vLLM plugin: what each feature does,
  which file implements it, how it is patched onto vLLM, and the knob that turns
  it on. Covers the ArcticPatch framework, the plugin bootstrap order, and the
  seven feature groups (advanced parallelism, speculative decoding, Forest
  Cascade Attention, SwiftKV, KV-cache/memory, RL/training, serving). Use when
  debugging an unfamiliar ArcticInference code path, explaining what changes when
  ARCTIC_INFERENCE_ENABLED=1, or locating the source/enable-knob for a feature.
disable-model-invocation: true
---

# ArcticInference feature inventory & design

The "what's in the box" reference for the ArcticInference vLLM plugin. Use it to
orient in an unfamiliar code path or to explain what changes when
`ARCTIC_INFERENCE_ENABLED=1`. Per-feature design detail is in
[reference.md](reference.md).

> **Snapshot baseline: vLLM 0.18.0.** File layout and feature set are stable, but
> exact line numbers / vLLM symbol paths drift per release — verify against the
> current tree (and see the `rebase-arctic-inference` skill for drift). For how
> the vLLM pin itself is declared/enforced, see the `arctic-vllm-versioning` skill.

## The one entry point

Everything hangs off a single `vllm.general_plugins` entry point:

```toml
[project.entry-points."vllm.general_plugins"]
arctic_inference = "arctic_inference.vllm.plugin:arctic_inference_plugin"
```

`arctic_inference_plugin()` (gated on `ARCTIC_INFERENCE_ENABLED` + an exact vLLM
pin match) calls `arctic_inference/vllm/patches.py::apply_arctic_patches()`, the
single ordered list of patch installs:

1. Register SwiftKV + Arctic Speculator model classes with HF `AutoConfig` and
   `vllm.ModelRegistry`.
2. `WorkerBasePatch` — lazily applies the model-runner / worker patches *after*
   worker fork (avoids CUDA-init-before-fork).
3. Static patches: `AsyncSchedulerPatch`, `EngineArgsPatch`, `AsyncEngineArgsPatch`,
   `ParallelConfigPatch`, `SpeculativeConfigPatch`, `SpecDecodingStatsPatch`,
   `SpecDecodingLoggingPatch`, `VllmConfigPatch`, `XgrammarBackendPatch`,
   `MLPSpeculatorConfigPatch`.
4. `apply_forest_cascade_patches()` — always installs the FCA-aware backend
   (runtime-gated).
5. `apply_shift_parallel_patches()` — always installs the Ulysses/shift surface
   (runtime-gated by `ulysses_sequence_parallel_size`).
6. FP32 LM-head patches (conditional).
7. kvcached prefix-cache patches when `KVCACHED_AUTOPATCH=1`.

**Order matters:** `WorkerBasePatch` is what triggers `GPUModelRunnerPatch` /
`WorkerPatch` post-fork, so CUDA imports don't poison the parent process.

## The patching framework (`ArcticPatch[T]`)

`arctic_inference/patching.py`. Non-invasive monkey-patch of a vLLM class **or
module** — no subclassing:

```python
class GPUModelRunnerPatch(ArcticPatch[GPUModelRunner]):
    _orig_load_model = GPUModelRunner.load_model      # save at class-def time
    def load_model(self):
        self._orig_load_model()
        # extra work
```

`apply_patch()` copies the patch class's fields/methods onto the target in place;
existing instances keep working. **Save `_orig_X = Target.X` at class-definition
time** (doing it in `__init__` captures the already-patched version → recursion).
Every `_orig_*` alias is a hard dependency on the base still having that attribute.

## The seven feature groups

| # | Group | Headline features |
|---|-------|-------------------|
| 1 | Advanced parallelism | Arctic Ulysses (sequence parallel), Shift Parallelism |
| 2 | Speculative decoding | Arctic Speculator (MLP, LSTM), Suffix Decoding, Hybrid |
| 3 | Attention | Forest Cascade Attention (FCA) |
| 4 | Model optimization | SwiftKV (Llama) |
| 5 | KV cache & memory | kvcached prefix caching, sleep/wake (level 1 & 2) |
| 6 | RL / training | FP32 LM head, NCCL weight sync |
| 7 | Serving infrastructure | Multi-model gRPC/HTTP server, embeddings server, Dynasor |

See [reference.md](reference.md) for each group's design.

## Quick reference: feature -> file -> enable knob

| Feature | Primary source | How to enable |
|---------|----------------|---------------|
| Plugin enablement (everything) | `arctic_inference/vllm/plugin.py` | `ARCTIC_INFERENCE_ENABLED=1` |
| Arctic Ulysses | `arctic_inference/vllm/ulysses.py` | `--ulysses-sequence-parallel-size N` |
| Shift parallelism | `arctic_inference/vllm/ulysses.py` | `--enable-shift-parallel --shift-parallel-threshold N` |
| Arctic LSTM/MLP speculator | `arctic_inference/vllm/spec_dec/arctic_speculator.py` | `speculative_config={"method": "arctic", "model": "<repo>"}` |
| Suffix decoding | `arctic_inference/suffix_decoding/cache.py` | `speculative_config={"method": "suffix"}` |
| Hybrid arctic + suffix | `spec_dec/` + `config.py` | `method="arctic", enable_suffix_decoding=true` |
| Forest Cascade Attention | `arctic_inference/vllm/attention/` | `--forest-cascade-attn-configs '{}'` |
| SwiftKV | `arctic_inference/vllm/swiftkv/llama_swiftkv.py` | use a `Snowflake/Llama-3.1-SwiftKV-*` model |
| FP32 LM head | `arctic_inference/vllm/fp32_lm_head.py` | `--fp32-lm-head` or `ARCTIC_FP32_LM_HEAD=1` |
| kvcached prefix caching | `arctic_inference/vllm/kvcached/` | `KVCACHED_AUTOPATCH=1` |
| Sleep/wake (level 2) | `arctic_inference/vllm/patches.py::WorkerPatch` | `enable_sleep_mode=True`, then `llm.sleep(level=2)/wake_up()` |
| NCCL weight sync | `arctic_inference/server/weight_sync/` | multi-model server `/sync_weights` |
| Multi-model server | `arctic_inference/server/` | `arctic_inference_server --port 8000` |
| Embeddings server | `arctic_inference/embedding/` | `python -m arctic_inference.embedding.replica_manager ...` |
| Dynasor (reasoning early-stop) | `arctic_inference/dynasor/` | `python -m arctic_inference.dynasor.vllm_server ...` + client |
| Skip version check (dev only) | `arctic_inference/vllm/plugin.py` | `ARCTIC_INFERENCE_SKIP_VERSION_CHECK=1` |

## Related skills
- `arctic-vllm-versioning` — how the vLLM pin is declared/enforced and the install extras.
- `rebase-arctic-inference` — porting the patches to a new vLLM release (drift map + evolution ledger).
