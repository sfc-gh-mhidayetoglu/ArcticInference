# ArcticInference vLLM versioning — reference

Detail behind [SKILL.md](SKILL.md). Paths are relative to the ArcticInference repo.

## Enforcement code

`arctic_inference/vllm/plugin.py::arctic_inference_plugin()` (the
`vllm.general_plugins` entry point):
- returns early unless `envs.ARCTIC_INFERENCE_ENABLED`.
- unless `ARCTIC_INFERENCE_SKIP_PLATFORM_CHECK`: requires `current_platform.is_cuda()`.
- **version gate:** unless `ARCTIC_INFERENCE_SKIP_VERSION_CHECK`, if
  `not plugin_version_compatible()` (i.e. `vllm.__version__ !=
  get_compatible_vllm_version()`) it **logs a warning and returns** (no patching);
  it used to `raise RuntimeError`.
- only when compatible: sets `os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"` (Arctic
  patches target the V1 model runner) and
  `from .patches import apply_arctic_patches; apply_arctic_patches()`.

`arctic_inference/utils.py`:
- `VLLM_PATCH_VERSION = "0.26.0"` — the single source of truth for the target
  version, decoupled from the install extras.
- `get_compatible_vllm_version()` — returns `VLLM_PATCH_VERSION` (no metadata
  parsing; the previous `re.match('vllm==(.*); extra == "vllm"', ...)` over
  `importlib.metadata.requires(...)` was removed).
- `plugin_version_compatible()` — `True` iff `vllm.__version__` equals that
  constant; used by both `plugin.py` and `server/worker.py`.
- Because the target is a code constant (not installed metadata), changing it
  takes effect without a reinstall, and unpinning the `vllm` extra cannot desync
  the check.

Relevant env flags (`arctic_inference/envs.py`, all default `0`):
`ARCTIC_INFERENCE_ENABLED`, `ARCTIC_INFERENCE_SKIP_VERSION_CHECK`,
`ARCTIC_INFERENCE_SKIP_PLATFORM_CHECK`, `ARCTIC_INFERENCE_SKIP_SPEC_MODEL_CHECK`,
`ARCTIC_FP32_LM_HEAD`.

## Install options (current)

| Command | Pulls in | vLLM pin? |
|---------|----------|-----------|
| `pip install -e .` | nothing (base has no `dependencies`) | — |
| `pip install -e .[vllm]` | `vllm` **unpinned** (accelerated only if it resolves to `VLLM_PATCH_VERSION`) | no |
| `pip install -e .[vllm-26]` | `vllm==0.26.0` (= `VLLM_PATCH_VERSION` -> accelerated) | yes |
| `pip install -e .[vllm-18]` | `vllm==0.18.0` (plugin skips; server/BYO, no accel) | yes |
| `pip install -e .[server]` | `ray`, `fastapi`, `uvicorn`, `pydantic>=2.0`, `tqdm` | no (BYO) |
| `pip install -e .[embedding]` | `vllm==0.9.2`, `protobuf==5.29.5` | yes (0.9.2) |
| `pip install -e .[dynasor]` | rich, prompt_toolkit, pydantic, datasets, latex2sympy2, transformers, word2number | — |
| `pip install -e .[test]` | `lm_eval[api,ifeval]==0.4.8`, pandas, pytest, sentence_transformers | — |
| `pip install -e .[docs]` | sphinx, sphinx-copybutton, sphinx-rtd-theme | — |

- Extras compose (`.[vllm,server]`).
- Extras with different vLLM pins conflict (`ResolutionImpossible`), so
  `[vllm]` and `[embedding]` are mutually exclusive in one environment.
- `[project.scripts]` exposes the `arctic_inference_server` console command.

## Build behavior

- `[build-system].requires` includes `torch>=2.10.0` (a floor, not an exact pin),
  `protobuf==...`, plus `cmake`, `ninja`, `nanobind`, `grpcio-tools`.
- `setup.py` imports only `torch` (for `torch.utils.cmake_prefix_path` + CUDA arch),
  runs CMake to compile `csrc`, and generates the embedding gRPC protobuf stubs.
  It never imports `vllm`.
- Under default build isolation, `csrc` is compiled against whatever torch the
  floor resolves to (the newest torch >= `2.10.0`). That torch must match the torch
  the installed vLLM pins, or the compiled ops are ABI-mismatched at runtime — so
  for a non-default torch, preinstall it and build with `--no-build-isolation`.

## vLLM -> (torch, protobuf)

- `0.11.0` -> torch `2.8.0`
- `0.14.1` -> torch `2.9.1`
- `0.18.0` -> torch `2.10.0`  (current `main`)
- `0.24.0` -> torch `2.11.0`
- `0.26.0` -> torch `2.11.0`, protobuf `>=5.29.6`
- embedding `0.9.2` -> protobuf `5.29.5`

The build-system `torch` pin only appeared starting at `0.11.0`; earlier pins
left torch unpinned.

## Selecting a stack by torch target

- torch->vLLM is **many-to-one** (torch `2.11.0` is shipped by both `0.24.0` and
  `0.26.0`). Rule: pick the **newest** vLLM pin carrying the required torch.
- There is **no code** that resolves vLLM from the installed torch. Selection is
  manual (choose the pin/branch, or an install extra).
- PEP 508 markers cannot express it: there is no `torch`/package-version marker;
  markers only see `python_version`, platform fields, and `extra`.

## History (first-parent main-line pin bumps)

| Date | PR | vLLM | torch |
|------|----|------|-------|
| 2025-03 | #1 | 0.8.1 | (unpinned) |
| 2025-04 | #24 | 0.8.4 | (unpinned) |
| 2025-06 | #89 | 0.9.0.1 | (unpinned) |
| 2025-07 | #132 | 0.9.2 | (unpinned) |
| 2025-10 | #162 | 0.10.1 | (unpinned) |
| 2025-12 | #216 | 0.11.0 | 2.8.0 |
| 2026-02 | #242 | 0.14.1 | 2.9.1 |
| 2026-06 | #264 | 0.18.0 | 2.10.0 |
| 2026-07 | — | 0.24.0 | 2.11.0 |
| (branch) | — | 0.26.0 | 2.11.0 |

## Server decoupling (BYO vLLM)

`arctic_inference/server/**` runs on any recent vLLM. `worker.initialize()` calls
`plugin_version_compatible()`: on the supported pin it uses `ArcticAsyncEngineArgs`
(the plugin has registered the Arctic fields); otherwise it imports vanilla
`AsyncEngineArgs` and strips the Arctic-only kwargs (`ulysses_sequence_parallel_size`,
`enable_shift_parallel`, `shift_parallel_threshold`, `forest_cascade_attn_configs`,
`fp32_lm_head`). It still needs *some* vLLM present, and the V1 surfaces it calls
(`AsyncLLM.from_vllm_config`, `SamplingParams`, `metrics.py`'s `StatLoggerBase`/
`VllmConfig`/`IterationStats`/`SchedulerStats`, `weight_sync/` NCCL/`CuMemAllocator`)
must exist on that vLLM — so "any vLLM" is really "any sufficiently-recent vLLM".
