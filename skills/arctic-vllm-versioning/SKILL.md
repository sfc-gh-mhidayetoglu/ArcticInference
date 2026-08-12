---
name: arctic-vllm-versioning
description: >-
  Explains how ArcticInference pins and enforces its vLLM version. Covers the
  exact-pin declaration (pyproject `vllm` extra), the runtime plugin gate
  (`arctic_inference.vllm.plugin` + `get_compatible_vllm_version`), the
  (vLLM, torch, protobuf) stack coupling, install/build behavior, and which parts
  of the package are pinned vs pin-agnostic. Use when asked how ArcticInference
  pins or checks the vLLM version, how the install extras or plugin activation
  work, or when changing the versioning/extras mechanism.
disable-model-invocation: true
---

# ArcticInference vLLM versioning

ArcticInference is a vLLM plugin pinned to an **exact** vLLM version. This skill
describes the current pinning/enforcement mechanism. For rebasing the plugin to a
new vLLM release, use the `rebase-arctic-inference` skill; for the multi-pin
future design, see `projects/multi_version_pins/DESIGN.md` in the repo.

Deeper detail (enforcement code, install matrix, build behavior, version tables)
is in [reference.md](reference.md).

## Mental model: vLLM anchors one stack

- **The vLLM pin is the anchor.** A vLLM release pins **torch exactly** (`==`)
  and **protobuf** as a floor (`>=`). torch and protobuf are *slaves* of the
  vLLM pin, not independent knobs.
- **One source tree = one stack.** A branch supports exactly one
  `(vLLM, torch, protobuf)` combination; `csrc` compiles against a single torch
  ABI (`[build-system].requires` torch). Multi-torch = multiple branches/wheels.
- **You select by choosing the pin.** Nothing reads the installed torch to pick
  vLLM (there is no such logic; PEP 508 markers cannot branch on torch).

```
vLLM pin ──┬── torch    (exact:  0.26.0 -> torch==2.11.0)
           └── protobuf (floor:  0.26.0 -> protobuf>=5.29.6)
build-system torch/protobuf = slaves chosen to match the pin (one ABI per build)
```

## Where the pin is declared

`pyproject.toml`:
- `[project.optional-dependencies] vllm = ['vllm==X']` — the plugin's canonical
  *supported* exact pin (the one the version check reads).
- `vllm-26 = ['vllm==0.26.0']`, `vllm-18 = ['vllm==0.18.0']` — explicit-version
  install shortcuts. The user picks one to match their torch; only the one equal to
  the supported pin gets Arctic patches (others -> plugin skips, no acceleration).
- `embedding = ['vllm==0.9.2', ...]` — a **separate** stack with its own pin.
- `[build-system].requires` — `torch==...`, `protobuf==...` (build-time slaves).
- `[project.entry-points."vllm.general_plugins"]` — registers the plugin entry.

Base `[project]` has **no `dependencies`**, so `pip install arctic-inference`
installs nothing vLLM-related; `pip install arctic-inference[vllm]` pulls the pin.

## How the pin is enforced (runtime only)

Install/build never check vLLM (`setup.py` imports only `torch`). The check is at
runtime, in the plugin entrypoint `arctic_inference/vllm/plugin.py`:

1. No-op unless `ARCTIC_INFERENCE_ENABLED=1`.
2. Platform check (cuda) unless `ARCTIC_INFERENCE_SKIP_PLATFORM_CHECK=1`.
3. **Version gate (graceful skip):** unless `ARCTIC_INFERENCE_SKIP_VERSION_CHECK=1`,
   if `vllm.__version__ != get_compatible_vllm_version()`, **warn and return
   without patching** — vLLM runs unmodified so the server / other vLLM users keep
   working. (It used to raise `RuntimeError`.)
4. Only when compatible: force `VLLM_USE_V2_MODEL_RUNNER=0` and apply the
   version-specific monkeypatches via `apply_arctic_patches()`.

`get_compatible_vllm_version()` / `plugin_version_compatible()` in
`arctic_inference/utils.py` read installed package metadata
(`importlib.metadata.requires("arctic_inference")`) for the requirement tagged
`; extra == "vllm"` (the canonical supported pin) and compare it to
`vllm.__version__`.

## Scope: what the pin actually governs

| Part | Pinned? | How |
|------|---------|-----|
| `arctic_inference/vllm/**` (plugin + patches) | yes, exact `==` | runtime check gates `apply_arctic_patches()` |
| `arctic_inference/embedding/**` | yes, separately | own `vllm==0.9.2` extra |
| `arctic_inference/server/**` | decoupled (BYO) | runs on any recent vLLM; uses Arctic engine args only when the plugin is applied, else vanilla `AsyncEngineArgs` + strips Arctic-only kwargs |
| `common/`, `suffix_decoding/`, `op_builder/`, `csrc/` | pin-agnostic | track torch/CUDA ABI |

## Per-branch model

Each `rebase/vllm_v<NN>` branch = one supported stack; `main` currently pins
`vllm==0.18.0` (torch 2.10.0). The historical support matrix and the file-by-file
surface map are in [reference.md](reference.md).

## Related tooling
- Rebasing to a new pin: `rebase-arctic-inference` skill.
- Cross-version drift tracking: that skill's `compare_pins.py` + `evolution_probes.json`.
- Multi-pin / multiple plugin implementations design: repo `projects/multi_version_pins/DESIGN.md`.

## Recently landed (on `rebase/vllm_v26`)
- **Graceful skip:** the version gate warns + skips instead of raising, so the
  server and other vLLM users keep working on a non-supported vLLM.
- **BYO server:** `server/worker.py` falls back to vanilla `AsyncEngineArgs`
  (stripping Arctic-only kwargs) when the plugin isn't applied.
- **Version-named shortcuts:** `vllm-26` / `vllm-18` extras (torch matching is the
  user's responsibility; only the supported pin is accelerated).

## Planned evolution (expand here)

To be grown into support for **multiple plugin implementations** from one tree
(see `DESIGN.md`): a `vllm/adapter.py` resolving a *set* of supported pins, and
per-torch-ABI wheels so more than one `(vLLM, torch)` stack ships with working
compiled ops. As each supported pin's plugin implementation lands, document its
stack and adapter surfaces here.
