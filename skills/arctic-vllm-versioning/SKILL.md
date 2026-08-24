---
name: arctic-vllm-versioning
description: >-
  Explains how ArcticInference targets and enforces its vLLM version. Covers the
  target-version declaration (`VLLM_PATCH_VERSION` in `arctic_inference/utils.py`),
  the unpinned `vllm` install extra, the runtime plugin gate
  (`arctic_inference.vllm.plugin` + `get_compatible_vllm_version`), the
  (vLLM, torch, protobuf) stack coupling, install/build behavior, and which parts
  of the package are pinned vs pin-agnostic. Use when asked how ArcticInference
  pins or checks the vLLM version, how the install extras or plugin activation
  work, or when changing the versioning/extras mechanism.
disable-model-invocation: true
---

# ArcticInference vLLM versioning

ArcticInference is a vLLM plugin whose patches target an **exact** vLLM version.
That target version is a constant (`VLLM_PATCH_VERSION`), decoupled from the
install extras: the `vllm` extra is left **unpinned** so vLLM resolves against the
user's torch, and Arctic patches apply only when the *installed* vLLM matches the
target (otherwise vLLM runs unmodified). This skill describes the current
targeting/enforcement mechanism. For rebasing the plugin to a new vLLM release,
use the `rebase-arctic-inference` skill; for the multi-pin future design, see
`projects/multi_version_pins/DESIGN.md` in the repo.

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
vLLM target ──┬── torch    (vLLM 0.26.0 itself pins torch==2.11.0)
              └── protobuf (vLLM 0.26.0 requires protobuf>=5.29.6)
build-system: torch is a floor (torch>=2.10.0) + protobuf pin, chosen to match
the target stack (build isolation resolves the newest torch >= the floor)
```

## Where the target version is declared

The **target version** lives in code, not in the extras:
- `arctic_inference/utils.py`: `VLLM_PATCH_VERSION = "0.26.0"` — the single source
  of truth the runtime version check reads. `get_compatible_vllm_version()` just
  returns this constant.

`pyproject.toml` (install shape only):
- `[project.optional-dependencies] vllm = ['vllm']` — **unpinned**; vLLM resolves
  against the user's torch. It no longer encodes the target version.
- `vllm-26 = ['vllm==0.26.0']`, `vllm-18 = ['vllm==0.18.0']` — explicit-version
  install shortcuts. The user picks one to match their torch; only the one equal to
  `VLLM_PATCH_VERSION` gets Arctic patches (others -> plugin skips, no acceleration).
- `embedding = ['vllm==0.9.2', ...]` — a **separate** stack with its own pin.
- `[build-system].requires` — `torch>=2.10.0` (floor), `protobuf==...` (build-time).
- `[project.entry-points."vllm.general_plugins"]` — registers the plugin entry.

Base `[project]` has **no `dependencies`**, so `pip install arctic-inference`
installs nothing vLLM-related; `pip install arctic-inference[vllm]` pulls vLLM
(unpinned), and the plugin accelerates only if that resolves to `VLLM_PATCH_VERSION`.

## How the pin is enforced (runtime only)

Install/build never check vLLM (`setup.py` imports only `torch`). The check is at
runtime, in the plugin entrypoint `arctic_inference/vllm/plugin.py`:

1. No-op unless `ARCTIC_INFERENCE_ENABLED=1`.
2. Platform check (cuda) unless `ARCTIC_INFERENCE_SKIP_PLATFORM_CHECK=1`.
3. **Version gate (hard fail):** unless `ARCTIC_INFERENCE_SKIP_VERSION_CHECK=1`,
   if `vllm.__version__ != get_compatible_vllm_version()`, **raise `RuntimeError`**.
   Enabling Arctic is an explicit request for acceleration, so a version mismatch
   fails loudly (with a message telling the user to install the matching vLLM,
   unset `ARCTIC_INFERENCE_ENABLED`, or set `ARCTIC_INFERENCE_SKIP_VERSION_CHECK=1`)
   rather than silently running unpatched. To run unmodified vLLM, leave
   `ARCTIC_INFERENCE_ENABLED` unset — then the plugin no-ops at step 1 and never
   reaches this gate.
4. Only when compatible: force `VLLM_USE_V2_MODEL_RUNNER=0` and apply the
   version-specific monkeypatches via `apply_arctic_patches()`.

`get_compatible_vllm_version()` in `arctic_inference/utils.py` returns the
`VLLM_PATCH_VERSION` constant; `plugin_version_compatible()` is `True` iff
`vllm.__version__` equals it. Because the target lives in code (not package
metadata), changing it does **not** require a reinstall to take effect, and the
unpinned `vllm` extra can never desync the check.

## `ARCTIC_INFERENCE_ENABLED` vs the version check

These are **two sequential gates**, and the enabled flag is the master switch that
runs *before* any version logic. vLLM invokes the plugin entry point in every
process that loads general plugins, so `ARCTIC_INFERENCE_ENABLED` (default `0`) is
what makes Arctic opt-in. The version check (`plugin_version_compatible()`, which
reads `VLLM_PATCH_VERSION`) is **downstream** — it is never evaluated unless the
flag is on.

The gate has three outcomes — patch, run vanilla, or hard-fail:

```
ENABLED off              -> vanilla vLLM (version never checked)
ENABLED on + (match or SKIP_VERSION_CHECK) -> patch
ENABLED on + mismatch (no SKIP)            -> raise RuntimeError
```

Evaluated as a short-circuit: (1) if `ENABLED` is off (default), nothing else is
even checked → vanilla vLLM; (2) once enabled, patch if the installed vLLM equals
`VLLM_PATCH_VERSION` **or** `SKIP_VERSION_CHECK` overrides a mismatch; (3) enabled
but wrong version with no override → **raise** (fail loud; enabling Arctic is an
explicit request for acceleration, so silently running unpatched is treated as a
misconfiguration). The skip flag only matters when the versions *don't* match.

| `ENABLED` | `version_match` | `SKIP_VERSION_CHECK` | Result |
|---|---|---|---|
| `0` (default) | — (not checked) | — (not checked) | Vanilla vLLM |
| `1` | T | — (irrelevant) | **Patch** |
| `1` | F | T | **Patch** (forced; dev/rebase escape hatch) |
| `1` | F | F | **`RuntimeError`** (install matching vLLM or unset `ENABLED`) |

**Server interaction.** The BYO server calls `vllm.plugins.load_general_plugins()`
itself in `worker.py::initialize()`, which runs the plugin entry point. vLLM only
wraps plugin *import* in try/except, not the *call*, so the hard-fail propagates:
running the server with `ARCTIC_INFERENCE_ENABLED=1` on a mismatched vLLM now
**crashes at `load_general_plugins()`** (this is intentional — it was the old
"CP-A" behavior, restored on purpose). The BYO path is therefore "run with
`ARCTIC_INFERENCE_ENABLED` unset on a non-target vLLM": then the plugin no-ops and
the server uses vanilla args. Because reaching the code after
`load_general_plugins()` with Arctic enabled implies the patches were applied,
`worker.py` selects args with just `use_arctic = arctic_inference_effective_enabled()`
(no separate version recheck) and, when false, falls back to vanilla
`AsyncEngineArgs` and strips Arctic-only kwargs.
`arctic_inference_effective_enabled()` also consults `extra_env` (e.g.
`ModelConfig.extra_env`), not just `os.environ`, so the driver can predict whether
*workers* will enable the plugin and omit Arctic engine kwargs accordingly. Note
`server/config.py` gates its Arctic kwargs on `arctic_inference_effective_enabled()`
too.

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
- **Target version decoupled from extras:** the supported version is now the
  `VLLM_PATCH_VERSION` constant in `utils.py`, and the `vllm` extra is unpinned
  (`'vllm'`). Previously the check parsed the exact pin out of the `vllm` extra's
  metadata, so unpinning that extra would have silently disabled patching on every
  version; the constant makes unpinning safe.
- **Build torch is a floor:** `[build-system].requires` uses `torch>=2.10.0`
  instead of an exact pin, so the build no longer hard-fails on nearby torch.
- **Hard fail on enabled + mismatch:** the version gate now **raises
  `RuntimeError`** (not warn+skip) when `ARCTIC_INFERENCE_ENABLED=1` but the
  installed vLLM != `VLLM_PATCH_VERSION`, with a message pointing at the fixes
  (install matching vLLM / unset the flag / `SKIP_VERSION_CHECK`). Rationale:
  enabling Arctic is an explicit request for acceleration, so silently running
  unpatched is a misconfiguration. (This restores the pre-decoupling "CP-A"
  behavior on purpose; the earlier graceful-skip is reverted.)
- **BYO server:** `server/worker.py` runs vanilla vLLM when
  `ARCTIC_INFERENCE_ENABLED` is unset (falls back to `AsyncEngineArgs`, stripping
  Arctic-only kwargs). On a non-target vLLM it must run with the flag unset, since
  enabling it now hard-fails at `load_general_plugins()`.
- **Version-named shortcuts:** `vllm-26` / `vllm-18` extras (torch matching is the
  user's responsibility; only the target version is accelerated).

## Planned evolution (expand here)

To be grown into support for **multiple plugin implementations** from one tree
(see `DESIGN.md`): a `vllm/adapter.py` resolving a *set* of supported pins, and
per-torch-ABI wheels so more than one `(vLLM, torch)` stack ships with working
compiled ops. As each supported pin's plugin implementation lands, document its
stack and adapter surfaces here.
