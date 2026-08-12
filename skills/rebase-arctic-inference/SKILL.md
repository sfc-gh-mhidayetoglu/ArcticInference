---
name: rebase-arctic-inference
description: Rebase the ArcticInference vLLM plugin from one pinned vLLM version to a newer one. Use when asked to rebase, bump, upgrade, or port ArcticInference to a new vLLM release (e.g. "rebase ArcticInference to vLLM 0.25", "bump the vllm pin"), or to reconcile ArcticInference's monkeypatches against vLLM API/signature drift.
disable-model-invocation: true
---

# Rebasing the ArcticInference vLLM plugin

ArcticInference is a vLLM plugin that monkeypatches large parts of the V1 engine
(model runner, scheduler, worker, args, config, attention, spec-decode, ulysses,
swiftkv). It pins an **exact** vLLM version. A "rebase" means making the patches
compatible with a newer vLLM — it is **not** a `git rebase`. It is a
signature + behavioral-parity reconciliation against vLLM API drift.

Read [reference.md](reference.md) for the module→vLLM-surface map, the recurring
drift patterns, and the known latent gaps. Read it before touching code.

## Setup: pick versions and get a diff base
Work on a dedicated branch (e.g. `rebase/vllm_v<NN>`) in the `ArcticInference/`
repo. You need the **target** vLLM source to diff against — the plugin overrides
mirror upstream method bodies, so side-by-side diffing is the core technique.

```bash
# In a vLLM checkout, materialize both versions as worktrees:
git -C <vllm_repo> worktree add /tmp/vllm_old v<OLD>
git -C <vllm_repo> worktree add /tmp/vllm_new v<TARGET>
```
Set `OLD` = the **latest already-rebased pin** (see below), `TARGET` = the version
you're moving to.
Export `ARCTIC_INFERENCE_ENABLED=1` and `ARCTIC_INFERENCE_SKIP_VERSION_CHECK=1`
for the dev loop; unset the skip at the end.

### Choosing the base (`OLD`) — always rebase incrementally
`OLD` is **not** whatever `main` pins — it is the newest pin that has a completed,
validated rebase, even if that rebase lives on an unmerged branch. Rebase from the
**smallest possible version delta**: base the new branch off the latest
`rebase/vllm_v<NN>` branch and diff `latest-rebased → TARGET`.

Why incremental beats jumping from `main`'s older pin:
- Each completed rebase already absorbed its era's drift fixes (e.g. the 0.18→0.24
  work solved `pin_memory` #9, the FP8 LM-head lazy-build chain #7, the FCA
  KV-cache layout #8, shift cudagraph-key registration, and the V2-runner force
  #10). Restarting from the older pin re-does all of it.
- The core technique is `OLD→TARGET` upstream diffing; a smaller delta shrinks
  Phase 3 (the bulk of the work) and the Phase 6 runtime loop.

Procedure: find the newest `rebase/vllm_v<NN>` branch (`git branch -a | grep
rebase/vllm`), confirm its `pyproject.toml` `vllm==` pin and that it was
GPU-validated (has a `REBASE_REPORT.md`/`REBASE_TEST.md`), then branch the new
`rebase/vllm_v<TARGET>` **off that branch**, not `main`. Only fall back to an
older base if the latest rebase branch was abandoned or never validated.

> Worked example: to move to **0.26.0**, base off `rebase/vllm_v24`
> (`vllm==0.24.0`, already GPU-validated) with `OLD=0.24.0`, `TARGET=0.26.0` — do
> **not** start from `main`'s `0.18.0`. Since 0.24 isn't merged to `main`, note the
> PR-stacking question (land 0.24 first vs. fold 0.24+0.26) — that's a merge
> concern, separate from the base-version choice.

### Cross-version consistency (the evolution ledger)
Every pin should be a *consistent continuation* of the ones before it, not a
one-off. Before and after touching code, compare the new pin against **all**
other pins (`main` + every `rebase/vllm_v*` branch + the in-progress working
tree) so recurring drift surfaces (the KV-cache layout, `_capture_cudagraphs`,
the V2 force, moved imports…) evolve predictably:
```bash
python ArcticInference/skills/rebase-arctic-inference/scripts/compare_pins.py \
  --repo ArcticInference --history --include-worktree [--write ArcticInference/EVOLUTION.md]
```
`--history` mines **every past bump** from first-parent commit history (the pin
walked `0.8.1→0.8.4→0.9.0.1→0.9.2→0.10.1→0.11.0→0.14.1→0.18.0→0.24.0→0.26.0`, most
as single commits on `main` long before the `rebase/vllm_v*` branches existed), so
each historical pin becomes a column even with no surviving branch. Without it,
only `main` + `rebase/vllm_v*` branches + the working tree are compared. It runs
the probes in `scripts/evolution_probes.json`, prints a version-sorted table
(with a legend mapping each column to its branch/commit), and emits **advisory
anomaly flags**:
- `[STALLED]` — a surface that changed on *every* prior bump but **not** this one
  (a likely missed port — the #1 thing this catches). Confirm it genuinely didn't
  need to change this release.
- `[VANISHED]` — present in the previous version, gone in the newest.
- `[REVERTED]` — the newest value matches an *older* version but differs from the
  immediately-previous one (possible regression).
Flags are advisory — a human confirms each. Works the same rebasing **newer or
older**: the table is sorted by version, so a back-port slots in correctly too.
See [examples/v0.24_to_v0.26/EVOLUTION.md](examples/v0.24_to_v0.26/EVOLUTION.md) for sample output (the full
pin table plus `[STALLED]` flags).

**This is how the skill compounds:** whenever Phase 3/6 uncovers a *novel
recurring* drift, add a probe for it to `evolution_probes.json` (a small
`{label, file, regex}`, `dotall: true` for multi-line matches, capture group 1
if present). The next bump then tracks that surface automatically across all
versions. Keep probes tight (one salient expression each).

## Workflow
Copy this checklist and track it with the todo tool:
```
- [ ] Phase 0: bump pins (pyproject) + set dev env + cross-version consistency check (compare_pins.py)
- [ ] Phase 1: static import/module resolution vs target tree
- [ ] Phase 2: signature-drift scan + per-method reconciliation
- [ ] Phase 3: behavioral re-derivation of high-churn overrides
- [ ] Phase 4: lower-risk modules + server/weight-sync imports
- [ ] Phase 5: no-GPU validation (compile, import-smoke, collect-only)
- [ ] Phase 6: hand off GPU validation (REBASE_TEST.md)
```

### Phase 0 — Pins
In `pyproject.toml`: set `vllm==<TARGET>` (the `[project.optional-dependencies] vllm`
extra — this is what the version check reads), and align `torch` / `protobuf` with
the target's own pins (see reference.md for the `uv pip install --dry-run` check).
Leave the `embedding` extra's separate vLLM pin unless explicitly rebasing it.
Then run the cross-version consistency check (see "Cross-version consistency"
above) to place the new pin in the evolution and surface any missed-port flags;
re-run it at the end of Phase 5 to confirm the finished branch is consistent.

### Phase 1 — Static import resolution (fast, catches moved/renamed/deleted symbols)
```bash
python ArcticInference/skills/rebase-arctic-inference/scripts/check_vllm_imports.py \
  --arctic ArcticInference/arctic_inference --vllm /tmp/vllm_new/vllm
```
Fix real breaks (moved module → fix import path; renamed/deleted symbol → update).
Confirm flagged symbols manually — `*`-re-exports and private module globals
(`_TP`, `current_platform`, lazy `__getattr__` exports like `SamplingParams`)
are **false positives**.

### Phase 2 — Signature drift
```bash
python ArcticInference/skills/rebase-arctic-inference/scripts/check_signatures.py \
  --arctic ArcticInference/arctic_inference --vllm /tmp/vllm_new/vllm
```
For each real mismatch apply the drift patterns in reference.md (arg added →
add+forward with base default; arg removed → drop + stash-on-self if still needed;
return arity changed → fix all unpack sites). `(self, *args, **kwargs)` passthrough
wrappers and module-target patches flag as noise — verify, don't "fix".

### Phase 3 — Behavioral re-derivation (the bulk of the work)
For each override whose **signature is stable but body mirrors upstream**, diff the
upstream base method `OLD` vs `TARGET` and port the changes into the override:
```bash
diff <(sed -n '/def <method>/,/def /p' /tmp/vllm_old/vllm/<path>) \
     <(sed -n '/def <method>/,/def /p' /tmp/vllm_new/vllm/<path>)
```
Priority order (highest churn first): `model_runner.py` (`execute_model`,
`sample_tokens`, `_bookkeeping_sync`, `_dummy_run`, `propose_draft_token_ids`,
`_capture_cudagraphs`, `profile_run`, `load_model`), then `patches.py`
(scheduler/worker), `ulysses.py`, `flash_attn_forest_cascade.py` (re-diff the whole
forked upstream file), `fp8.py`. See reference.md for the full target map.

### Phase 4 — Lower-risk + server
Re-verify `stats.py`, `structured_output.py`, `fp32_lm_head.py`, swiftkv,
remaining `spec_dec/*`, and the `server/` imports (`load_general_plugins`, metrics
classes, weight-sync deps). Compile as you go: `python -m py_compile <file>`.

### Phase 5 — No-GPU validation gate (stop here without a GPU)
```bash
python -m compileall -q -x dynasor ArcticInference/arctic_inference
python ArcticInference/skills/rebase-arctic-inference/scripts/check_vllm_imports.py --arctic ArcticInference/arctic_inference --vllm /tmp/vllm_new/vllm   # clean
python ArcticInference/skills/rebase-arctic-inference/scripts/check_signatures.py  --arctic ArcticInference/arctic_inference --vllm /tmp/vllm_new/vllm    # only intentional flags
```
If a vLLM `==TARGET` env is importable here, also run
`python -c "import vllm; vllm.plugins.load_general_plugins()"` and
`pytest tests/unit_tests --collect-only`. Re-enable the version check (unset
`ARCTIC_INFERENCE_SKIP_VERSION_CHECK`).

> **This gate is NOT "done" — it is the floor.** Static scans have a large blind
> spot: they cannot see runtime-only drift (drift patterns #7, #9 and its variant
> in reference.md — a base `__init__` that stops setting an attribute, an attribute
> whose *type* changed, or a called vLLM function whose signature changed). Passing
> Phase 5 only means "imports resolve and it compiles." Budget for an
> iterate-on-GPU fix loop in Phase 6; do not report the rebase as working until the
> engine actually starts and runs on a GPU.

### Phase 6 — GPU validation + runtime-fix loop (handoff)
Runtime + functional validation needs CUDA and is run on a GPU node. The runbook
maps each code change to a targeted test and gives ordered phases (smoke → CPU
units → GPU units → spec-decode → ulysses → swiftkv → forest-cascade/FP8 →
weight-sync → shift-parallel → server). Regenerate/update it per rebase to reflect
the actual diff, then hand it to the GPU agent. A completed example is
[examples/v0.24_to_v0.26/REBASE_TEST.md](examples/v0.24_to_v0.26/REBASE_TEST.md) (the 0.24→0.26 run: the three
runtime-only breaks it caught, the phase-by-phase results, and the SwiftKV
deprecation finding).

Expect this to be a **loop**, not a single pass: runtime-only breaks surface one
at a time (each worker traceback points at one `arctic_inference/...` line), and
they keep coming **past engine init, into the first prefill / first decode / first
spec-decode step** — so "engine started" is not "it works." For each: read the
failing line, diff the relevant upstream base against the target tree, fix per the
reference.md drift patterns, recompile, re-run. The 0.18→0.24 loop, in order:
- **engine init:** `pin_memory` (#9) → fp8 kernel signature/location (#7) →
  `positions`/`seq_lens` buffer type (#9 variant) → shift cudagraph key
  registration (config-gated gap).
- **first decode step:** FCA KV-cache layout `unbind(0)`→`unbind(1)` (#8).
- **first spec-decode step:** fp8 LM-head `fp8_linear is None` lazy-build (#7 gotcha).
**Log every new runtime break and its fix back into reference.md as you go** so the
next bump starts from a richer pattern list.

## Guardrails
- **Check for silent-bypass drift first** (reference.md #10): each bump, verify
  vLLM hasn't added a *parallel implementation* of a class Arctic patches in a
  new module (e.g. the V2 GPU model runner `vllm.v1.worker.gpu.model_runner`,
  auto-enabled for Qwen3/Llama/Mistral/… via `use_v2_model_runner`). Arctic
  patches V1 only and forces it in the plugin bootstrap
  (`plugin.py::arctic_inference_plugin()` sets `VLLM_USE_V2_MODEL_RUNNER=0`);
  confirm that force still works and no other patched class gained a V2 variant.
  This class of break passes every static scan AND a GPU run — it just silently
  no-ops the patches.
  **TODO carried forward: port Arctic patches to the V2 runner.**
- Prefer real fixes over `getattr(..., default)` shims — a silent shim hides the
  next break.
- Every `_orig_<name>` alias assumes the base still has that attribute; if the
  static scan says the base lost it, the alias will `AttributeError` at import.
- The forked `flash_attn_forest_cascade.py` must track upstream `flash_attn.py`
  changes that are not Arctic-specific — re-diff the whole file, don't spot-fix
  (0.18→0.24 example: the KV-cache k/v-dim layout move — see reference.md #8).
- Config-gated incomplete features (e.g. the shift-parallel cudagraph block, which
  referenced the undefined `_register_shift_cudagraph_keys` **and**
  `compilation_cases_shift`) are **not** rebase drift. Flag them, and get the
  maintainer's sign-off before implementing — do not silently fabricate a
  definition. (This one was implemented for 0.18→0.24 with sign-off; see
  reference.md "Config-gated incomplete-feature gaps".)
- Deployment/forward-port tail is out of scope for the plugin rebase (see
  reference.md) — call it out in the PR rather than doing it on this branch.

## Completed rebases (worked examples)
Two full rebases to learn the *shape* of the work from. The recurring truth in
both: the no-GPU gate (Phase 5) touched only a handful of files, but **most of the
real fixes landed in the Phase-6 GPU runtime loop** — budget for it.

**0.18.0 → 0.24.0** — report + plan at
[examples/v0.18_to_v0.24/REBASE_REPORT.md](examples/v0.18_to_v0.24/REBASE_REPORT.md)
and [REBASE_PLAN.md](examples/v0.18_to_v0.24/REBASE_PLAN.md) (report = per-module
survey, plan = executed to-do list).
No-GPU touched ~5 files (pins + `model_runner.py`, `ulysses.py`,
`flash_attn_forest_cascade.py`, `fp8.py`); the GPU loop then re-edited three of
them: `model_runner.py` shift-cudagraph keys, `flash_attn_forest_cascade.py`
KV-cache layout (`unbind(0)`→`unbind(1)`, drift #8), `fp8.py` lazy kernel build (#7).

**0.24.0 → 0.26.0** — runbook at [examples/v0.24_to_v0.26/REBASE_TEST.md](examples/v0.24_to_v0.26/REBASE_TEST.md)
(change→test map, phase-by-phase results). No-GPU again looked small; the GPU loop
surfaced **three** runtime-only breaks, all past engine init: `args.py`
plugin-timing backfill (`_ensure_arctic_fields`, #11), `ulysses.py`
`graph_capture(graph_capture_context=)` kwarg (#1), and `ulysses.py`
`step_with_batch_queue` re-derivation (#12). Plus the FCA KV-cache layout drifted
**again** (packed `(num_blocks, num_kv_heads, block_size, 2*head_size)`, #8) and a
CUDA-13 build fix (`csrc/custom_ops/CMakeLists.txt` arch list). SwiftKV was found
**deprecated/unsupported on 0.26** (unmaintained since 0.14.1; wrong output even in
eager mode — bulk KV kernel not ported to the packed layout).

**Cross-rebase lesson:** `model_runner.py` / `ulysses.py` /
`flash_attn_forest_cascade.py` churn on essentially every bump, and the **FCA
KV-cache layout (#8) has now moved in two consecutive releases** — treat it as a
must-check surface each time (the evolution ledger tracks it). Full technical fixes
for both live in reference.md; use these two as templates for scoping and for
expecting a substantial post-Phase-5 GPU loop.

## Adjacent artifacts that also pin a vLLM version (not the plugin, but bump too)
- `ArcticInference/benchmark/rollout/*.patch` (`sampling_params.patch`,
  `parallel_sampling.patch`, applied via `patch_sampling.sh`) add a `max_tokens_n`
  rollout-replay feature and are written against a specific vLLM. Re-diff their
  target files (`vllm/sampling_params.py`, `vllm/v1/engine/parallel_sampling.py`)
  against the target tree and regenerate. `SamplingParams` is a `msgspec.Struct`
  (`omit_defaults=True`) — a new field after `n` (both have defaults) is safe.
