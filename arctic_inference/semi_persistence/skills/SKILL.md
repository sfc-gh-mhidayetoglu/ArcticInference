---
name: semi-persistence
description: >-
  Orientation map for the semi_persistence subsystem: CRIU-based checkpoint and
  restore of whole vLLM instances so many models share a small GPU pool without
  cold starts. Covers the process hierarchy (Instance handle -> worker -> vLLM
  child), the saved/checkpoint/sleep/up state ladder, the Slots buddy allocator,
  the per-model Op pipeline, the orchestrator/client split, and the CRIU
  complications. Use when debugging checkpoint or restore failures, tracing a
  model state transition, reasoning about GPU slot allocation or eviction, or
  locating which file owns a given behaviour.
disable-model-invocation: true
---

# semi_persistence: checkpoint/restore multiplexing for vLLM

This subsystem lets a vLLM engine be **checkpointed to CPU and restored onto a
GPU in a second or two**, instead of paying a ~1 minute cold start. That makes
it possible to keep hundreds of models resident and multiplex them over a
handful of GPUs. Per-subsystem detail is in [reference.md](reference.md).

> This is **not** a KV-cache offload feature. Checkpointing saves the whole CUDA
> context (via `cuCheckpointProcess`) and leaves a 0 GB trace on the GPU; the
> on-disk form is a CRIU image of the entire child process tree.

## The one thing to know first

**The package is not importable as a package.** Every module imports its
siblings flat (`import semip_logging`, `from instance import Instance`), so:

```python
import arctic_inference.semi_persistence   # ModuleNotFoundError: semip_logging
```

Code here only works with the `semi_persistence/` directory itself on
`sys.path`. That is why `tests/` and `scripts/` each begin with a
`sys.path.insert(0, <package dir>)` bootstrap. Anything you add must do the
same, or the import fix must land first.

## Layer map

Read bottom-up; each layer only knows about the one below it.

| Layer | File | Responsibility |
|---|---|---|
| Child | `vllm_child.py` | Owns the GPU. vLLM engine, pinned CPU buffer, async generate loop, CRIU dump prep |
| Worker | `worker.py` | Command loop; `cuCheckpointProcess` via ctypes; CRIU save/load |
| Handle | `instance.py` | `Instance`: chainable non-blocking primitives, one per vLLM config |
| Demux | `demuxer.py` | Sole consumer of the result queue; keeps `Instance` state fresh |
| Slots | `slots.py` | Buddy allocator handing out fractional GPU slots (metadata only) |
| Pipeline | `pipeline.py` | One worker thread + FIFO queue per model, ops as `Op` subclasses |
| Orchestrator | `orchestrator.py` | `model_id` -> `Instance`, the state ladder, eviction policy |
| Serving | `orch_server.py`, `client.py` | HTTP front end and the job-keyed client |
| Observability | `state_server.py`, `dashboard.py` | `/state` endpoint, curses dashboard |
| Support | `abstract.py`, `semip_logging.py` | `InstanceBase` interface, logging |

## Process hierarchy

Both the worker and the vLLM child are **spawned**, never forked — fork can
deadlock on glibc mutexes held by other threads, and spawn gives the child a
clean address space with no inherited CUDA context.

```
Main process
  |-- Instance (handle)
        `-- Worker process (spawn)      <- cuCheckpointProcess ctypes, CRIU
              `-- vLLM child (spawn)    <- owns GPU + pinned CPU buffer
                    `-- EngineCore (in-process, VLLM_ENABLE_V1_MULTIPROCESSING=0)
```

`EngineCore` runs in-process deliberately: as a subprocess it would pickle GPU
tensors across the boundary during `restore_weights`, which fails for models
over 4 GiB and costs ~16 s even for small ones. GPU memory queries use NVML
rather than `torch.cuda.mem_get_info`, to avoid initializing CUDA in the main
process.

## The state ladder

`move()` walks this ladder one rung at a time, in either direction:

```
saved  <->  checkpoint  <->  sleep  <->  up  ( -> running, transient)
```

| State | Image on disk | Live process | Slot held | CUDA context | Weights on GPU |
|---|---|---|---|---|---|
| `saved` | yes | no | no | no | no |
| `checkpoint` | yes | yes | no | no | no |
| `sleep` | yes | yes | usually | yes (small) | no |
| `up` | yes | yes | usually | yes | yes |
| `running` | yes | yes | **always** | yes | yes |

Two states are transient and are **not** valid `move()` targets: `running`
(exists only during a `generate()`), and `wait` (published while blocked in the
`Slots` FIFO during `checkpoint -> sleep`; it is purely a dashboard signal).

After a generate finishes, a model parks in **slotless `up`** — still warm, but
its slot is released, which is what makes it eligible for eviction.

## Slots in one paragraph

`Slots` is a singleton buddy allocator over GPUs, and pure metadata — it does no
CUDA work. A `Slot` is an immutable `(gpu_id, level, index)`; a level-`L` slot
covers `1 / 2^(L-1)` of a GPU, so level 1 is a whole GPU, level 2 a half, level
3 a quarter. A model's level derives from its `gpu_memory_utilization`. Nodes
split on allocation when no exact-size slot is free, and coalesce with their
buddy (`index ^ 1`) on deallocation. Waiters queue FIFO; free GPUs are picked
coldest-first by last-used time.

## Pipeline in one paragraph

Each `model_id` gets one worker thread and one FIFO queue. Operations are `Op`
subclasses — `RegisterOp`, `MoveOp`, `EvictForPeerOp`, `GenerateOp`, `PauseOp`,
`ResumeOp`, `RemoveOp` — and the FIFO is what orders them, replacing an older
implicit future-chaining design. Pause works by `interrupt_now()` plus
`submit_front()`, and long-running ops cooperate via `InterruptFlag`
(`raise_if_set()` / `wait_or_interrupt()`) instead of polling loops.
Cross-model eviction goes through `submit_to_peer_and_wait`, which carries
cycle detection.

## Where things live

```
semi_persistence/
  *.py           library code (orchestrator, instance, worker, client, dashboard, ...)
  tests/         pytest: CPU-only, hermetic, ~2.5s, no GPU
  scripts/       imperative repros: need real GPUs and real vLLM
  skills/        this skill (SKILL.md + reference.md) and every design doc
```

Running things:

```bash
cd arctic_inference/semi_persistence
python -m pytest tests/ -q        # 35 tests, no GPU needed
python scripts/test_env.py 0 1    # needs two real GPUs
python dashboard.py               # needs a running orchestrator on :8157
```

`tests/` is the only part runnable in CI. Everything in `scripts/` allocates
real GPUs and loads real weights.

## Gotchas that bite

- **CRIU dump is destructive.** The child is killed once the image is written,
  so after `save_image` the model is always `saved` with no live process.
- **`/usr/lib/criu/empty` must exist** or dump aborts at plugin init. See
  [`INSTALL.md`](INSTALL.md).
- **Reserved env keys** (`CUDA_VISIBLE_DEVICES`,
  `VLLM_ENABLE_V1_MULTIPROCESSING`, `USE_LIBUV`) are silently dropped from
  `vllm_config["_env"]` at apply time, but retained on disk in `meta.json`.
- **`load_image` does not re-apply `_env`** — the child's environment is baked
  into the CRIU image and restored verbatim.
- **Staging buffers must be freed via `storage().resize_(0)`**, not
  `caching_allocator_delete`, because vLLM's cumem allocator intercepts
  `torch.empty(..., device="cuda")`.

## Design docs

| Doc | Covers |
|---|---|
| [`instance_DESIGN.md`](instance_DESIGN.md) | Primitives table, process hierarchy, pin management, pause/resume |
| [`orchestrator_DESIGN.md`](orchestrator_DESIGN.md) | State machine, slot allocation, public API, Known Issues |
| [`pipeline_DESIGN.md`](pipeline_DESIGN.md) | Op model, interrupts, cross-model eviction, regression plan |
| [`slots_DESIGN.md`](slots_DESIGN.md) | Buddy allocator algorithms, invariants, worked example |
| [`client_DESIGN.md`](client_DESIGN.md) | Job/model two-layer split, calling shapes, session persistence |
| [`CRIU_PLUMBING.md`](CRIU_PLUMBING.md) | The nine CRIU complications and the FD keep-list |
| [`async_generate_DETAILS.md`](async_generate_DETAILS.md) | Async generate, IPC protocol, drain points |
| [`INSTALL.md`](INSTALL.md) | CRIU install (PPA and from-source), draft model sync |

## Related skills

- `arctic-inference-architecture` — the surrounding vLLM plugin.
