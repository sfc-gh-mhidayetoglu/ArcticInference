# Orchestrator -- Design

The orchestrator is a higher-level API on top of `Instance` primitives.
It maps human-readable model IDs to `Instance` objects and manages a
state machine for each model.  GPU resources are managed via a single
buddy allocator (`Slots`) that hands out fractional **slots**; each
model has an intrinsic *level* (derived from its
`gpu_memory_utilization`) and acquires a slot of that level when it
needs to run.

## State Machine

Models live on a single ordered ladder.  `move()` walks up or down,
executing each step in between.

```
                         ┌───────────────────────────┐
                         │       register()          │
                         │   (cold-start new model)  │
                         └─────────────┬─────────────┘
                                       │
                                       ▼
  ┌───────────┐  load   ┌────────────┐  alloc+restore  ┌───────────┐  alloc?+wake_up  ┌─────────┐  generate  ┌─────────┐
  │   saved   │ ──────► │ checkpoint │  repin          │   sleep   │  h2d+scatter    │   up    │ ─────────► │ running │
  │           │         │            │ ──────────────► │           │ ──────────────► │         │            │         │
  │ image     │         │ image +    │                 │ CUDA on   │                 │ image + │            │ weights │
  │ on disk   │ ◄────── │ live proc  │ ◄─────────────  │ GPU,      │ ◄─────────────  │ process │ ◄───────── │ on GPU, │
  │           │ teardown│            │  unpin+ckpt     │ slotted   │  sleep_weights  │ + slot  │  generate  │ slotted │
  └─────┬─────┘ remove  └────────────┘  +dealloc       │ or none   │  (slot stays)   └─────────┘  completes └─────────┘
        │                                              └───────────┘                       │
        ▼                                                                                  │
  ┌──────────┐                              ┌─────────────────────────────────────────────┘
  │ remove() │                              │
  │ delete   │                              ▼ slot deallocated when generate finishes
  │ image +  │                       ┌──────────────┐
  │ registry │                       │  up slotless │ (eligible for eviction)
  └──────────┘                       └──────────────┘


  move() walks the ladder:

      UP:    saved ──────► checkpoint ──────► sleep ──────► up
      DOWN:  saved ◄────── checkpoint ◄────── sleep ◄────── up

  generate() is transient:  (auto-up) ──► running ──► up (slotless)
```

### Upward sequence

```
saved
  load(image_dir)                          # CRIU restore from disk -> live process
checkpoint
  Slots.allocate(level=L)                  # block on FIFO queue, coldest-first auto-pick
  inst.restore(slot.gpu).repin()           # CUDA context on GPU, small footprint
sleep
  (slot already held)                      # tier A/B/C only kicks in if slotless
  inst.wake_up_weights().h2d()
  .scatter().wake_up_kv_cache()            # weights on GPU, ready to serve
up
  inst.generate(prompts)                   # transient sub-state of up
running
  (generate completes -> Slots.deallocate)
up (slotless)                              # parks here; can be evicted to make room
```

### Downward sequence

```
up (running or slotted)
  inst.sleep().wait()                      # free GPU mem (weights + KV cache)
sleep
  inst.unpin().checkpoint().wait()
  Slots.deallocate(slot) (if held)         # CUDA context torn down, slot released
checkpoint
  inst.teardown().wait().remove()          # kill process, keep image on disk
saved
  remove()                                 # delete image + registry entry
(gone)
```

### State definitions

| State        | Image on disk | Live process | Slot held       | CUDA context | Weights on GPU |
| ------------ | ------------- | ------------ | --------------- | ------------ | -------------- |
| `saved`      | yes           | no           | no              | no           | no             |
| `checkpoint` | yes           | yes          | no              | no           | no             |
| `sleep`      | yes           | yes          | usually (\*)    | yes (small)  | no             |
| `up`         | yes           | yes          | usually (\*\*)  | yes          | yes            |
| `running`    | yes           | yes          | **always**      | yes          | yes            |

(\*) `sleep` is normally slotted (allocated at `checkpoint -> sleep`
and held until either `running -> up` or `sleep -> checkpoint`
releases it).  It is **slotless** only when the user explicitly pins
a target GPU via `move(model_id, "sleep", target_gpu=G)` (see
"Slotless-sleep flavour" below).

(\*\*) `up` is slotted while running and immediately after climbing
from sleep.  It becomes **slotless** after `running -> up` deallocates
the slot, allowing other models to acquire it.  Slotless `up` models
are eligible for eviction back to `sleep`.

`running` is a transient sub-state of `up` that exists only during a
`generate()` call.  It is **not** a valid target for `move()` — the
move-validator raises `ValueError` if requested.

`wait` is a second transient state, published only while the model
is enqueued in the `Slots` FIFO during a `checkpoint -> sleep`
transition (i.e. blocked inside `Slots.allocate`).  It is not part
of the state ladder and not a valid `move()` target — it is purely a
dashboard signal and is replaced by `sleep` as soon as the slot is
acquired.

## Slot Allocation (`Slots`)

`Slots` is a singleton buddy allocator (`slots.py`).  Each GPU starts
as one whole slot at level 1.  Splitting a level-L slot yields two
level-(L+1) buddies that each cover half the area.  A level-L slot
covers `1 / 2**(L-1)` of a GPU.

Public API:

| Method                                  | Semantics |
| --------------------------------------- | --------- |
| `Slots.init(gpu_ids)`                   | Initialise pools, one root slot per GPU. |
| `Slots.allocate(level, gpu=None)`       | **Blocking, FIFO.**  Picks coldest GPU when `gpu is None`. |
| `Slots.try_allocate(level, gpu=None)`   | **Non-blocking.**  Returns `None` if not satisfiable now.  Bypasses FIFO. |
| `Slots.deallocate(slot)`                | Release and coalesce buddies upward. |

Coldest-first auto-pick is implemented inside `_try_allocate`: it
sorts GPUs by `(last_used_time, gpu_id)` and returns the first one
with a satisfiable subtree.  FIFO is enforced by a single
`_waiters` deque guarded by a condition variable.

### Per-model intrinsic level

Each registry entry carries a `level` property derived once at
discovery / registration time:

```python
def _pick_level(util):
    """Smallest L with 1 / 2**(L-1) >= util."""
    if util <= 0.0 or util >= 1.0:
        return 1
    return max(1, ceil(log2(1.0 / util)))
```

Boundaries:

| `gpu_memory_utilization`  | level | slot fraction |
| ------------------------- | ----- | ------------- |
| `(0.5, 1.0]`              | L1    | 1.0  (whole)  |
| `(0.25, 0.5]`             | L2    | 0.5  (half)   |
| `(0.125, 0.25]`           | L3    | 0.25 (quarter)|
| `(0.0625, 0.125]`         | L4    | 0.125 (eighth)|
| ...                       | ...   | ...           |

The level is a static property of the model, computed once from its
`vllm_config.gpu_memory_utilization` and stored as `entry["level"]`.
All future allocations for the model use this level.

### Slot lifecycle on the ladder

| Transition                | Slot action                                                    |
| ------------------------- | -------------------------------------------------------------- |
| `saved -> checkpoint`     | none                                                           |
| `checkpoint -> sleep`     | publish `wait` + `Slots.allocate(level)` (blocking, FIFO)      |
| `sleep -> up`             | tier A/B/C if slotless; else no-op (Phase 1) + evict slotless squatters until HBM fits (Phase 2) |
| `up -> running`           | tier A if slotless (else no-op); see "Slot acquisition at `up -> running`" |
| `running -> up`           | `Slots.deallocate(slot)`  → slotless `up`                      |
| `up -> sleep`             | none (slot, if any, stays)                                     |
| `sleep -> checkpoint`     | `Slots.deallocate(slot)` if held                               |
| `checkpoint -> saved`     | none                                                           |

Allocation and deallocation are **deliberately asymmetric**:
`checkpoint -> sleep` always allocates, but `running -> up` (not
`up -> sleep`) is the symmetric release point.  This lets one
slotless model wait at `up` for traffic without holding a slot,
while another sleeper takes that slot.

### Slot acquisition at `up -> running`

`running` always holds a slot.  A slotless `up` model (left there by
a previous `running -> up`) must reacquire a slot before publishing
`running` on the next generate:

1. **Tier A** -- `Slots.try_allocate(level, gpu=home_gpu)`.  If the
   home GPU is free, claim the slot in place.  Weights stay in HBM,
   no movement.  Common case for the second generate on a model
   that nobody has displaced.
2. **Fallback (migration / FIFO)** -- if Tier A fails, the model
   retreats to `sleep` (`_step_down(up, sleep)` frees HBM) and
   climbs back up via `_step_up(sleep, up)`, which uses the full
   tier-A/B/C path below (including Phase-2 eviction and the
   blocking FIFO `Slots.allocate`).

After this, the published state transitions `up -> running` with the
slot held.

### Tier A/B/C: slotless `sleep -> up`

If a sleeper has lost its slot (e.g. via the slotless-sleep flavour),
`_step_up` reacquires opportunistically before falling back to FIFO:

1. **Tier A** -- `Slots.try_allocate(level, gpu=home_gpu)`.  Fast
   path: same GPU, no migration.
2. **Tier B** -- `Slots.try_allocate(level)`.  Any free GPU.  If the
   returned slot is on a different GPU, migrate via
   `inst.unpin().checkpoint()` then `inst.restore(new_gpu).repin()`.
3. **Tier C** -- nothing free.  Retreat: `_step_down(sleep,
   checkpoint)` (no-op for slot, since slotless), then
   `_step_up(checkpoint, sleep)` which calls the blocking
   `Slots.allocate(level)` and respects FIFO.

### Phase 2: HBM eviction

After the slot is held on `home_gpu`, `_step_up` frees just enough
HBM for the new model's weights by evicting slotless `up`
incumbents on the same GPU, oldest first by `state_since`.  The
amount evicted is computed from the GPU's HBM budget, not a fixed
"evict one" or "evict all" rule:

- Each model's HBM footprint is its level share, `1 / 2**(L-1)`.
- `slotted_others` is the sum of slot shares on `home_gpu` for
  every *other* slotted model, **regardless of state**.  We use
  the slot share rather than live `up`/`running` state because slot
  allocation is what serialises concurrent wake-ups: a slotted
  `sleep` model whose Phase 3 is already in flight on the worker
  will be HBM-resident by the time *our* Phase 3 commands run,
  even though its registry state still reads `sleep`.  Counting
  every slot is the only race-free upper bound; `slotted_others`
  may slightly over-account when a slotted sleeper is genuinely
  not waking, but that just causes a small over-eviction, never
  an OOM.  "On `home_gpu`" is keyed off `slot.gpu_id` (the buddy
  allocator's source of truth, mirrored by `Slots._live`), not the
  registry's `entry["gpu"]`: during a peer's Tier B migration the
  slot flips to its new GPU before `entry["gpu"]` does, and only
  the slot reflects where the share is actually owed.
- `new_share` is the slot share we just allocated for this model.
  After Phase 3 those weights will be resident, so they consume
  `new_share` of HBM.
- `slack = 1.0 - slotted_others - new_share` is the HBM the buddy
  allocator has *not* handed out — exactly the budget available
  for slotless squatters.  Because the buddy allocator keeps the
  sum of all slot shares on a GPU at `<= 1.0`, `slack` is always
  `>= 0`.

`_step_up` then sorts slotless `up` incumbents on `home_gpu` by
`state_since` (oldest first) and evicts them one by one back to
`sleep` until the remaining slotless share fits inside `slack`.
This evicts the minimum number of incumbents needed and prefers
the coldest victims.  Phase 3
(`wake_up_weights().h2d().scatter().wake_up_kv_cache().wait()`)
has no try/except, so under-eviction would OOM and crash the
orchestrator; the formula above is the precise condition for
Phase 3 to fit within the GPU under any concurrent wake-up
pattern the buddy allocator allows.

Worked example: an L2 model (`new_share=0.5`) wakes up on a GPU
hosting two slotless L3 squatters (each `0.25`), one slotless L2
squatter (`0.5`), and no other slotted models (`slotted_others=0`).
Then `slack=0.5` and the slotless total is `1.0`.  If the two L3
squatters are oldest, evicting both frees `0.5`, leaving the
slotless L2 in place — exactly enough to fit.

Concurrent-wake example: two L2 models `A` and `B` are both being
woken on the same GPU.  After Phase 1 the GPU's slots are
`A`(0.5) + `B`(0.5) = 1.0.  When `B`'s Phase 2 runs, `A` may
still be in registry state `sleep` (its Phase 3 commands are
queued but `_set_state(up)` hasn't fired yet).  Counting only
"up/running slotted" would yield `slack=0.5` and skip evicting a
slotless squatter `C`, then `A` and `C` and `B` would all try to
fit (1.5 total → OOM).  Counting every slot gives
`slotted_others=0.5`, `slack=0`, forcing `C` to be evicted.

### Slotless-sleep flavour

`move(model_id, "sleep", target_gpu=G)` parks a model on GPU `G`
**without allocating a slot**:

- `_step_up`'s `checkpoint -> sleep` skips `Slots.allocate` when
  `target_gpu` is given, and just calls `inst.restore(G).repin()`.
- A tail in `_move_sync` deallocates any slot the model still holds
  after the ladder walk (e.g. coming from `up` on the same GPU, or
  already sleeping but slotted).

The model competes for HBM but not slots.  When woken later via
`generate()` or `move("up")`, the standard tier-A/B/C logic
re-acquires a slot.

### What `register()` allocates

Cold-start always uses a full GPU regardless of the model's eventual
level: `Slots.allocate(level=1)`.  This `register_slot` is local to
`_register_sync` and is released inline once the image is saved.
The model's intrinsic `level` (used in subsequent
`checkpoint -> sleep` transitions) is computed from its
`gpu_memory_utilization` at the same time.

## API

```python
Orchestrator.init(image_cache="/data-fast/image-cache")

Orchestrator.register("qwen3-8b",
                      {"model": "Qwen/Qwen3-8B",
                       "gpu_memory_utilization": 0.4})  # -> intrinsic level L2
Orchestrator.wait()

# generate auto-transitions to 'up' if needed; leaves model in 'up'
# (slotless) after running.
fut = Orchestrator.generate("qwen3-8b", "Hello, how are you?")
results = fut.result()

# explicit state control
Orchestrator.move("qwen3-8b", "checkpoint")
Orchestrator.wait()

# slotless-sleep flavour: park on a specific GPU without using a slot
Orchestrator.move("qwen3-8b", "sleep", target_gpu=5)
Orchestrator.wait()

# remove auto-transitions to 'saved' if needed
Orchestrator.remove("qwen3-8b")
Orchestrator.wait()

Orchestrator.status()
```

## `init(image_cache, gpus=None)`

- Discovers GPUs via NVML (`pynvml`) without initializing CUDA in the
  main process.
- Resets `Slots` if it was already initialised (test re-entry safety),
  then calls `Slots.init(gpu_ids)`.
- Scans `image_cache` for subdirectories containing `meta.json`.
  Each entry is registered in `saved` state and tagged with its
  intrinsic `level` (computed from
  `meta.json -> vllm_config.gpu_memory_utilization`).
- Initializes the `ThreadPoolExecutor`.

## `register(model_id, vllm_config)`

Non-blocking.  Submits `_register_sync` to the thread pool.

1. **Acquire a full GPU** via `Slots.allocate(level=1)` (blocks on
   FIFO).
2. **Cold-start sequence** --
   `Instance(vllm_config).init(gpu).attach().repin().stage().unpin().sleep().checkpoint().wait()`.
3. **Save image** -- `inst.save(image_dir).wait()` writes the CRIU
   image to disk.  The dump is destructive — the child is killed.
4. **Release slot** -- inline `Slots.deallocate(register_slot)`.
5. **Store in registry** with intrinsic `level` derived from
   `gpu_memory_utilization`; state set to `saved` (image on disk,
   no live process, no slot).

With N GPUs, up to N models cold-start in parallel.

## `move(model_id, target, target_gpu=None)`

Walks the state ladder from the current state to the target.  Valid
targets: `"saved"`, `"checkpoint"`, `"sleep"`, `"up"`.  Also called
internally by `generate()` and `remove()`.  Non-blocking.

- **Going up / down**: walks the ladder one step at a time via
  `_step_up` / `_step_down`.
- **Already at target**: no-op (unless `target_gpu` is given, see
  "Slotless-sleep flavour" above).
- **Model in `running` state**: raises `RuntimeError` (wait for
  generate to finish first).
- **`target == "running"`**: rejected with `ValueError` (running is
  only reachable via `generate()`).
- **`target_gpu` set with `target != "sleep"`**: rejected.
- **Model not registered**: prints a warning and returns (no error).

### Step details

| Transition                | Instance primitives + slot action                                                                        |
| ------------------------- | -------------------------------------------------------------------------------------------------------- |
| `saved -> checkpoint`     | `Instance(config).load(image_dir).wait()`                                                                |
| `checkpoint -> sleep`     | (alloc) `Slots.allocate(level)` + `inst.restore(slot.gpu).repin().wait()`                                |
| `sleep -> up`             | Phase 1 tier A/B/C (alloc if needed, possibly migrate); Phase 2 evict oldest slotless incumbents on same GPU until HBM fits (`slack = 1 - active_others - new_share`); Phase 3 `inst.wake_up_weights().h2d().scatter().wake_up_kv_cache().wait()` |
| `up -> sleep`             | `inst.sleep().wait()` (slot, if any, stays)                                                              |
| `sleep -> checkpoint`     | `inst.unpin().checkpoint().wait()` + `Slots.deallocate(slot)` if held; clear `entry["gpu"]`              |
| `checkpoint -> saved`     | `inst.teardown().wait().remove()`                                                                        |
| `up -> running`           | (occurs inside `_move_sync` announce branch) Tier A `Slots.try_allocate`, else fallback through `sleep -> up` |
| `running -> up`           | (occurs inside `_start_generate_waiter`, not `_step_down`) `Slots.deallocate(slot)` after queue drains    |

The `move(.., "sleep", target_gpu=G)` flavour is implemented as a
combination of (a) skipping `Slots.allocate` in `checkpoint -> sleep`
when `target_gpu` is set and (b) a tail in `_move_sync` that
deallocates any leftover slot.

## `generate(model_id, prompts, sampling_params=None)`

Non-blocking.  Submits `_generate_sync` to the thread pool and returns
a `Future[list]`.  Can be called from **any** state.

1. **Set t0 (first generate only)** -- anchors `t0` for relative
   request timestamps (`submit_rel_s`, etc.).
2. **Auto-transition to `up`** -- if not already there, walks the
   ladder up via `_move_sync(model_id, "up", announce_state="running")`.
3. **State becomes `running`** -- atomic with the final `_step_up`.
4. **Drain queue and submit** -- batch all queued requests for this
   model under `gen_lock` and submit to the worker.
5. **Wait for results** (lock released).
6. **`_start_generate_waiter`** drains `inst._result_queue`.  When
   no in-flight requests remain and `inst._pending_count == 0`:
     - `Slots.deallocate(entry["slot"])` (if any),
     - state ← `"up"` (slotless),
     - waiter exits.

The model is left in **slotless `up`** after generate.  Other models
that need a slot can acquire it (and may evict this one back to
`sleep` via Phase 2 of `_step_up`).

## `remove(model_id=None)`

Non-blocking.  Automatically walks the model down to `saved` state
(via `_move_sync`), then deletes the image directory from disk and
removes the model from the registry.  Pass `None` to remove all
models.  All slot bookkeeping is handled by the ladder walk
(specifically, `sleep -> checkpoint` deallocates).

## `wait(model_id=None)`

Blocks the main thread until futures complete.  If `model_id` is given,
waits on that model only.  Otherwise waits on all outstanding futures.

## `status()`

Prints orchestrator config, GPU memory usage, and a compact one-line
summary per model sorted by pinned memory (largest first).  Each line
shows state, GPU assignment, pinned memory (for non-saved), actual
GPU memory usage via NVML per-PID lookup (for up), and image path
(for saved).  A lightweight `_print_states()` one-liner is also
printed automatically after every state change.

## Non-Blocking Design -- Why Threading

The orchestrator uses `ThreadPoolExecutor`, not `asyncio`.

- The actual work runs in **forked worker processes** (see
  `instance_DESIGN.md`).  The pool thread just enqueues Instance
  primitive commands and blocks on `instance.wait()`.  There is no
  async I/O to benefit from coroutines.
- The existing codebase uses multiprocessing + threading exclusively.
- The caller (demo.py) stays plain synchronous Python.
- `ThreadPoolExecutor.submit()` returns a `Future` that maps cleanly
  to the existing `wait()` pattern.

### Thread-per-model-ID model

Each public method (`register`, `generate`, `remove`) and internal
`move` submit work to the thread pool.  Within a model ID, operations
are serialized via future chaining (`prev_future.result()`).  Across
model IDs, everything runs concurrently — `Slots` provides the only
cross-model synchronisation point.

```
Main thread           Pool threads              Worker processes
-----------           ------------              ----------------
register("a") -----> Thread 1: _register_sync  --> Instance A worker (gpu0, L1 slot)
register("b") -----> Thread 2: _register_sync  --> Instance B worker (gpu1, L1 slot)
register("c") -----> Thread 3: _register_sync  --> (FIFO-blocks on Slots.allocate)
  |                       |
wait()                    |
  |                   Thread 3: _register_sync  --> Instance C worker (gpu0)
  |                       |
generate("b") -----> Thread 4: _generate_sync
                       auto: move_sync("up", announce_state="running")
                         _step_up: saved -> checkpoint  (load)
                         _step_up: checkpoint -> sleep  (Slots.allocate(L=b.level))
                         _step_up: sleep -> up          (tier A/B/C; evict
                                                         slotless up on same gpu;
                                                         wake_up + h2d + scatter)
                       inst.generate -> _start_generate_waiter
                         drains queue, then:
                           Slots.deallocate(slot)
                           state -> "up" (slotless)
                       |
remove("b") --------> Thread 5: _remove_sync
                       auto: move_sync("saved")
                         _step_down: up -> sleep        (no slot move)
                         _step_down: sleep -> checkpoint (Slots.deallocate)
                         _step_down: checkpoint -> saved (teardown)
                       delete image, pop registry
```

## Console Output

The orchestrator separates verbose logs from high-level status messages.

**Verbose logs** go to `stdout` (which `demo.py` redirects to a log
file via `os.dup2`).  These include detailed orchestrator internals,
worker/child process output, and library warnings.

**Console messages** go through the `_console()` helper, which writes
to a separate stream that stays visible to the user.  Console messages
are concise, one-line status updates:

```
qwen3-8b: register received
qwen3-8b: registered (wait=0.0s, cold-start=45.2s, total=45.2s)
  models: qwen3-8b[saved]
qwen3-8b: generate received
qwen3-8b: placed on GPU 0
qwen3-8b: saved -> up (3.1s)
  models: qwen3-8b[up:gpu0]
qwen3-8b: generated (1.2s) -> "Albert Einstein..."
qwen3-8b: remove received
qwen3-8b: up -> saved (2.4s)
  models: qwen3-8b[saved]
qwen3-8b: removed
  models: (none)
```

### `_console()` stream resolution

`_console()` writes to the first available of:

1. `_console_stream` -- module-level override (set by Jupyter notebooks
   to `sys.stderr` so output appears in the active cell).
2. `_terminal` -- a dup of `sys.stderr` taken at import time, before
   any `os.dup2` redirection.  In `demo.py`, this preserves the
   original terminal fd.
3. `sys.stderr` -- fallback.

### Output redirection in `demo.py`

`demo.py` redirects fd 1 (stdout) and fd 2 (stderr) to a timestamped
log file via `os.dup2`.  Forked worker processes inherit these
redirected fds, so all child output (vLLM, CUDA warnings, tqdm) goes
to the log.  `_terminal` (duped before redirection) keeps console
messages on the real terminal.

### Output redirection in `demo.ipynb`

Jupyter's `sys.stderr` is a ZMQ `OutStream` that routes to the active
cell.  The notebook sets `orchestrator._console_stream = sys.stderr`
after importing, so `_console()` output appears in the cell that runs
the code.

## Process Cleanup

Teardown kills the entire process tree (worker + vLLM child +
descendants) using `_kill_process_tree`, which walks descendants
bottom-up via `psutil` and sends SIGKILL.  `Instance._reset()` also
force-kills the worker if it doesn't exit within the join timeout.

## Dashboard

The dashboard (`dashboard.py`) is a standalone curses terminal UI that
polls `GET /state` from the orchestrator's embedded HTTP server
(`state_server.py`).  Each model row shows its memory footprint and,
when applicable, its slot level:

- **Live rows** (`sleep`, `up`, `running`): `"(67.2G, L2)"` when the
  model holds a slot, `"(67.2G)"` when slotless.
- **Image cache rows** (`saved`): `"(40.0G, L2)"` showing the
  model's intrinsic level (the level it *would* allocate at when
  woken).

The slot allocator state is exposed under `/state -> "slots"` as a
per-GPU list of leaves (`{level, index, alloc}`) so the dashboard can
render the buddy tree as a horizontal bar.  `slot_waiters` reports
the size of the FIFO waiter queue.

### Recording and replay

`--record FILE` writes each polled state snapshot to a JSONL file.
Recording starts only after the first generate request appears in
the state.  `--replay FILE` replays a recorded JSONL file in real
time.

### Timestamps

All relative timestamps in the state snapshot (`submit_rel_s`,
`gen_start_rel_s`, `done_rel_s`) are relative to `t0`, which is
anchored to the first `generate()` call (not orchestrator init).

## File Structure

```
semi_persistence/
  orchestrator.py          -- Orchestrator class
  orchestrator_DESIGN.md   -- This file
  slots.py                 -- Buddy allocator (Slots singleton)
  slots_DESIGN.md          -- Buddy allocator design
  instance.py              -- Instance class (see instance_DESIGN.md)
  worker.py                -- Worker process + child thread
  vllm_child.py            -- Spawned vLLM child process
  state_server.py          -- Embedded HTTP server (GET /state)
  dashboard.py             -- Curses dashboard (--record / --replay)
  demo.py                  -- CLI demo script (stdout/stderr -> log file)
  demo.ipynb               -- Jupyter notebook demo
  CRIU_PLUMBING.md         -- CRIU complications and fixes
```
