# Orchestrator -- Design

The orchestrator is a higher-level API on top of `Instance` primitives.
It maps human-readable model IDs to `Instance` objects and manages a
state machine for each model, with GPU assignment handled by state
transitions.

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
  ┌───────────┐  load   ┌────────────┐  restore  ┌───────────┐  acquire  ┌─────────┐  generate  ┌─────────┐
  │   saved   │ ──────► │ checkpoint │  repin    │   sleep   │  gpu     │   up    │ ─────────► │ running │
  │           │         │            │ ────────► │           │  wake_up │         │            │         │
  │ image     │         │ image +    │           │ CUDA on   │  h2d     │ image + │            │ weights │
  │ on disk   │ ◄────── │ live proc  │ ◄──────── │ GPU,      │  scatter │ process │ ◄───────── │ on GPU, │
  │           │ teardown│            │  unpin    │ small     │ ───────► │ + GPU   │  generate  │ serving │
  └─────┬─────┘ remove  └────────────┘  ckpt    │ footprint │          └─────────┘  completes └─────────┘
        │                                        │ GPU free  │
        ▼                                        └───────────┘
  ┌──────────┐
  │ remove() │
  │ delete   │
  │ image +  │
  │ registry │
  └──────────┘


  move() walks the ladder:

      UP:    saved ──────► checkpoint ──────► sleep ──────► up
      DOWN:  saved ◄────── checkpoint ◄────── sleep ◄────── up

  generate() is transient:  sleep ──► up ──► running ──► up ──► sleep
```

### Upward sequence (each arrow = one step)

```
saved
  load(image_dir)                          # CRIU restore from disk -> live process
checkpoint
  peek_coldest_gpu (no acquire)
  restore(gpu).repin()                     # CUDA context on GPU, small footprint
sleep
  acquire home GPU from pool
  wake_up_weights().h2d()
  .scatter().wake_up_kv_cache()            # weights on GPU, ready to serve
up
  generate(prompts)                        # transient
running
  (generate completes -> step down to sleep)
```

### Downward sequence

```
up
  sleep().wait()                           # free GPU mem (weights + KV cache)
  release_gpu                              # GPU back to pool, keep entry["gpu"] as home
sleep
  unpin().checkpoint().wait()              # unregister pinned mem, freeze CUDA state
  clear entry["gpu"]
checkpoint
  teardown().wait().remove()               # kill process, keep image on disk
saved
  remove()                                 # delete image + registry entry
(gone)
```

### State definitions

| State        | Image on disk | Live process | GPU held | CUDA context | Weights on GPU |
| ------------ | ------------- | ------------ | -------- | ------------ | -------------- |
| `saved`      | yes           | no           | no       | no           | no             |
| `checkpoint` | yes           | yes          | no       | no           | no             |
| `sleep`      | yes           | yes          | no       | yes (small)  | no             |
| `up`         | yes           | yes          | yes      | yes          | yes            |
| `running`    | yes           | yes          | yes      | yes          | yes            |

`sleep` has a CUDA context restored on a GPU with a small memory
footprint (~few GB) but does **not** lock the GPU.  Multiple sleep
models can coexist with an `up` model on the same GPU.  The model
remembers its "home GPU" for fast wake-up.

`running` is a transient sub-state of `up` that only exists during a
`generate()` call.  It is not a valid target for `move()`.

## API

```python
Orchestrator.init(image_cache="/data-fast/image-cache")

Orchestrator.register("qwen3-8b", {"model": "Qwen/Qwen3-8B", "gpu_memory_utilization": 0.4})
Orchestrator.wait()

# generate auto-transitions to 'up' if needed, leaves model in 'sleep'
fut = Orchestrator.generate("qwen3-8b", "Hello, how are you?")
results = fut.result()

# explicit state control
Orchestrator.move("qwen3-8b", "checkpoint")
Orchestrator.wait()

# remove auto-transitions to 'saved' if needed
Orchestrator.remove("qwen3-8b")
Orchestrator.wait()

Orchestrator.status()
```

## `init(image_cache, gpus=None)`

- Discovers GPUs via `nvidia-smi` (never initializes CUDA in the main
  process).
- Scans `image_cache` for subdirectories containing `meta.json`.  Each
  one becomes a registry entry in `saved` state.
- Initializes the `ThreadPoolExecutor` and GPU pool.

## `register(model_id, vllm_config)`

Non-blocking.  Submits `_register_sync` to the thread pool.

1. **Acquire a GPU** -- blocks on the GPU pool until one is free.
2. **Cold-start sequence** --
   `Instance(vllm_config).init(gpu).attach().repin().stage().unpin().sleep().checkpoint().wait()`.
3. **Save image** -- `inst.save(image_dir).wait()` writes the CRIU
   image to disk.  The dump is destructive — the child is killed.
4. **Clean up worker** -- the worker process exits.
5. **Release GPU** -- returns the GPU to the pool.
6. **Store in registry** -- state is set to `saved` (image on disk,
   no live process).

With N GPUs, up to N models cold-start in parallel.

## `move(model_id, target)`

Walks the state ladder from the current state to the target.  Valid
targets: `"saved"`, `"checkpoint"`, `"sleep"`, `"up"`.  Also called
internally by `generate()` and `remove()`.

Non-blocking.  Submits `_move_sync` to the thread pool.

The implementation uses an ordered list `_STATES = ["saved",
"checkpoint", "sleep", "up"]` and iterates through intermediate steps:

- **Going up** (e.g. `saved` -> `up`): executes `saved` -> `checkpoint`
  -> `sleep` -> `up`.
- **Going down** (e.g. `up` -> `saved`): executes `up` -> `sleep`
  -> `checkpoint` -> `saved`.
- **Already at target**: no-op.
- **Model in `running` state**: raises `RuntimeError` (wait for
  generate to finish first).
- **Model not registered**: prints a warning and returns (no error).

### Resilient upward stepping

The upward path uses a `while` loop that re-reads `entry["state"]`
after each step, rather than a pre-computed `for` loop.  This handles
concurrent evictions: if another thread moves a model backward
mid-climb (e.g. via `SP_DEMO_MODE` eviction to `saved`), the loop
detects the regression via a `_StateRegressed` exception from
`_step_up` and re-plans from the model's actual current state.

Without this, a stale `from_state` would cause `_step_up` to access
`entry["gpu"]` when it is `None`, resulting in `KeyError`.

### Demo-mode eviction resilience

When `SP_DEMO_MODE=1`, the `sleep -> up` transition evicts victims all
the way to `saved` via `_move_sync(victim_id, "saved")`.  These calls
are fire-and-forget and may race with the victim's own climbing thread.
Both eviction sites (home-GPU eviction and migration-path eviction) are
wrapped in `try/except` — failures are logged as warnings rather than
propagated.

### Step details

| Transition                  | Instance primitives                                                                |
| --------------------------- | ---------------------------------------------------------------------------------- |
| `saved` -> `checkpoint`    | `Instance(config).load(image_dir).wait()`                                          |
| `checkpoint` -> `sleep`    | `peek_coldest_gpu` + `inst.restore(gpu).repin().wait()`                            |
| `sleep` -> `up` (home free)| `acquire home GPU` + `inst.wake_up_weights().h2d().scatter().wake_up_kv_cache().wait()` |
| `sleep` -> `up` (home busy)| `_step_down(sleep, checkpoint)` + `acquire_gpu` + `inst.restore(gpu).repin().wake_up_weights().h2d().scatter().wake_up_kv_cache().wait()` |
| `up` -> `sleep`            | `inst.sleep().wait()` + `release_gpu` (keep home GPU ref)                          |
| `sleep` -> `checkpoint`    | `inst.unpin().checkpoint().wait()` (clear GPU ref)                                 |
| `checkpoint` -> `saved`    | `inst.teardown().wait().remove()`                                                  |

GPU acquire happens in `sleep` -> `up`.  GPU release happens in
`up` -> `sleep`.  `checkpoint` -> `sleep` peeks at the coldest GPU
without acquiring (sleep doesn't lock a GPU).

## `generate(model_id, prompts, sampling_params=None)`

Non-blocking.  Submits `_generate_sync` to the thread pool and returns
a `Future[list]`.

Can be called from **any** state.  If the model is not already `up`,
generate automatically walks the ladder up first (via `move`).
The model is left in **sleep** state after generate completes (GPU
released).

1. **Set t0 (first generate only)** -- the first generate call anchors
   `t0` for all relative request timestamps (`submit_rel_s`, etc.).
2. **Auto-transition to `up`** -- if not already there, walks the
   ladder up (e.g. `saved` -> `checkpoint` -> `sleep` -> `up`).
3. **Set state to `running`** -- prevents concurrent `move()` calls.
4. **Run inference** -- `inst.generate(prompts, sampling_params).wait()`.
5. **Step down to `sleep`** -- `inst.sleep().wait()`, release GPU.
6. **Return results** -- `inst.last_generate_result`.

## `remove(model_id=None)`

Non-blocking.  Automatically walks the model down to `saved` state if
needed (via `move`), then deletes the image directory from disk
and removes the model from the registry.

Pass `None` to remove all models.

## `wait(model_id=None)`

Blocks the main thread until futures complete.  If `model_id` is given,
waits on that model only.  Otherwise waits on all outstanding futures.

## `status()`

Prints orchestrator config, GPU memory usage, and a compact one-line
summary per model sorted by pinned memory (largest first).  Each line
shows state, GPU assignment, pinned memory (for checkpoint/up), actual
GPU memory usage via nvidia-smi PID lookup (for up), and image path
(for saved).

A lightweight `_print_states()` one-liner is also printed automatically
after every state change (register, move, remove).

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
`move` submit work to the thread pool.  Within a model ID, operations are serialized
via future chaining (`prev_future.result()`).  Across model IDs,
everything runs concurrently.

```
Main thread           Pool threads              Worker processes
-----------           ------------              ----------------
register("a") -----> Thread 1: _register_sync  --> Instance A worker (gpu0)
register("b") -----> Thread 2: _register_sync  --> Instance B worker (gpu1)
register("c") -----> Thread 3: _register_sync  --> (waits for GPU...)
  |                       |
wait()                    |
  |                   Thread 3: _register_sync  --> Instance C worker (gpu0)
  |                       |
generate("b") -----> Thread 4: _generate_sync
                       auto: move("up")
                         _step_up: checkpoint -> sleep
                         peek coldest GPU, restore, repin
                         _step_up: sleep -> up
                         acquire home GPU, wake_up, h2d, scatter, wake_up_kv
                       generate -> wait
                       _step_down: up -> sleep
                         sleep, release GPU
                       |
remove("b") --------> Thread 5: _remove_sync
                       auto: move("saved")
                         _step_down: sleep -> checkpoint
                         unpin -> checkpoint
                         _step_down: checkpoint -> saved
                         teardown -> remove process
                       delete image, pop registry
```

## GPU Assignment

GPUs are managed via a pool of `(release_time, gpu_id)` tuples
protected by a `threading.Condition`.

**Coldest-first selection**: when multiple GPUs are free,
`_acquire_gpu` picks the one that has been idle the longest (smallest
`release_time`).  On tie, smallest GPU index wins.  This is a simple
`min()` over `(release_time, gpu_id)` tuples.

**FCFS blocking**: when no GPU is free, waiting threads block on the
condition variable and are woken one at a time as GPUs are released
(`notify()`), providing approximate first-come-first-served ordering.

`init()` populates the pool with `(0.0, gpu_id)` for each GPU (idle
since the beginning = coldest).  `_release_gpu` appends
`(time.perf_counter(), gpu_id)` and notifies one waiter.

GPU acquire is done in the `sleep` -> `up` transition.  GPU release
is done in the `up` -> `sleep` transition.  `checkpoint` -> `sleep`
uses `_peek_coldest_gpu` which touches the GPU's timestamp without
acquiring, so multiple sleep models can share a GPU.  `register()`
also acquires/releases a GPU for the initial cold-start.

With N GPUs, up to N models can be in `up` state simultaneously.
Many more models can be in `sleep` state, sharing GPUs with a small
footprint.

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
qwen3-8b: waiting for GPU...
qwen3-8b: acquired GPU 0
qwen3-8b: registered (wait=0.0s, cold-start=45.2s, total=45.2s)
  models: qwen3-8b[checkpoint]
qwen3-8b: generate received
qwen3-8b: checkpoint -> up (3.1s)
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
the code.  Verbose output is suppressed via environment variables
(`VLLM_LOGGING_LEVEL=ERROR`, `FLASHINFER_LOG_LEVEL=ERROR`,
`PYTHONWARNINGS=ignore`) and `logging.disable(logging.WARNING)`, set
before any library imports.

## Process Cleanup

Teardown kills the entire process tree (worker + vLLM child +
descendants) using `_kill_process_tree`, which walks descendants
bottom-up via `psutil` and sends SIGKILL.  `Instance._reset()` also
force-kills the worker if it doesn't exit within the join timeout.

## Dashboard

The dashboard (`dashboard.py`) is a standalone curses terminal UI that
polls `GET /state` from the orchestrator's embedded HTTP server
(`state_server.py`).

### Recording and replay

`--record FILE` writes each polled state snapshot to a JSONL file.
Recording starts only after the first generate request appears in the
state (so the file is not filled with idle snapshots).  If the
dashboard loses its connection to the orchestrator, recording stops
automatically and the file is closed.

`--replay FILE` replays a recorded JSONL file in real time, mimicking
the live poller.  Useful for reviewing past runs without the
orchestrator running.

### Timestamps

All relative timestamps in the state snapshot (`submit_rel_s`,
`gen_start_rel_s`, `done_rel_s`) are relative to `t0`, which is
anchored to the first `generate()` call (not orchestrator init).
The dashboard recording `t` field is relative to the first snapshot
that contains requests.

## File Structure

```
semi_persistence/
  orchestrator.py          -- Orchestrator class
  orchestrator_DESIGN.md   -- This file
  instance.py              -- Instance class (see instance_DESIGN.md)
  worker.py                -- Worker process + child thread
  vllm_child.py            -- Spawned vLLM child process
  state_server.py          -- Embedded HTTP server (GET /state)
  dashboard.py             -- Curses dashboard (--record / --replay)
  demo.py                  -- CLI demo script (stdout/stderr -> log file)
  demo.ipynb               -- Jupyter notebook demo
  CRIU_PLUMBING.md         -- CRIU complications and fixes
```
