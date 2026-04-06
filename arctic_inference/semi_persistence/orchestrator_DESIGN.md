# Orchestrator -- Design

The orchestrator is a higher-level API on top of `Instance` primitives.
It maps human-readable model IDs to `Instance` objects and manages GPU
assignment, HF checkpoint downloading, and the cold-start sequence.

## API

```python
Orchestrator.init(local_cache="/data-fast/model-cache")

Orchestrator.register("qwen3.5-9b", {"model": "Qwen/Qwen3.5-9B", "gpu_memory_utilization": 0.4})
Orchestrator.register("qwen3.5-2b", {"model": "Qwen/Qwen3.5-2B", "gpu_memory_utilization": 0.2})

Orchestrator.wait()

fut = Orchestrator.generate("qwen3.5-9b", ["Hello"], {"max_tokens": 10})
results = fut.result()

Orchestrator.print_status()
Orchestrator.remove("qwen3.5-9b")
```

## `init(local_cache, gpu_carveout_gb=0)`

- Discovers GPUs via `nvidia-smi` (never initializes CUDA in the main
  process).
- Creates the local cache directory.
- Initializes the `ThreadPoolExecutor` used by `register`, `generate`,
  and `remove`.

## `register(model_id, vllm_config)`

Two-phase design: download is **blocking** (main thread), GPU work is
**non-blocking** (pool thread).

### Phase 1: Download (main thread, blocking)

1. **Download HF checkpoint** -- `snapshot_download(hf_model,
   local_dir=<local_cache>/<hf_model>)`.  If the checkpoint already
   exists locally, this is a fast no-op (~0.1s, hash-verified).
2. **Rewrite vllm_config** -- creates a copy with `model` set to the
   local path.

Downloads run in the main thread to avoid concurrent download
contention and to ensure model files are available before submitting
GPU work to the thread pool.

### Phase 2: GPU cold-start (pool thread, non-blocking)

Submits `_register_sync` to the thread pool.  The future is stored in
`_futures[model_id]`.

1. **Acquire a GPU** -- waits on a `threading.Condition` until a GPU is
   available, then picks the one with the most free memory.  Multiple
   models can init in parallel on different GPUs.
2. **Cold-start sequence** --
   `Instance(vllm_config).init(gpu).attach().repin().stage().unpin().sleep().checkpoint().wait()`.
   - `init(gpu)` -- spawn vLLM with real weights on the GPU.
   - `attach()` -- allocate unpinned CPU buffer sized to model params.
   - `repin()` -- `cudaHostRegister` the buffer for DMA transfers.
   - `stage()` -- snapshot GPU model params into the CPU buffer.
   - `unpin()` -- `cudaHostUnregister` the buffer (speeds up restore).
   - `sleep()` -- free GPU memory (vLLM sleep mode).
   - `checkpoint()` -- CUDA-checkpoint the process (freeze it).
3. **Release GPU** -- discards the GPU from `_assigned_gpus` and
   notifies waiting threads.
4. **Store in registry** -- `_registry[model_id] = {"instance": inst,
   "vllm_config": vllm_config}`.

After registration, the instance is checkpointed with weights staged
in unpinned CPU memory.  The GPU is free for other models.

With N GPUs, up to N models cold-start in parallel.  Additional models
queue on the GPU condition variable and proceed as GPUs free up.

## `generate(model_id, prompts, sampling_params)`

Non-blocking.  Submits `_generate_sync` to the thread pool and returns
a `Future[list]`.

### `_generate_sync` (runs in a pool thread)

1. **Wait for prior future** -- calls `prev_future.result()` to
   ensure any prior operation on this model ID has completed.
2. **Acquire a GPU** -- same condition variable pattern as register.
3. **Restore and run inference** (timed as `restore` phase) --
   ```
   restore(gpu)          -- unfreeze checkpointed process onto GPU
   repin()               -- cudaHostRegister the CPU buffer for DMA
   wake_up_weights()     -- re-allocate weight tensors on GPU
   h2d()                 -- copy staged weights from CPU buffer → GPU
   scatter()             -- place weights into model parameters
   wake_up_kv_cache()    -- re-allocate KV cache on GPU
   wait()                -- block until ready
   ```
4. **Generate** (timed as `generate` phase) --
   `generate(prompts, sp).wait()`.
5. **Re-checkpoint** (timed as `checkpoint` phase) --
   `unpin().sleep().checkpoint().wait()` to unregister pinned memory,
   free GPU memory, and re-freeze.
6. **Release GPU** -- notify waiting threads.
7. **Return results** -- `inst.last_generate_result`, a list of lists
   (one inner list per prompt, multiple outputs if `n > 1`).

## `wait(model_id=None)`

Blocks the main thread until futures complete.  If `model_id` is given,
waits on that model only.  Otherwise waits on all outstanding futures.

## `remove(model_id=None)`

Waits for any pending registration, then tears down the instance
(kills the worker process tree) and removes it from the registry.

If `model_id` is `None`, removes **all** registered models.

## `print_status()`

Prints orchestrator config, GPU memory usage, and registered models
with their instance details (id, gpu, state, pid, pinned memory).

## Console Output

The orchestrator separates verbose logs from high-level status messages.

**Verbose logs** go to `stdout` (which `demo.py` redirects to a log
file via `os.dup2`).  These include detailed orchestrator internals,
worker/child process output, and library warnings.

**Console messages** go through the `_console()` helper, which writes
to a separate stream that stays visible to the user.  Console messages
are concise, one-line status updates:

```
[register] qwen3-8b-instruct: register submitted
[acquired] qwen3-8b-instruct: GPU 2
[waiting]  qwen3-32b-instruct: waiting for GPU ...
[register] qwen3-8b-instruct: done (wait=0.0s, register=45.2s)
[generate] qwen3-8b-instruct: generation submitted
[acquired] qwen3-8b-instruct: generate: GPU 1
[generate] qwen3-8b-instruct: done (wait=0.0s, restore=3.1s, generate=1.2s, checkpoint=2.4s) -> "Albert Einstein..."
[remove]   qwen3-8b-instruct: removal submitted
[remove]   qwen3-8b-instruct: removed
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

## Timing Breakdowns

Both `_register_sync` and `_generate_sync` report wall-clock timing
breakdowns in their console messages.

**Register**: `wait` (time spent waiting for a GPU) and `register`
(time from GPU acquisition through init, attach, stage, sleep,
checkpoint).

**Generate**: `wait` (GPU queue), `restore` (restore + wake_up_weights +
h2d + scatter + wake_up_kv_cache), `generate` (llm.generate), and
`checkpoint` (sleep + checkpoint).

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

Each `register` call gets its own pool thread for GPU work.  Each
`generate` call gets its own pool thread.  Within a model ID, generate
waits on the register future, so they are serialized.  Across model
IDs, everything runs concurrently.

```
Main thread           Pool threads              Worker processes
-----------           ------------              ----------------
register("a")         (download "a")
register("b")         (download "b")
register("c")         (download "c")
  |                       |
  |                   Thread 1: _register_sync  --> Instance A worker (gpu0)
  |                   Thread 2: _register_sync  --> Instance B worker (gpu1)
  |                   Thread 3: _register_sync  --> (waits for GPU...)
  |                       |
wait()                    |
  |                   Thread 3: _register_sync  --> Instance C worker (gpu0)
  |                       |
generate("b") -----> Thread 4: _generate_sync
                       acquire GPU 1
                       restore -> repin -> wake_up_weights -> h2d -> scatter -> wake_up_kv_cache -> generate -> wait
                       unpin -> sleep -> checkpoint -> wait
                       release GPU 1
```

## GPU Assignment

GPUs are managed via a `queue.Queue` (FIFO).  `init()` populates the
queue with all discovered GPU IDs.  `_acquire_gpu(label)` blocks on
`queue.get()` until a GPU is available.  `_release_gpu(gpu)` returns
the GPU via `queue.put()`.  Both `_register_sync` and `_generate_sync`
use the same acquire/release pattern.

With N GPUs, up to N operations proceed in parallel.  Additional
requests block on the queue until a GPU is released.

## Process Cleanup

Teardown kills the entire process tree (worker + vLLM child +
descendants) using `_kill_process_tree`, which walks descendants
bottom-up via `psutil` and sends SIGKILL.  `Instance._reset()` also
force-kills the worker if it doesn't exit within the join timeout.

## File Structure

```
semi_persistence/
  orchestrator.py          -- Orchestrator class
  orchestrator_DESIGN.md   -- This file
  instance.py              -- Instance class (see instance_DESIGN.md)
  worker.py                -- Worker process + child thread
  vllm_child.py            -- Spawned vLLM child process
  demo.py                  -- CLI demo script (stdout/stderr -> log file)
  demo.ipynb               -- Jupyter notebook demo
```
