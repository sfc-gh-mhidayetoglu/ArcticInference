# Semi-Persistent Instances -- Design

Semi-persistence allows vLLM instances to checkpoint and restore without
needing a cold start each time.  Checkpointing saves an instance's whole
CUDA context to CPU, leaving 0 GB trace on GPU.  Restoring wakes up the
model and loads the weights.

TTFT is directly impacted by the restore time.  A cold-start
initialization of a vLLM instance typically takes a minute.  Restore
takes a second or few.  We need to maintain hundreds of checkpointed
instances.

## Design Objective

Support cases such as below.  In the first case, multiple instances can
be checkpointed on a GPU yet only one instance can be alive.  This is
the case with large models.

```
   GPU 0     GPU 1
| inst 1  | inst 2* |
| inst 3* |         |
```

In the second case, multiple instances with small models (inst 4 and 5)
can be active on a GPU.

```
   GPU 0     GPU 1
| inst 1  | inst 2  |
| inst 3* | inst 4* |
|         | inst 5* |
```

There are many cases like these.

## Instance-Based Design

An instance represents a vLLM engine with a specific vLLM configuration.

```python
vllm_config_1 = {"model": "Qwen/Qwen3-1.7B"}
vllm_config_2 = {"model": "nvidia/Llama-3.1-70B-Instruct-FP8"}
vllm_config_3 = {"model": "Qwen/Qwen3-32B"}
```

A model is composed of an a) architecture, and b) weights.  For example,
Qwen3-32B and Qwen3-32B-Instruct have the same model architecture yet
different weights.

```python
instance_1 = Instance(vllm_config_1)
instance_2 = Instance(vllm_config_2)
```

Instances 1 and 2 initialize in parallel with real weights
(`load_format=auto`).  The GPU is specified at `init()` time:

```python
instance_1.init(gpu=0)
instance_2.init(gpu=1)
```

## Process Hierarchy

Each Instance owns one **worker process**, created when `init(gpu)` or
`load_image(filename)` is called.  Both the worker and the vLLM child are **spawned** via
`mp.get_context("spawn")`.  Spawning is safe to call from any thread
(e.g. from a `ThreadPoolExecutor` in the orchestrator), unlike fork
which can deadlock on glibc mutexes held by other threads.

- **Worker (spawned)**: runs the command loop, executes
checkpoint/restore via `cuCheckpointProcess`* ctypes.  Queues and
shared counters are passed as constructor args.
- **vLLM child (spawned)**: starts with a clean address space -- no
inherited CUDA contexts.  vLLM can freely use its own multiprocessing
internally.  Tensor parallelism works.
- **EngineCore (in-process)**: `VLLM_ENABLE_V1_MULTIPROCESSING=0` runs
the EngineCore inside the vLLM child process (no separate subprocess).
This avoids IPC serialization overhead during `load_weights` (which
would otherwise pickle GPU tensors across processes, failing for
models >4 GiB and adding ~16s latency even for small models).

```
Main process
  |
  |-- Instance 1  (handle, main-process side)
  |     |
  |     `-- Worker process 1  (spawned)
  |           |-- [checkpoint/restore via cuCheckpointProcess ctypes]
  |           |
  |           `-- vLLM child process  (spawned)
  |                 |-- owns CPU buffer (allocated on attach, pinned via repin)
  |                 |-- EngineCore (in-process, holds GPU memory)
  |                 `-- resource_tracker  (no GPU, skipped during checkpoint)
  |
  |-- Instance 2  (handle, main-process side)
  |     |
  |     `-- Worker process 2  (spawned)
  |           `-- vLLM child process  (spawned)
  |                 |-- EngineCore (in-process)
  |                 `-- resource_tracker
  |
  `-- ...
```

**Important**: GPU memory queries use NVML (`pynvml`) instead of
`torch.cuda.mem_get_info` to avoid initializing CUDA in the main
process.

## Primitives

Primitives are the minimum number of scheduling operations for
describing complicated schemes.  In a compositional design, the
primitives must be:

1. **Non-blocking** (except `wait`) so that we can manage multiple
  instances at the same time.
2. **Optimized** so that their combination will also be optimized.
3. **Chainable** -- all primitives return `self`.


| Primitive               | What it does                                                 | Executed in                                |
| ----------------------- | ------------------------------------------------------------ | ------------------------------------------ |
| `init(gpu)`             | Cold start a model with real weights on the given GPU.  Spawns the worker process and vLLM child. | Worker + Child                 |
| `wait()`                | Block the main process until all pending commands complete.  | Main process                               |
| `sleep()`               | `llm.sleep(level=2)` -- frees GPU memory.                    | Child                                      |
| `checkpoint_cuda()`          | Save CUDA state to CPU via `cuCheckpointProcess`*.  Instance becomes stateless (`gpu=None`). | Worker (ctypes) |
| `save_image(filename)`        | CRIU-dump the child process tree to disk (destructive).  The child is killed after the image is written.  Writes `meta.json` with `vllm_config` and CRIU metadata. | Worker (child thread) |
| `load_image(filename)`        | Restore a process from a CRIU image on disk.  Validates that the image's `vllm_config` matches this instance.  Spawns a new worker and CRIU-restores the child. | Worker |
| `restore_cuda(gpu)`          | Restore checkpointed CUDA state onto the specified GPU.  `gpu` is required. | Worker (ctypes) |
| `attach()`              | Allocate unpinned CPU memory sized to `model.named_parameters()`. | Child                                  |
| `detach()`              | Free CPU memory buffer.                                      | Child                                      |
| `repin()`               | `cudaHostRegister` the buffer for DMA transfers.             | Child                                      |
| `unpin()`               | `cudaHostUnregister` the buffer (data stays, CUDA registration removed). | Child                           |
| `stage()`               | Snapshot model params (GPU -> pinned CPU) in vLLM's internal format. | Child                                |
| `plan_load_weights()`   | Self-compute `max_buffer_bytes = min(pinned_cpu_bytes, allotment - pinned_cpu_bytes)` from instance state and walk `index` once to build a chunk plan (each chunk packs whole params under the budget).  Cache the plan in the child for the next `load_weights()`. | Instance + Child |
| `load_weights()`        | Pure execution against the cached chunk plan: per chunk, copy a slice of the pinned buffer to a single reused GPU staging buffer, then scatter into `model.named_parameters()` in place.  Falls back to a single chunk if no plan was cached.  Frees the staging buffer before returning. | Child |
| `wake_up_weights()`     | Re-allocate weight tensors on GPU.                           | Child                                      |
| `wake_up_kv_cache()`    | Re-allocate KV cache on GPU.                                 | Child                                      |
| `generate(prompts, sp)` | Submit inference to the engine.  Assigns a unique `req_id`; result stored in `generate_results[req_id]` and `last_generate_result`. | Child (async engine loop) |
| `teardown()`            | Tear down the instance, worker, and child.  Resets to created state, ready for `init(gpu)` again. | Worker + Child |
| `remove()`              | Remove from the instance registry.  Call after `teardown().wait()`. | Main process |


## CPU Buffer and Pin Management

`attach()` allocates a regular (unpinned) CPU buffer via
`torch.empty(total_size, dtype=torch.uint8)`.  The buffer is sized to
the total bytes of all `model.named_parameters()`, computed via
`apply_model` after init.  An `index` dict maps each parameter name to
its `(offset, nbytes, dtype, shape)` in the buffer.

Pinning is a separate step: `repin()` calls `cudaHostRegister` (via
ctypes on `libcudart.so`) to register the buffer for DMA transfers.
`unpin()` calls `cudaHostUnregister` to remove the registration while
keeping the memory allocated and data intact.

`detach()` frees the CPU buffer entirely.

### Why separate attach / repin / unpin

`cuCheckpointProcessRestore` must re-map all CUDA-registered (pinned)
host memory pages, which dominates restore latency (5-7s for large
models).  By unpinning before checkpoint and re-pinning after restore,
restore only needs to reconstruct the GPU context (~1s) and the pin
cost is paid separately via `cudaHostRegister` (~1-3s depending on
buffer size).  Net savings: ~1.5-2s per restore cycle for large models.

Using `cudaHostRegister` / `cudaHostUnregister` (instead of
`cudaHostAlloc` / `cudaHostFree`) is required because `cudaHostAlloc`
memory cannot be selectively unregistered -- `cudaHostUnregister` only
works on memory registered via `cudaHostRegister`.

### Standard sequences

- **Registration**: `attach() -> repin() -> stage() -> unpin() -> sleep() -> checkpoint_cuda()`
- **Save to disk**: `... -> checkpoint_cuda() -> save_image(filename)`
- **Load from disk**: `load_image(filename) -> plan_load_weights() -> restore_cuda(gpu) -> ...`
- **Generate restore**: `restore_cuda(gpu) -> wake_up_weights() -> repin() -> load_weights() -> wake_up_kv_cache() -> ...`
- **Generate checkpoint**: `... -> unpin() -> sleep() -> checkpoint_cuda()`

`plan_load_weights()` is chained right after `load_image(filename)` because that
is when the instance has hydrated `total_gpu_bytes` and `pinned_cpu_bytes`
from `meta.json`.  The plan caches in the worker, survives `up <-> sleep`
cycles, and is rebuilt on each fresh `load_image(filename)`.  Cold start does not
need it (cold start never calls `load_weights()`), and in-memory
checkpoint+restore paths that skip `save_image`/`load_image` rely on the single-chunk
fallback inside `load_weights()`.

## stage / plan_load_weights / load_weights Pipeline

`stage()` (host capture) and `load_weights()` (device populate) are an
inverse pair around the pinned CPU buffer.  Between `load_image(filename)` and
the first `load_weights()`, the instance calls `plan_load_weights()` to
build and cache a chunk plan in the worker.  `load_weights()` then
executes the cached plan as pure I/O.

### stage (GPU model params -> pinned CPU)

`stage()` uses `apply_model` to iterate `model.named_parameters()` and
copy each parameter's `.data` (contiguous, viewed as uint8) into the
pinned buffer at the offset recorded in `index`.  This captures weights
in vLLM's post-processed internal format (e.g. Marlin-packed for GPTQ,
cutlass layout for FP8, plain tensors for BF16).

### plan_load_weights (cache the chunk plan)

`plan_load_weights()` computes `max_buffer_bytes` from instance state
on the parent side:

```
allotment        = total_gpu_bytes * gpu_memory_utilization
max_buffer_bytes = min(pinned_cpu_bytes, allotment - pinned_cpu_bytes)
```

`total_gpu_bytes` is an NVML `.total` snapshot taken once in
`Instance.init(gpu)` (safe under the orchestrator contract that init
always takes a full L1 slot).  `pinned_cpu_bytes` is set by `attach()`
in the cold-start path or read from `meta.json` in the restore path.
When weights crowd the model's allotment, the budget shrinks; when
they fit comfortably (`pinned_cpu_bytes <= allotment / 2`) the budget
equals `pinned_cpu_bytes` and only one chunk is needed.

The worker then walks `index` in offset order and packs whole
parameters into chunks of `<= max_buffer_bytes` (no intra-parameter
splits).  If a single parameter exceeds the budget, the planner raises
`param X exceeds chunk_size`.  The plan is cached as
`(chunk_lo, chunk_hi, members)` triples plus `chunk_size`, and lives
on the worker until `detach()` resets it.

### load_weights (cached plan -> GPU staging -> model params)

For each chunk in the cached plan:

1. **Pinned CPU -> GPU staging buffer.**  Allocates one
   `buf_gpu = torch.empty(chunk_size, dtype=torch.uint8, device="cuda:0")`
   before the loop.  Per chunk, `buf_gpu[:n].copy_(pinned_buf[lo:hi],
   non_blocking=True)` followed by `torch.cuda.synchronize()`.
2. **GPU staging buffer -> model params (in place).**  `apply_model`
   scatters this chunk's members; each src view is
   `buf_gpu[(off - lo):(off - lo) + nbytes].view(dtype).reshape(shape)`,
   copied into the corresponding parameter via `param.data.copy_(src)`.
   No `model.load_weights()` or `process_weights_after_loading()` is
   needed because the staged data is already in vLLM's internal format.

After the loop, `buf_gpu` is freed via `buf_gpu.storage().resize_(0)`
followed by `torch.cuda.empty_cache()`.  This releases memory through
PyTorch's normal caching allocator path, keeping allocator metadata
consistent for CRIU checkpoint/restore (see *Known Issues* below).

When `chunk_plan` is `None` (paths that never called
`plan_load_weights`, such as in-memory checkpoint+restore tests that
skip `save_image`/`load_image`), the handler falls back to a single-chunk plan
covering the entire `index`, which is byte-identical to the
pre-chunking behavior.

### Staging buffer budget

The budget formula is the single source of truth and lives in
`Instance.plan_load_weights`.  No safety margin, no minimum-budget
floor: pathological cases self-surface in the chunk planner with a
precise `param X exceeds chunk_size` message rather than via a
separate threshold.

`total_gpu_bytes` (NVML `.total` at `init`) and `pinned_cpu_bytes` are
written into `meta.json` at `save_image` time in the order
`{vllm_config, total_gpu_bytes, pinned_cpu_bytes}`.  Old images that
predate `total_gpu_bytes` are still loadable: `Instance.load_image` falls
back to the legacy `pinned_bytes` key for `pinned_cpu_bytes`, and a
missing `total_gpu_bytes` causes `plan_load_weights` to send
`max_buffer_bytes=None`, which yields the single-chunk fallback in
`load_weights`.

KV cache is not yet allocated when `load_weights` runs (it is
`wake_up_kv_cache`'s job afterwards), so the `allotment - pinned`
slack is fully usable for the staging buffer.  No NVML calls happen
at runtime past the per-instance `init` snapshot.

## Command Sequences

### Cold start and checkpoint

```python
instance_1 = Instance(vllm_config_1)
instance_2 = Instance(vllm_config_2)

instance_1.init(gpu=0).attach().repin().stage().unpin().sleep().checkpoint_cuda()
instance_2.init(gpu=1).attach().repin().stage().unpin().sleep().checkpoint_cuda()
```

`init()` loads real weights via `load_format=auto`, so vLLM runs
`process_weights_after_loading()` and the model is ready to use.
`attach()` allocates an unpinned CPU buffer sized to the model's
parameters.  `repin()` registers it with CUDA for DMA.  `stage()`
snapshots the post-processed GPU parameters into the buffer.  `unpin()`
removes the CUDA registration so that `checkpoint_cuda()` is fast (the CUDA
driver does not need to re-map pinned pages on restore).  The buffer
data survives `unpin()`, `sleep()`, and `checkpoint_cuda()` since it is CPU
memory.

### Save to disk

After checkpoint, `save_image()` writes a CRIU image to disk.  The dump is
destructive — the child process is killed after the image is written.
The worker exits and the instance returns to a clean state:

```python
inst = Instance(vllm_config)
inst.init(gpu=0).attach().repin().stage().unpin().sleep().checkpoint_cuda()
inst.save_image("/data-fast/image-cache/my_model").wait()
# child is dead, worker exits
```

### Load from disk

Every use after save goes through `load_image()`, which restores a fresh
process from the on-disk image.  The instance's `vllm_config` must
match the saved image's config (validated automatically):

```python
inst = Instance(vllm_config)
inst.load_image("/data-fast/image-cache/my_model").plan_load_weights().wait()

inst.restore_cuda(gpu=0).wake_up_weights().repin().load_weights().wake_up_kv_cache()
inst.generate(prompts, sampling_params).wait()
```

`plan_load_weights()` is chained right after `load_image()` because that is
when the instance has hydrated `total_gpu_bytes` and `pinned_cpu_bytes`
from `meta.json`.

### Initialize a new instance after another finishes

We need to wait for instance 1's checkpoint on GPU 0 before
initializing instance 3 on the same GPU.

```python
instance_1.wait()

instance_3 = Instance(vllm_config_3)
instance_3.init(gpu=0).attach().repin().stage().unpin().sleep().checkpoint_cuda()

instance_3.wait()
instance_2.wait()
```

### Restore and generate (hot path)

After cold-start, weights are already staged in pinned CPU memory
(unpinned from CUDA).  Restore re-pins and moves them CPU→GPU:

```python
instance_1.restore_cuda(gpu=0)
instance_1.repin()
instance_1.wake_up_weights()
instance_1.load_weights()
instance_1.wake_up_kv_cache()
instance_1.generate(prompts, sampling_params)
instance_1.wait()
result = instance_1.last_generate_result
```

To re-checkpoint after generate:

```python
instance_1.unpin().sleep().checkpoint_cuda().wait()
```

### Swap active model on a GPU

```python
instance_1.unpin().sleep().checkpoint_cuda().wait()

instance_3.restore_cuda(gpu=0).repin()
instance_3.wake_up_weights()
instance_3.load_weights()
instance_3.wake_up_kv_cache()
instance_3.wait()
```

### Two instances active on the same GPU

```python
vllm_config_4 = {"model": "Qwen/Qwen2.5-1.5B", "gpu_memory_utilization": 0.4}
vllm_config_5 = {"model": "Qwen/Qwen3-1.7B", "gpu_memory_utilization": 0.4}

instance_4 = Instance(vllm_config_4)
instance_5 = Instance(vllm_config_5)

instance_2.sleep().detach().checkpoint_cuda().wait()

instance_4.init(gpu=1)
instance_4.wait()
instance_5.init(gpu=1)
instance_4.attach().repin().stage()
instance_5.attach().repin().stage()
instance_4.unpin().sleep().checkpoint_cuda()
instance_5.unpin().sleep().checkpoint_cuda()
instance_4.wait()
instance_5.wait()
```

Reload both on the same GPU at the same time:

```python
# In-memory checkpoint+restore (no save/load), so plan_load_weights is
# not chained: load_weights falls back to a single-chunk plan.
instance_4.restore_cuda(gpu=1).repin().wake_up_weights().load_weights().wake_up_kv_cache()
instance_5.restore_cuda(gpu=1).repin().wake_up_weights().load_weights().wake_up_kv_cache()

instance_4.wait()
instance_5.wait()
```

If on the same GPU, `init()` does not overlap.  It is the caller's
responsibility to serialize (e.g. by calling `wait()` between inits).

### Cross-GPU migration

Once checkpointed, an instance is stateless (`gpu=None`).  `restore_cuda(gpu)`
specifies which GPU to restore onto -- it can be the same or a different
GPU.

```python
instance = Instance(vllm_config)
instance.init(gpu=0).attach().repin().stage().unpin().sleep().checkpoint_cuda().wait()
# instance.gpu is now None

# Restore on GPU 1
instance.restore_cuda(gpu=1).repin()
instance.wake_up_weights().load_weights().wake_up_kv_cache()
instance.wait()
# instance.gpu is now 1
```

## Concurrency Model

All primitives are **non-blocking** from the main process -- they
enqueue a command and return immediately.  Each Instance has its own
worker process, so **commands for different instances run concurrently**.
Commands for the same instance are serialized by its worker.

```
Main process          Worker              vLLM child
-----------          ------              ----------
inst.sleep() -----> cmd_queue
                     pipe.send("sleep") -----------> pipe.recv
                     pipe.recv("done")  <----------- pipe.send("done")
                     completed_counter += 1
                     result_queue.put()

inst.checkpoint_cuda()--> cmd_queue
                     _worker_checkpoint(pid)
                       enumerate descendants via psutil
                       checkpoint EngineCore (leaf first)
                       checkpoint vLLM child (parent last)
                       store checkpointed PID list
                     completed_counter += 1
                     result_queue.put()

inst.wait() <------- result_queue.get()
```

### `wait()` semantics

`instance.wait()` blocks the main process and drains results for that
instance from its result_queue.  You can batch many commands and call
`wait()` once:

```python
instance.unpin().sleep().checkpoint_cuda().wait()
```

## Checkpoint / Restore Ordering

Checkpoint and restore walk the full process tree of the vLLM child
using `psutil`.  The `resource_tracker` process is skipped (no GPU).

With `VLLM_ENABLE_V1_MULTIPROCESSING=0`, the EngineCore runs in-process
so there is typically only the vLLM child process to checkpoint.  The
process tree walk still handles any descendants that may hold GPU state.

The list of checkpointed PIDs is stored so that restore uses the exact
same set.

### Cross-GPU Restore

`restore_cuda(gpu)` supports restoring onto a different GPU using the CUDA
driver's `CUcheckpointRestoreArgs` with `CUcheckpointGpuPair` UUID
mapping (requires driver 580+).  The GPU pair mapping must be a valid
permutation: old_gpu swaps with new_gpu, all others map to themselves.

## Instance Status

`Instance.print_status()` shows the state of all instances grouped by
GPU, including per-GPU memory usage.  It uses a class-level
`WeakValueDictionary` registry to auto-discover all instances.

Before printing, it calls `_sync_state()` on each instance, which
non-blockingly drains completed results from the worker's result queue
(using `_completed_counter` to know how many are available).  This
updates the instance's local state without requiring `wait()`.

`_sync_state()` is skipped when `inst._external_waiter` is True.  The
orchestrator's generate waiter thread sets this flag while it owns the
result queue, preventing the dashboard from stealing generate results.

GPU memory is queried via NVML (`pynvml`, not
`torch.cuda.mem_get_info`) to avoid initializing CUDA in the main
process.

## Architecture

Two models share the same **architecture** if they have the same
structure but potentially different weights.  Architecture is determined
by reading `config.json` from the model path and extracting:

```
(architectures, hidden_size, num_hidden_layers, num_attention_heads,
 num_key_value_heads, intermediate_size, head_dim, vocab_size)
```

## CRIU Installation (v4.2, from source)

Ubuntu archive mirrors may be unreachable on some machines.  The
reliable path is to build CRIU and its dependencies from source using
GitHub (which is always reachable).

### Dependencies (built from source)

```bash
mkdir -p /tmp/criu-build && cd /tmp/criu-build

# libcap
git clone --depth 1 --branch libcap-2.73 https://git.kernel.org/pub/scm/libs/libcap/libcap.git
cd libcap && make -j$(nproc) && sudo make install prefix=/usr && cd ..

# protobuf-c (requires protoc + libprotobuf-dev already installed)
#   If /usr/include/google/protobuf/compiler/ is missing, copy headers
#   from the protobuf source matching your installed protoc version:
#     git clone --depth 1 --branch v3.21.12 https://github.com/protocolbuffers/protobuf.git protobuf-src
#     sudo cp -r protobuf-src/src/google/protobuf/compiler /usr/include/google/protobuf/compiler
#   If libprotoc.so symlink is missing:
#     sudo ln -sf /usr/lib/x86_64-linux-gnu/libprotoc.so.32 /usr/lib/x86_64-linux-gnu/libprotoc.so
git clone --depth 1 --branch v1.5.0 https://github.com/protobuf-c/protobuf-c.git
cd protobuf-c && ./autogen.sh && ./configure --prefix=/usr && make -j$(nproc) && sudo make install && cd ..

# libnet
git clone --depth 1 --branch v1.3 https://github.com/libnet/libnet.git
cd libnet && ./autogen.sh && ./configure --prefix=/usr && make -j$(nproc) && sudo make install && cd ..

# uuid.h (if /usr/include/uuid/uuid.h is missing)
git clone --depth 1 --branch v2.40.4 https://github.com/util-linux/util-linux.git
sudo mkdir -p /usr/include/uuid
sudo cp util-linux/libuuid/src/uuid.h /usr/include/uuid/uuid.h
sudo ln -sf /usr/lib/x86_64-linux-gnu/libuuid.so.1 /usr/lib/x86_64-linux-gnu/libuuid.so
```

### CRIU 4.2

```bash
cd /tmp/criu-build
git clone --depth 1 --branch v4.2 https://github.com/checkpoint-restore/criu.git
cd criu
PKG_CONFIG_PATH="/usr/lib64/pkgconfig:/usr/lib/pkgconfig:$PKG_CONFIG_PATH" make -j$(nproc)
sudo PIP_BREAK_SYSTEM_PACKAGES=1 make install-criu PREFIX=/usr
sudo PIP_BREAK_SYSTEM_PACKAGES=1 make install-lib PREFIX=/usr
sudo PIP_BREAK_SYSTEM_PACKAGES=1 make install-crit PREFIX=/usr

# Empty plugin directory (required by --libdir during dump)
sudo mkdir -p /usr/lib/criu/empty
```

### Verify

```
criu --version          # Version: 4.2
which crit              # /usr/local/bin/crit
ls -d /usr/lib/criu/empty
```

## File Structure

```
semi_persistence/
  instance.py        -- Instance class (GPU-agnostic handle, owns worker process)
  worker.py          -- Worker loop, child thread, checkpoint/restore/CRIU save/load
  vllm_child.py      -- vLLM child process (owns GPU, pinned memory, async engine loop)
  orchestrator.py    -- Orchestrator class (see orchestrator_DESIGN.md)
  gpu_pool.py        -- GPU pool with semaphore-based acquisition
  state_server.py    -- HTTP /state endpoint for dashboard/monitor
  dashboard.py       -- Curses-based live dashboard (GPU/CPU tiers, requests)
  monitor.py         -- Plotext-based live monitor (scatter + utilization charts)
  replay.py          -- Curses replay viewer (dashboard + charts side by side)
  compare.py         -- Side-by-side comparison of two recordings
  abstract.py        -- Abstract InstanceBase interface (reference only)
  demo.py            -- CLI demo script (stdout/stderr redirected to log file)
  demo.ipynb         -- Jupyter notebook demo
  main_test.py       -- Integration test
  test_migrate.py    -- Cross-GPU migration test
  test_image.py      -- CRIU save/load image cache test
  instance_DESIGN.md -- This file
  async_generate_DETAILS.md -- Async generate implementation details
  CRIU_PLUMBING.md   -- CRIU dump/restore complications and fixes
```

## Known Issues

### Staging buffer must be freed via `storage().resize_(0)`

When vLLM's sleep mode is enabled, the cumem pluggable allocator
intercepts all `torch.empty(..., device="cuda")` calls -- including the
GPU staging buffer allocated inside `load_weights`.  The buffer **must**
be freed via `buf_gpu.storage().resize_(0)`, not
`caching_allocator_delete(ptr)`.

`caching_allocator_delete` tells PyTorch the memory was freed
externally, which corrupts the caching allocator's internal block
tracking.  When CRIU restores the process, the allocator uses stale
metadata and subsequent GPU allocations crash with
`c10::Error: invalid device pointer`.

`storage().resize_(0)` releases memory through the normal allocator
`free` path, keeping block metadata consistent across
checkpoint/restore cycles.

### Cold start is slower than dummy-weight init

`init()` uses `load_format=auto` to load real weights from disk and
run `process_weights_after_loading()`.  This is slower than the
previous `load_format=dummy` approach, but it only happens once per
instance.  The benefit is that `stage()` captures weights in vLLM's
internal format, so restore works correctly for all model types
including quantized models (GPTQ, FP8).
