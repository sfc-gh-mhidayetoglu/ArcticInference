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
different weights.  In the initialization, we only read the model
architecture and weight headers from the model file, therefore the
configs can be a generic HF model.

```python
instance_1 = Instance(vllm_config_1)
instance_2 = Instance(vllm_config_2)
```

Instances 1 and 2 initialize in parallel with random weights.  The GPU
is specified at `init()` time:

```python
instance_1.init(gpu=0)
instance_2.init(gpu=1)
```

## Process Hierarchy

Each Instance owns one **worker process**, created when `init(gpu)` is
called.  The worker is **forked** from the main process (cheap, inherits
file descriptors and queues).  The worker then spawns the vLLM child via
`mp.get_context("spawn")` when it receives the `init` command.

The two-level fork-then-spawn design:

- **Worker (forked)**: fast to create, runs the command loop, executes
checkpoint/restore via `cuCheckpointProcess`* ctypes.  CUDA driver
bindings are lazily initialized per-process (tracked by PID) so they
work correctly after fork.
- **vLLM child (spawned)**: starts with a clean address space -- no
inherited CUDA contexts.  vLLM can freely use its own multiprocessing
internally.  Tensor parallelism works.
- **EngineCore (in-process)**: `VLLM_ENABLE_V1_MULTIPROCESSING=0` runs
the EngineCore inside the vLLM child process (no separate subprocess).
This avoids IPC serialization overhead during `scatter` (which would
otherwise pickle GPU tensors across processes, failing for models >4 GiB
and adding ~16s latency even for small models).

```
Main process  [mp.set_start_method("fork")]
  |
  |-- Instance 1  (handle, main-process side)
  |     |
  |     `-- Worker process 1  (forked from main)
  |           |-- [checkpoint/restore via cuCheckpointProcess ctypes]
  |           |
  |           `-- vLLM child process  (spawned via mp.get_context("spawn"))
  |                 |-- owns pinned CPU memory (allocated on attach)
  |                 |-- EngineCore (in-process, holds GPU memory)
  |                 `-- resource_tracker  (no GPU, skipped during checkpoint)
  |
  |-- Instance 2  (handle, main-process side)
  |     |
  |     `-- Worker process 2  (forked from main)
  |           `-- vLLM child process  (spawned)
  |                 |-- EngineCore (in-process)
  |                 `-- resource_tracker
  |
  `-- ...
```

**Important**: The main process must never initialize CUDA (e.g. via
`torch.cuda.*` calls) because workers are forked from it.  A
CUDA-initialized parent produces children with corrupted driver state
where `cuInit` fails with `CUresult=3`.  GPU memory queries use
`nvidia-smi` instead of `torch.cuda.mem_get_info` for this reason.

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
| `init(gpu)`             | Cold start a model with random weights on the given GPU.  Spawns the worker process and vLLM child. | Worker + Child                 |
| `wait()`                | Block the main process until all pending commands complete.  | Main process                               |
| `after(instance)`       | Sync with another instance (non-blocking from main process). | Worker (blocks until dependency satisfied) |
| `sleep()`               | `llm.sleep(level=2)` -- frees GPU memory.                    | Child                                      |
| `checkpoint()`          | Save CUDA state to CPU via `cuCheckpointProcess`*.  Instance becomes stateless (`gpu=None`). | Worker (ctypes) |
| `restore(gpu)`          | Restore checkpointed CUDA state onto the specified GPU.  `gpu` is required. | Worker (ctypes) |
| `attach()`              | Allocate pinned CPU memory and register for DMA.             | Child                                      |
| `detach()`              | Free pinned CPU memory.                                      | Child                                      |
| `stage(model_path)`     | Load safetensors into pinned CPU buffer.                     | Child                                      |
| `h2d()`                 | Async host-to-device transfer, then synchronize.             | Child                                      |
| `scatter()`             | Place weights from GPU staging buffer into model params.     | Child                                      |
| `wake_up(["weights"])`  | Re-allocate weight tensors on GPU.                           | Child                                      |
| `wake_up(["kv_cache"])` | Re-allocate KV cache on GPU.                                 | Child                                      |
| `teardown()`            | Tear down the instance, worker, and child.  Resets to created state, ready for `init(gpu)` again. | Worker + Child |
| `remove()`              | Remove from the instance registry.  Call after `teardown().wait()`. | Main process |


## `after(instance)` -- Cross-Instance Dependencies

`after(other)` is **non-blocking** from the main process.  It enqueues a
`wait_for` command on `self`'s worker and returns `self` immediately.
The **worker** (not the main process) blocks until the dependency is
satisfied before processing subsequent commands.

This allows the main process to keep issuing commands to other instances
while one instance's worker waits on a dependency.

### Implementation

Each Instance has:

- `_completed_counter`: an `mp.Value('i', 0)` shared with its worker.
The worker increments it after each command completes.
- `_total_sent`: a plain int (main-process side) tracking total commands
ever enqueued.

`after(other)` snapshots `target = other._total_sent` and enqueues
`("wait_for", {"instance_id": other.instance_id, "target": target})`
on `self`'s cmd_queue.  The worker resolves the counter from a
module-level `_counter_registry` (inherited at fork time) and polls
until `other._completed_counter.value >= target`.

Only the instance ID (an int) is sent through the queue -- not the
`mp.Value` itself, which cannot be pickled.  The `_counter_registry`
maps instance IDs to their `mp.Value` counters and is inherited by all
workers at fork time.

This works across instances on different GPUs / different worker
processes because `mp.Value` is backed by shared memory (`/dev/shm`).

## Pinned Memory

Each child allocates its own pinned CPU memory on `attach()` via
`torch.empty(total_size, dtype=torch.uint8, pin_memory=True)`.  This
uses `cudaHostAlloc` under the hood, which allocates memory that is both
page-locked and registered with the CUDA driver for DMA -- no separate
`cudaHostRegister` step needed.

On `detach()`, the pinned tensor is deleted, freeing the memory via
`cudaHostFree`.

The pinned buffer is sized to the model's total weight footprint
(computed from safetensors headers, no data I/O).

## stage / h2d / scatter Pipeline

### stage (NVMe -> pinned CPU, one thread per shard)

Each `.safetensors` shard file is loaded by its own thread into its
corresponding region of the pinned buffer using O_DIRECT via kvikio.
Since each thread writes to a disjoint region, no synchronization is
needed.

### h2d (pinned CPU -> GPU, async per shard)

Each shard's pinned region is copied to a GPU staging buffer via
`dst.copy_(src, non_blocking=True)`.  All shards are launched async,
then a single `torch.cuda.synchronize()` waits for completion.

### scatter (GPU staging -> model params)

`_scatter_into_model` reads from the GPU staging buffer and copies into
vLLM model parameter tensors.  Handles:

- q/k/v -> qkv_proj stacking
- gate/up -> gate_up_proj stacking
- FP8 weight transposition
- Per-tensor scale merging

After scatter, the staging buffer is freed via
`torch.cuda.caching_allocator_delete(ptr)` followed by
`torch.cuda.empty_cache()`.

## Command Sequences

### Cold start and checkpoint

```python
instance_1 = Instance(vllm_config_1)
instance_2 = Instance(vllm_config_2)

instance_1.init(gpu=0).attach().sleep().checkpoint()
instance_2.init(gpu=1).attach().sleep().checkpoint()
```

### Initialize a new instance after another finishes

We need to wait for instance 1's checkpoint on GPU 0 before
initializing instance 3 on the same GPU.

```python
instance_3 = Instance(vllm_config_3)
instance_3.after(instance_1).init(gpu=0).attach().sleep().checkpoint()

instance_3.wait()
instance_2.wait()
```

We do not have to wait on instance 1.  The same sequence with chaining:

```python
instance_3 = Instance(vllm_config_3).after(instance_1).init(gpu=0).attach().sleep().checkpoint()
```

### Restore (hot path)

```python
instance_1.restore(gpu=0)
instance_1.wake_up(["weights"])
instance_1.stage("/data-fast/Qwen/Qwen3-1.7B")
instance_1.h2d()
instance_1.scatter()
instance_1.wake_up(["kv_cache"])

instance_2.restore(gpu=1)
instance_2.wake_up(["weights"])
instance_2.stage("/data-fast/nvidia/Llama-3.1-70B-Instruct-FP8")
instance_2.h2d()
instance_2.scatter()
instance_2.wake_up(["kv_cache"])

instance_1.wait()
instance_2.wait()
```

Here we load the user's desired weights.  We can load any set of weights
as long as it matches the architecture.

### Swap active model on a GPU

```python
instance_1.sleep().checkpoint()

instance_3.after(instance_1)
instance_3.restore(gpu=0)
instance_3.wake_up(["weights"])
instance_3.stage("/data-fast/Qwen/Qwen3-32B")
instance_3.h2d()
instance_3.scatter()
instance_3.wake_up(["kv_cache"])
instance_3.wait()
```

### Two instances active on the same GPU

```python
vllm_config_4 = {"model": "Qwen/Qwen2.5-1.5B", "gpu_memory_utilization": 0.4}
vllm_config_5 = {"model": "Qwen/Qwen3-1.7B", "gpu_memory_utilization": 0.4}

instance_4 = Instance(vllm_config_4)
instance_5 = Instance(vllm_config_5)

instance_2.sleep().detach().checkpoint()

instance_4.after(instance_2).init(gpu=1)
instance_5.after(instance_4).init(gpu=1)
instance_4.attach()
instance_5.attach()
instance_4.sleep().checkpoint()
instance_5.sleep().checkpoint()
instance_4.wait()
instance_5.wait()
```

Reload both on the same GPU at the same time:

```python
instance_4.restore(gpu=1).wake_up(["weights"]).stage("/data-fast/Qwen/Qwen2.5-1.5B").h2d().scatter().wake_up(["kv_cache"])
instance_5.restore(gpu=1).wake_up(["weights"]).stage("/data-fast/Qwen/Qwen3-1.7B").h2d().scatter().wake_up(["kv_cache"])

instance_4.wait()
instance_5.wait()
```

If on the same GPU, `init()` does not overlap.  It is the user's
responsibility to serialize using `after`.

### Cross-GPU migration

Once checkpointed, an instance is stateless (`gpu=None`).  `restore(gpu)`
specifies which GPU to restore onto -- it can be the same or a different
GPU.

```python
instance = Instance(vllm_config)
instance.init(gpu=0).attach().sleep().checkpoint().wait()
# instance.gpu is now None

# Restore on GPU 1
instance.restore(gpu=1)
instance.wake_up(["weights"]).stage(path).h2d().scatter().wake_up(["kv_cache"])
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

inst.checkpoint()--> cmd_queue
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
instance.sleep().checkpoint().wait()
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

`restore(gpu)` supports restoring onto a different GPU using the CUDA
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

GPU memory is queried via `nvidia-smi` (not `torch.cuda.mem_get_info`)
to avoid initializing CUDA in the main process.

## Architecture

Two models share the same **architecture** if they have the same
structure but potentially different weights.  Architecture is determined
by reading `config.json` from the model path and extracting:

```
(architectures, hidden_size, num_hidden_layers, num_attention_heads,
 num_key_value_heads, intermediate_size, head_dim, vocab_size)
```

## File Structure

```
semi_persistence/
  instance.py       -- Instance class (GPU-agnostic handle, owns worker process)
  worker.py         -- Worker loop, child thread, checkpoint/restore/migrate ctypes
  vllm_child.py     -- vLLM child process (owns GPU, pinned memory, staging)
  abstract.py       -- Abstract InstanceBase interface (reference only)
  main_test.py      -- Integration test
  main_migrate.py   -- Cross-GPU migration test
  DESIGN.md         -- This file
```

## Known Issues

### cumem staging buffer free (PyTorch #145168)

When vLLM's sleep mode is enabled, the cumem pluggable allocator
intercepts all `torch.empty(..., device="cuda")` calls -- including the
GPU staging buffer allocated during `h2d`.  After scatter, the staging
buffer is freed via:

```python
torch.cuda.caching_allocator_delete(staging_ptr)
torch.cuda.empty_cache()
```
