# Async Generate -- Implementation Details

## Architecture Overview

Four processes/threads form the pipeline, connected by three
queue/pipe boundaries:

```
Orchestrator          Instance            Worker _child_thread     vLLM Child
(ThreadPoolExecutor)  (main process)      (worker process)         (child process)
      |                    |                     |                       |
      |               cmd_queue (mp)        child_queue (thd)       pipe (mp)
      |              ──────────────►       ──────────────►       ──────────────►
      |              result_queue (mp)     (emit_result)         pipe (mp)
      |              ◄──────────────       ◄──────────────       ◄──────────────
```

## What Is Async

All four layers support fully concurrent generates for the same model:

- **Child engine loop**: drives `LLMEngine` via `add_request()` +
  `step()` instead of blocking `LLM.generate()`.  When a generate
  command arrives, the child drains ALL pending generate commands
  from the pipe before the first `step()`, so they all get
  `add_request`'d to the engine and decode together in every GPU
  forward pass.  New generates arriving mid-decode are picked up
  via `pipe_conn.poll(0)` between steps and added to the batch.
- **Worker fire-and-forget**: sends generate to child pipe without
  waiting for a response; `_get_next_command()` polls both the pipe
  (for `generate_done` results) and the child_queue (for new
  commands) concurrently.
- **Orchestrator per-request Events**: each `_generate_sync` thread
  submits its request and waits only on its own `threading.Event`,
  not on `inst.wait()`.  A per-model waiter thread reads from
  `inst._result_queue` and resolves Events as results arrive.
  Multiple generates for the same model run truly concurrently --
  the second does not wait for the first to finish.
- **Orchestrator future separation**: generate futures are stored
  separately from move/register futures.  Generates chain only
  behind moves (to ensure the model is up), not behind other
  generates.

### Concurrency model

```
orch.generate(model, promptA, 5000)  -- submits immediately
orch.generate(model, promptB, 1000)  -- submits immediately, does NOT wait for A

Child engine batches A and B:
  step()  →  decodes 1 token for A AND 1 token for B
  step()  →  decodes 1 token for A AND 1 token for B
  ...
  B finishes at ~2s  →  generate_done(B) sent
  A finishes at ~10s →  generate_done(A) sent

Total wall time ≈ 10s (max), not 12s (sum)
```

### Serialization points (by design)

- **Move-up under gen_lock**: when a model is not yet `"running"`,
  the first `_generate_sync` thread holds `gen_lock` during the
  move-up (load, restore, wake_up).  Other generate threads block
  here until the model is ready.  Once `"running"`, the lock
  acquisition is instant (state check + skip).
- **Move/remove waits for generates**: `move()` and `remove()` chain
  behind `_last_generate_future[model_id]` so they wait for all
  in-flight generates to complete before stepping the model down.

## Backward Compatibility with Old CRIU Images

CRIU restores a process from the exact binary/bytecode that was
checkpointed.  Images dumped with the old code run `llm.generate()`
(synchronous) and respond with `("generate", elapsed, error, info)`.
Images dumped with the new code run the engine loop and respond with
`("generate_done", elapsed, error, info)`.

The worker's `_handle_pipe_result()` accepts **both** response types:

```python
if result[0] in ("generate_done", "generate"):
    _pending_generates -= 1
    _emit_result("generate", result[1], result[2], result[3])
```

No re-dump is needed for correctness, but **re-dumping is required to
get the async engine benefit** (concurrent decoding in the child).
Old images serialize generates inside the child via blocking
`llm.generate()`.

## Waiting and Polling Points

### vLLM Child (vllm_child.py)

The child main loop has **two modes**: idle and active.

| Location | Mechanism | Blocks? | Why |
|----------|-----------|---------|-----|
| Main loop, idle path | `pipe_conn.recv()` | **Yes** | No engine work, no CPU to spend. Wakes when worker sends a command. |
| Main loop, active path | `pipe_conn.poll(0)` | **No** | Engine is stepping; check if a new command arrived without blocking. |
| `engine.step()` | GPU kernel execution | **Yes** -- GPU-bound | Decodes one batch of tokens. This is the productive work. |
| Generate pipe drain | `pipe_conn.poll(0)` in loop | **No** | After receiving a generate, drains all additional generates from the pipe before the first `step()`. |

When active, the child alternates:
```
engine.step()  →  poll(0)  →  [if data: recv+handle]  →  engine.step()  → ...
```
When idle (no engine requests), it blocks on `recv()` with zero CPU.

### Worker _child_thread (worker.py)

| Location | Mechanism | Blocks? | Why |
|----------|-----------|---------|-----|
| `_get_next_command()`, no pending | `child_queue.get()` | **Yes** | No async work, wait for next command. |
| `_get_next_command()`, pending | `pipe.poll(0)` + `child_queue.get(timeout=0.01)` | **Polls** -- 10ms | Must monitor both pipe (for generate_done) and child_queue (for new commands). |
| `_drain_pipe_generates()` | `pipe.recv()` | **Yes** | Called before CRIU ops. Blocks until all in-flight generates complete. |
| `_recv_sync()` | `pipe.recv()` | **Yes** | Waiting for a synchronous command response. Transparently consumes any generate completions that arrive first. |

The 10ms poll loop only runs when `_pending_generates > 0`.

### Instance (instance.py)

| Location | Mechanism | Blocks? | Why |
|----------|-----------|---------|-----|
| `_send()` | `cmd_queue.put()` | **No** | All primitives are fire-and-forget. |
| `wait()` | `result_queue.get()` | **Yes** | Drains results until `_pending_count` reaches 0. |
| `_sync_state()` | `result_queue.get_nowait()` | **No** / **Skipped** | Non-blocking drain for UI. Skipped when `_external_waiter` is True (orchestrator waiter owns the queue). |

### Orchestrator (orchestrator.py)

| Location | Mechanism | Blocks? | Why |
|----------|-----------|---------|-----|
| `_generate_sync` Phase 1 | `gen_lock` + `_move_sync` | **Yes** (first thread) | Ensures model is running. Others skip if already running. |
| `_generate_sync` Phase 2 | `gen_lock` + `inst.generate()` | **Short lock** | Drains queue, submits to engine, starts waiter. |
| `_generate_sync` Phase 3 | `event.wait()` | **Yes** | Blocks only until THIS request completes. Other requests resolve independently. |
| `_start_generate_waiter` | `result_queue.get(timeout=0.5)` | **Yes** | Background thread, sole reader of result_queue while generates are active. |

## Data Flow for Concurrent Generate

```
Time →

Orchestrator         Instance            Worker               Child
    |                    |                   |                    |
    |  generate(A)       |                   |                    |
    |  generate(B)       |                   |                    |
    |──────────────► cmd_queue              |                    |
    |                    | ──────────► child_queue               |
    |                    |                   |──► pipe("generate",A)
    |                    |                   |──► pipe("generate",B)
    |                    |                   |    (fire-and-forget)
    |                    |                   |                    |
    |                    |                   |     recv A
    |                    |                   |     drain pipe → recv B
    |                    |                   |     engine.add_request(A)
    |                    |                   |     engine.add_request(B)
    |                    |                   |                    |
    |                    |                   |     engine.step() ── batched (A+B)
    |                    |                   |     engine.step() ── batched (A+B)
    |  event_A.wait()    |                   |     ...             |
    |  event_B.wait()    |                   |                    |
    |                    |                   |     B finishes:    |
    |                    |                   | ◄── pipe("generate_done",B)
    |                    |   _get_next_cmd   |                    |
    |                    |   polls pipe ──────┘                    |
    |                    | ◄── result_queue("generate",B)         |
    |                    |                   |                    |
    |  waiter:           |                   |     engine.step()   |
    |  event_B.set()     |                   |     A finishes:    |
    |                    |                   | ◄── pipe("generate_done",A)
    |                    | ◄── result_queue("generate",A)         |
    |  waiter:           |                   |                    |
    |  event_A.set()     |                   |                    |
    |                    |                   |                    |
    |  both futures      |                   |                    |
    |  resolve           |                   |                    |
```

Generates arriving later (while the engine is active) are picked up
via `poll(0)` between steps and added to the running batch
immediately.

## IPC Protocol

### Worker → Child (pipe)

| Message | Response | Behavior |
|---------|----------|----------|
| `("generate", {"req_id", "prompts", "sampling_params"})` | None (fire-and-forget) | New child: adds to engine, no immediate reply. Old child: responds synchronously with `("generate", ...)`. |
| `("sleep", {})` | `("sleep", elapsed, error, info)` | Synchronous. Child drains engine first. |
| `("exit", {})` | `"exit_ack"` | Synchronous. Child drains engine first. |
| All other commands | `(cmd, elapsed, error, info)` | Synchronous request-response. |

### Child → Worker (pipe, unsolicited)

| Message | When |
|---------|------|
| `("generate_done", elapsed, error, {"req_id", "outputs", ...})` | After all prompts in a req_id finish in the engine |

### Worker → Instance (result_queue)

All results use the same 4-tuple `("generate", elapsed, error, info)`.
The worker maps both `"generate_done"` (new child) and `"generate"`
(old child) to `"generate"` on the result_queue.

## CRIU Safety (Drain Points)

Before any CRIU-sensitive operation, all in-flight generates must
complete.  Drain happens at **both** layers:

| Layer | Operation | Drain mechanism |
|-------|-----------|-----------------|
| **Child** | `sleep` | `_drain_engine()` before `llm.sleep()` |
| **Child** | `prepare_criu_dump` | `_drain_engine()` before FD cleanup |
| **Child** | `exit` | `_drain_engine()` before `exit_ack` |
| **Worker** | `checkpoint` | `_drain_pipe_generates()` before `cuCheckpointProcess` |
| **Worker** | `save` | `_drain_pipe_generates()` before CRIU dump |
| **Worker** | `teardown` | `_drain_pipe_generates()` before kill |
| **Worker** | `exit` | `_drain_pipe_generates()` before sending exit to child |
| **Worker** | Any sync cmd | `_drain_pipe_generates()` before `pipe.send(cmd)` |

## Orchestrator Concurrency Details

### Per-model state

- `_generate_locks[model_id]`: serializes Phase 1 (move-up) and
  Phase 2 (submit).  Short hold when model is already running.
- `_inflight[model_id]`: list of `(req_id, req_record, Event)`
  for in-flight requests.  Modified under `gen_lock`.
- `_waiter_active[model_id]`: whether the waiter thread is running.
- `_last_generate_future[model_id]`: last generate Future, used by
  `move()` / `remove()` to chain behind.
- `inst._external_waiter`: flag set by the waiter to prevent
  `_sync_state()` (dashboard) from stealing results from the queue.

### Waiter thread lifecycle

Started by the first `_generate_sync` that submits requests.  Reads
from `inst._result_queue`, matches `req_id` to `_inflight` entries,
updates `req_record`, sets the Event.  Exits when `_inflight` is
empty and `inst._pending_count == 0`, transitions state to `"up"`,
clears `_external_waiter`, notifies GPU pool waiters.

## Result Attribution

Each `inst.generate()` assigns a unique `req_id` (e.g.
`"inst0-3"`) and stores it in `inst.last_req_id`.  The orchestrator
captures this at submission time.

Results are stored in `inst.generate_results[req_id]` (keyed dict
with `outputs`, `prompt_tokens`, `completion_tokens`).  The
orchestrator's waiter matches results by `req_id` and sets each
request's Event independently.
