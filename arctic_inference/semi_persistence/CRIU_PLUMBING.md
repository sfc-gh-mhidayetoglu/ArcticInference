# CRIU Plumbing: What Gets Cleaned Up and Why

## Overview

CRIU (Checkpoint/Restore In Userspace) dumps a process tree to disk and
restores it later, potentially on a different GPU.  A vLLM child process
is not a simple program — it has CUDA contexts, PyTorch distributed
backends, io_uring rings, POSIX semaphores, and hundreds of file
descriptors.  CRIU is strict: every FD, memory mapping, and file
reference in the image must be restorable at restore time, or it fails.

The dump is **destructive**: CRIU kills the child process after writing
the image to disk.  Every subsequent use of the model goes through
`load()` which creates a fresh process from the on-disk image.  This
avoids dangling processes from non-destructive (`-R`) dumps and
simplifies the state machine — after `save()`, the model is always
in `"saved"` state with no live process.

This document describes the complications we hit and how each is handled.

---

## Installation (Ubuntu 24.04)

CRIU is not in the default Ubuntu repos at a recent-enough version.
Install from the official CRIU PPA:

```bash
# 1. Add the CRIU PPA
sudo apt-get install -y software-properties-common
sudo add-apt-repository -y ppa:criu/ppa
sudo apt-get update

# 2. Install CRIU (brings in crit, protobuf, etc.)
sudo apt-get install -y criu

# 3. Verify
criu --version          # should print 4.x
which crit              # /usr/sbin/crit  (CRIU image tool)
```

After installing, create the empty plugin directory that the dump
command references via `--libdir`:

```bash
sudo mkdir -p /usr/lib/criu/empty
```

Without this directory the dump aborts at plugin initialization
(see Complication 7 below).

---

## The Pipeline

```
[Dump side — prepare_criu_dump in vllm_child.py + _worker_criu_save in worker.py]

  (in child)
  1. Drain in-flight engine requests
  2. Destroy PyTorch process group   (NCCL, TCPStore threads)
  3. Wait for store threads to exit  (poll /proc/<pid>/task, up to 2.5s)
  4. dup2 /dev/null over stdout/stderr (fd 1, fd 2)
  5. Walk /proc/<pid>/fd and close every FD that does not match the
     keep-list  (see "FD Policy" below); pipe_fd and fd 0/1/2 are skipped
  6. Munmap every "io_uring" region from /proc/<pid>/maps
  7. Remove /dev/shm/sem.*   (mappings stay live as anonymous memory)
  8. Audit /proc/<pid>/task for non-"python" threads (informational)

  (back in worker)
  9. Scan child fds for socket:[ino] → --external unix[ino]
 10. Scan child fds for /dev/nvidia*   → record into meta.json
 11. → criu dump (destructive): image written, child killed

[Every use after dump — load from image]

  1. Pass pipe FD via Unix socket (SCM_RIGHTS) into sudo'd helper
  2. helper dup2's the pipe fd into place, execvp's criu restore
  3. criu restore --inherit-fd fd[N]:pipe_resource
                  --inherit-fd fd[1]:stdout, fd[2]:stderr
                  --link-remap --tcp-close --shell-job
  4. cuda-checkpoint restore / driver API restores GPU context
```

---

## FD Policy at Dump Time

The child does not blanket-null its FDs.  `prepare_criu_dump`
walks `/proc/<pid>/fd/` and applies a **keep-list** based on the symlink
target; everything else is `os.close()`'d:

| FD class                | Action                                                     |
|-------------------------|------------------------------------------------------------|
| `fd 0` (stdin)          | left untouched (skipped: `<= 2`)                           |
| `fd 1`, `fd 2`          | `dup2(/dev/null, …)` first, then skipped                   |
| pipe fd to worker       | preserved (skipped explicitly via `pipe_fd` argument)      |
| `/dev/nvidia*`          | preserved — CUDA driver fds; restored by CRIU CUDA plugin  |
| `/dev/shm/*`            | preserved — POSIX shm, including pinned host buffers       |
| `anon_inode:*`          | preserved — eventfd, epoll, **and io_uring** (rings munmap'd separately) |
| `socket:[…]`            | preserved — worker tags unix sockets `--external unix[ino]`; TCP gets `--tcp-close` |
| `pipe:[…]`              | preserved                                                  |
| Everything else         | closed (regular files, Triton `.so` opens, log files, etc.) |

The keep-list lives in `prepare_criu_dump`:

```python
keep_prefixes = ("/dev/nvidia", "/dev/shm", "anon_inode:",
                 "socket:", "pipe:")
```

### What survives into the CRIU image

- The Python interpreter, the main thread, and any threads that aren't
  the NCCL/TCPStore set torn down above.
- All Python objects: the `LLM` engine, tokenizer, scheduler, KV cache
  manager, etc.
- Model weights and KV cache on the GPU (handled by the CUDA plugin at
  dump and by `cuCheckpointProcess*` / `cuda-checkpoint` at restore).
- `/dev/nvidia*` FDs (also recorded as `nvidia_fds` in `meta.json`).
- `/dev/shm` FDs and their mappings (anon pages now that `sem.*` files
  are deleted; pinned host buffers remain mapped).
- `anon_inode:` FDs (eventfd, epoll; io_uring FDs too — see Complication 2).
- `socket:` FDs marked `--external unix[<ino>]`, plus the worker pipe FD
  re-inherited via `--inherit-fd`.
- `pipe:` FDs.
- Triton JIT `.so` mappings, including ones already `(deleted)` on disk
  (see Complication 9 — these are intentionally *not* pre-handled and
  rely on `--link-remap` + the destructive dump to keep ghost-remap
  collisions rare).

---

## Complication 1: PyTorch Distributed (NCCL + TCPStore)

**Problem:** `dist.init_process_group()` spawns background threads
(`pt_tcpstore`, `pt_nccl_watchdg`, `pt_nccl_heartbt`) and opens TCP
sockets.  CRIU cannot restore TCP connections or threads that are
blocked on network I/O.

**Fix (dump side):** Call `dist.destroy_process_group()` before dump.
Then poll `/proc/<pid>/task/` until the store threads exit (up to 2.5s).
The TCPStore thread sometimes lingers — the dump proceeds with a
warning, and CRIU handles the remaining thread via `--tcp-close`.

---

## Complication 2: io_uring

**Problem:** Modern PyTorch/libtorch uses `io_uring` for async I/O.
CRIU cannot checkpoint `io_uring` instances — the kernel ring buffers
and submission queues are not serializable.  Both the FDs
(`anon_inode:[io_uring]`) and the memory mappings show up in
`/proc/<pid>/maps`.

**Fix (dump side):** Munmap every `io_uring` region found in
`/proc/<pid>/maps` (scanned line-by-line, `libc.munmap()` via ctypes).
The `anon_inode:[io_uring]` FDs themselves currently fall under the
`anon_inode:` keep-prefix and are *not* explicitly closed — in
practice this has worked because once the rings are unmapped, what
remains is a bare `anon_inode` that CRIU can dump.

> Caveat: if a future PyTorch/libtorch revision triggers a
> dump failure on `anon_inode:[io_uring]`, tighten the keep-list in
> `prepare_criu_dump` to close those FDs explicitly while keeping
> other `anon_inode:` FDs (eventfd, epoll, …).

---

## Complication 3: POSIX Semaphores (`/dev/shm/sem.*`)

**Problem:** Python's `multiprocessing` module creates POSIX named
semaphores in `/dev/shm/`.  These are memory-mapped into the process.
The semaphore file is unlinked shortly after creation (standard POSIX
pattern), so `/proc/<pid>/maps` shows the path as `(deleted)`.  CRIU
then needs to handle the deleted file — either via ghost remaps or
`link_remap`, both of which cause problems at restore time (see
Complication 9).

**Fix (dump side):** In `prepare_criu_dump` (child process), delete
all `/dev/shm/sem.*` files before the dump.  The live process retains
its existing mmap (the kernel keeps the inode alive).  CRIU then
captures the mapping as anonymous memory — no file reference, no ghost,
no `link_remap`.

**Side effect:** At process exit, Python's multiprocessing finalizers
try to `sem_unlink()` the already-removed files, producing harmless
`FileNotFoundError` warnings.  These are cosmetic only.

---

## Complication 4: stdout/stderr (`outt_test` file)

**Problem:** When running with `&>` or `| tee`, the child's fd 1/2
point to a regular file.  CRIU records the file path and size at dump
time.  Between dump and restore, more output gets written to the file,
changing its size.  CRIU's file validation rejects the restore:

```
File outt_test has bad size 17042 (expect 15267)
```

**Fix (dump side):** Redirect fd 1 and fd 2 to `/dev/null` via
`os.dup2()` before the dump.  CRIU captures `/dev/null` instead of
the log file — no size validation issues.

**Fix (restore side):** Pass `--inherit-fd fd[1]:stdout` and
`--inherit-fd fd[2]:stderr` to `criu restore`.  CRIU skips
restoring these FDs and inherits them from the restoring process.

---

## Complication 5: Pipe FD passing through `sudo`

**Problem:** The worker communicates with the child via a pipe.  CRIU
needs the pipe FD to be inherited by the restored process
(`--inherit-fd fd[N]:resource`).  But `criu restore` runs under `sudo`,
which closes all FDs >= 3.

**Original approach:** `sudo -C 1024` (raise the close-from limit).
Failed because the sudoers policy doesn't permit `-C`.

**Fix:** Use a Unix domain socket with `SCM_RIGHTS` to pass the pipe FD:
1. Worker creates a temporary Unix socket and listens
2. A background thread accepts and sends the pipe FD via `SCM_RIGHTS`
3. A Python helper script runs under `sudo`, connects to the socket,
   receives the FD, `dup2`s it into place, then `execvp`s `criu restore`

---

## Complication 6: CUDA Context (driver API vs CLI)

**Problem:** After CRIU restore, the CUDA context needs to be
re-established on the target GPU.  The `cuda-checkpoint` CLI
(`sudo cuda-checkpoint --action restore`) works but spawns a subprocess
for every lock/checkpoint/restore/unlock call (~2-5s overhead each).

**Fix:** When running as root (`euid == 0`), use `libcuda.so` driver
API directly via ctypes:
- `cuCheckpointProcessLock(pid, NULL)`
- `cuCheckpointProcessCheckpoint(pid, NULL)`
- `cuCheckpointProcessRestore(pid, args)`  (with GPU UUID mapping for migration)
- `cuCheckpointProcessUnlock(pid, NULL)`

Non-root falls back to the `sudo cuda-checkpoint` CLI.  The driver API
path is 2-8x faster per operation.

---

## Destructive Save

The CRIU dump kills the child process after writing the image.  This
simplifies the lifecycle to a single code path:

- **First run (cache miss):** `init → ... → checkpoint → save` (child dies)
  → `load → restore → generate → teardown`
- **Every subsequent run (cache hit):** `load → restore → generate → teardown`

After `save()`, the model is in `"saved"` state with no live process.
The worker exits cleanly.  This eliminates dangling processes from
non-destructive dumps and avoids the POSIX semaphore futex deadlock
that occurred when remapping deleted shared-memory files.

---

## Image Metadata (`meta.json`)

`save()` writes a `meta.json` file alongside the CRIU image containing:

- **CRIU plumbing**: `child_pid`, `pipe_fd`, `pipe_resource`, `nvidia_fds`, `rank`
- **Instance metadata**: `vllm_config`, `total_gpu_bytes`, `pinned_cpu_bytes`

On `load()`, the instance validates that the image's `vllm_config`
matches the instance's config.  A mismatch raises `RuntimeError`
immediately, before any worker is spawned or CRIU restore attempted.

---

## Complication 7: CRIU Plugin Directory

**Problem:** The dump passes `--libdir /usr/lib/criu/empty` to prevent
CRIU from loading plugins (the CUDA plugin is loaded separately by the
CRIU CUDA infrastructure).  If the directory does not exist, CRIU
aborts at plugin initialization and the dump fails.

**Fix:** Ensure `/usr/lib/criu/empty` exists (an empty directory).
The CRIU PPA package does not create it automatically.

---

## Complication 8: PID Collisions at Restore

**Problem:** CRIU restores the process tree with the same PIDs recorded
in the image.  When multiple models restore concurrently, the target PID
may already be in use by an unrelated process on the host:

```
Can't fork for 47619: File exists
```

**Fix (worker):** `worker_loop` wraps `_worker_criu_load` in a retry
loop (up to 5 attempts with 0.5s backoff).  If the error message
contains `"File exists"`, the attempt is retried with fresh pipe
descriptors.  The conflicting PID is typically short-lived, so a brief
delay is sufficient.

---

## Complication 9: Ghost Remap Race (CRIU 4.2)

**Problem:** When a memory-mapped file has been deleted from disk (the
path shows `(deleted)` in `/proc/<pid>/maps`), CRIU uses a *ghost
remap*: it embeds the file content in the image and creates a temporary
`.cr.<id>.ghost` hard link at restore time, then unlinks it afterwards.

With `--link-remap`, CRIU 4.2 has a race condition: when the restored
process has multiple threads (vLLM processes have 400+), each thread
can attempt to unlink the same `.ghost` file.  The first `unlink()`
succeeds; subsequent threads get `ENOENT`, causing CRIU to exit with
`rc=1` even though the restore actually completed:

```
Couldn't unlink remap /tmp/torchinductor_root/.../tmp12345.ghost: No such file or directory
```

Two sources of `(deleted)` mappings were identified:

1. **Triton JIT `.so` files** — Triton compiles kernel `.so` files,
   `dlopen()`s them, then `unlink()`s the file from disk.  Primarily
   affects FP8 models.
2. **POSIX semaphores** — Python's multiprocessing creates and
   immediately unlinks `/dev/shm/sem.*` files.

Tolerating `rc=1` was attempted but led to unstable processes — the
CUDA context was sometimes left in an inconsistent state, and
subsequent `cudaHostRegister` (repin) calls would crash the child.

Recreating deleted files from `/proc/map_files/` (worker side) was
also attempted, but creating a new file at the same path doesn't
clear the `(deleted)` flag on the existing mapping's inode.
Remapping via `mmap(MAP_FIXED)` inside the child corrupted semaphore
futex state, causing deadlocks.

**Fix:** The destructive dump (no `-R`) eliminates the ghost race by
a combination of two approaches:

1. **Semaphores:** Deleted in `prepare_criu_dump` before dump.  CRIU
   captures the mapping as anonymous memory — no file reference at all.
2. **Triton `.so` files:** These remain as `(deleted)` mappings, but
   with the destructive dump the number of ghost remaps is typically
   small (only triton kernels, no semaphores).  The race is still
   theoretically possible but far less likely with fewer ghosts.

If ghost remap races resurface, the remaining fix would be to also
handle triton `.so` files in `prepare_criu_dump` by `munmap` +
`mmap(MAP_FIXED)` from a recreated file (safe for `.so` mappings
which are `MAP_PRIVATE`, unlike semaphores which are `MAP_SHARED`).

---

## Summary Table

| Resource           | Problem at dump time               | Dump-side fix                      | Restore-side fix                     |
|--------------------|-------------------------------------|------------------------------------|--------------------------------------|
| NCCL/TCPStore      | Background threads, TCP sockets     | `destroy_process_group()` + poll   | `--tcp-close`                        |
| io_uring           | Non-serializable kernel state       | Munmap rings (FDs kept via keep-list) | —                                 |
| POSIX semaphores   | `(deleted)` files → ghost/link remap| Delete sem files; captured as anon | —                                    |
| stdout/stderr      | File size changes between dump/load | Redirect to /dev/null              | `--inherit-fd fd[1]/fd[2]`           |
| Pipe FD            | sudo closes FDs >= 3                | —                                  | SCM_RIGHTS via Unix socket           |
| CUDA context       | GPU state not in CRIU image         | CRIU CUDA plugin at dump           | Driver API or cuda-checkpoint        |
| Plugin directory   | `--libdir` path missing             | Create `/usr/lib/criu/empty`       | —                                    |
| PID collisions     | —                                   | —                                  | Retry loop (5 attempts, 0.5s delay)  |
| Ghost remap race   | `(deleted)` .so → ghost race        | Destructive dump + sem deletion    | `--link-remap`                       |

---

## CRIU-Related Commit History

Chronological summary of CRIU plumbing changes across the branch.

| Commit    | Date       | Summary |
|-----------|------------|---------|
| `586641e` | 2026-03-27 | **Initial semi_persistence package.** GPU sleep/wake lifecycle, `cuda-checkpoint` CLI integration. No CRIU yet. |
| `e8098d6` | 2026-03-31 | **Cross-GPU migration.** Add instance migration with `cuda-checkpoint --device-map`. |
| `b0524b7` | 2026-03-31 | **Orchestrator + multiplexing demo.** Multi-model orchestration, first end-to-end save/load with CRIU dump/restore. |
| `9a04880` | 2026-04-06 | **CRIU save/load, driver API, cross-GPU migration.** Replace `cuda-checkpoint` CLI with `libcuda.so` driver API (`cuCheckpointProcess*`). Add `--link-remap`, `--ext-unix-sk`, `--shell-job`, `--tcp-close`. Add pipe FD passing via SCM_RIGHTS. Add `/dev/shm/sem.*` cleanup pre-dump. Add `--leave-running` for non-destructive save. |
| `fc3a3db` | 2026-04-07 | **State machine ladder.** `saved ↔ checkpoint ↔ up` lifecycle; CRIU load integrated into orchestrator state transitions. |
| `ad31cca` | 2026-04-17 | **CRIU v4.2 build instructions.** Added build-from-source instructions for CRIU 4.2 with CUDA plugin support. |
| `ccaaf81` | 2026-04-20 | **CRIU restore robustness.** Init-time cleanup of stale ghost files and `/dev/shm` leftovers. PID collision retry loop (5 attempts). Orphan process killing on failed restores. Pipe error handling in `_child_thread`. Process settle wait before CUDA restore. `CUresult=401` tolerance for redundant CUDA restores. |
| `df7d82c` | 2026-04-21 | **Destructive dump + ghost remap fixes.** Switch from `--leave-running` to destructive dump (child killed after save). Delete `/dev/shm/sem.*` in `prepare_criu_dump` so CRIU captures semaphores as anonymous memory. Remove `_recover_deleted_mappings()`, `_clear_ghost_files()`, and debug instrumentation. Clean up orchestrator init. All models must be re-dumped. |
