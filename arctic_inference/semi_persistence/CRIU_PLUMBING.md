# CRIU Plumbing: What Gets Cleaned Up and Why

## Overview

CRIU (Checkpoint/Restore In Userspace) dumps a process tree to disk and
restores it later, potentially on a different GPU.  A vLLM child process
is not a simple program — it has CUDA contexts, PyTorch distributed
backends, io_uring rings, POSIX semaphores, and hundreds of file
descriptors.  CRIU is strict: every FD, memory mapping, and file
reference in the image must be restorable at restore time, or it fails.

The dump uses `--leave-running` (`-R`) so the child process **stays
alive** after the dump.  This makes `save()` non-destructive: the
instance remains in "checkpointed" state and can proceed directly to
`restore(gpu)` without needing a separate `load()`.  The on-disk image
can also be used later via `load()` on a fresh instance (second run).

This document describes the complications we hit and how each is handled.

---

## The Pipeline

```
[Dump side — prepare_criu_dump in vllm_child.py + _worker_criu_save in worker.py]

  1. Destroy PyTorch process group  (NCCL, TCPStore threads)
  2. Wait for store threads to exit
  3. Redirect stdout/stderr to /dev/null
  4. Close io_uring FDs
  5. Munmap io_uring memory regions
  6. Remove /dev/shm/sem.* files (before dump, in worker)
  7. Remove /dev/shm/link_remap.* stale files (before dump, in worker)
  8. → CRIU dump with -R (--leave-running): image written, child stays alive

[After dump — first run continues]

  Instance is in "checkpointed" state → restore(gpu) → wake_up_weights → ... → wake_up_kv_cache → generate → ...

[Restore side — _worker_criu_load in worker.py, used on second run]

  1. Pass pipe FD via Unix socket (SCM_RIGHTS)
  2. → CRIU restore brings a new process from the image
  3. cuda-checkpoint restore / driver API restores GPU context
```

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
and submission queues are not serializable.  Both the FDs (`anon_inode:[io_uring]`)
and the memory mappings show up in `/proc/<pid>/maps`.

**Fix (dump side):**
- Close all FDs pointing to `io_uring` (scan `/proc/<pid>/fd/`)
- Munmap all `io_uring` regions (scan `/proc/<pid>/maps`)

---

## Complication 3: POSIX Semaphores (`/dev/shm/sem.*`)

**Problem:** Python's `multiprocessing` module creates POSIX named
semaphores in `/dev/shm/`.  These are memory-mapped into the process
(and possibly its descendants like the resource_tracker).  With
`--leave-running`, CRIU records the semaphore file via `link_remap`.
At restore time on a later run, the semaphore file no longer exists
(the original process was torn down), so CRIU fails:

```
Can't link dev/shm/link_remap.163 -> dev/shm/sem.XYZ: No such file or directory
```

**Fix (dump side):** In `_worker_criu_save`, glob and remove all
`/dev/shm/sem.*` files right before the CRIU dump.  This removes the
named file from the filesystem while the live process retains its
existing anonymous mmap.  CRIU then captures the mapping as anonymous
memory rather than a file-backed `link_remap` — no dangling references.

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

## Non-destructive Save (`--leave-running`)

The CRIU dump uses `-R` (`--leave-running`) so the child process stays
alive after the image is written to disk.  This enables two flows:

- **First run (cache miss):** `init → ... → checkpoint → save → restore → generate → teardown`
  The same process continues after save — no load needed.
- **Second run (cache hit):** `load(image) → restore → generate → teardown`
  A fresh process is created from the on-disk image.

After `save()`, the instance is in "checkpointed" state with both the
child process alive (CUDA-checkpointed) and a valid on-disk image.  The
worker and child thread keep running, ready for the next command.

---

## Image Metadata (`meta.json`)

`save()` writes a `meta.json` file alongside the CRIU image containing:

- **CRIU plumbing**: `child_pid`, `pipe_fd`, `pipe_resource`, `nvidia_fds`, `rank`
- **Instance metadata**: `vllm_config`, `weight_bytes`, `pinned_bytes`

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

## Summary Table

| Resource           | Problem at dump time               | Dump-side fix                    | Restore-side fix                |
|--------------------|-------------------------------------|----------------------------------|---------------------------------|
| NCCL/TCPStore      | Background threads, TCP sockets     | `destroy_process_group()` + poll | `--tcp-close`                   |
| io_uring           | Non-serializable kernel state       | Close FDs + munmap               | —                               |
| POSIX semaphores   | link_remap to files gone at restore | Remove `/dev/shm/sem.*` pre-dump | —                               |
| stdout/stderr      | File size changes between dump/load | Redirect to /dev/null            | `--inherit-fd fd[1]/fd[2]`      |
| Pipe FD            | sudo closes FDs >= 3                | —                                | SCM_RIGHTS via Unix socket      |
| CUDA context       | GPU state not in CRIU image         | CRIU CUDA plugin at dump         | Driver API or cuda-checkpoint   |
| Plugin directory   | `--libdir` path missing             | Create `/usr/lib/criu/empty`     | —                               |
