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

## Installation

CRIU install instructions (PPA path on Ubuntu 24.04, plus a
build-from-source fallback) have moved to [`INSTALL.md`](./INSTALL.md).
The plugin directory caveat referenced from *Complication 7* below is
covered there.

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
  9. Scan child AND descendant fds for socket:[ino] → --external unix[ino]
     (TP>1 worker subprocesses hold the multiproc-executor IPC sockets)
 10. Scan child fds for /dev/nvidia*   → record into meta.json
 11. → criu dump (destructive): image written, child killed

[Every use after dump — load from image]

  1. Pass pipe FD via Unix socket (SCM_RIGHTS) into sudo'd helper
  2. helper dup2's the pipe fd into place, then unshares a fresh PID
     namespace and forks: PID 1 is a reaper that unshares a mount ns with
     a private /proc and execs criu restore *inside* the new namespace
     (see Complication 10 — this frees the image's recorded PIDs)
  3. criu restore --inherit-fd fd[N]:pipe_resource
                  --inherit-fd fd[1]:stdout, fd[2]:stderr
                  --link-remap --tcp-close        (no --shell-job)
  4. worker finds the restored root's *host* PID via the inherited pipe
  5. cuda-checkpoint restore / driver API restores GPU context
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

- **CRIU plumbing**: `child_pid`, `pipe_fd`, `pipe_resource`, `nvidia_fds`
- **Placement and hardware identity**: `rank`, `gpus` (the physical GPU
  list; single-element at TP=1), and `gpu_uuids` — the *capture* node's
  GPU UUIDs, which the restore device map needs as its "old" side so the
  image can be restored on a different node
- **Instance metadata**: `vllm_config` (which may include the reserved
  `_env` mapping of per-model env vars), `model_dir`, `total_gpu_bytes`,
  `pinned_cpu_bytes`, `n_gpus`, `max_pinned_bytes_per_worker`

On `criu_restore()`, the instance validates that the image's
`vllm_config` and `model_dir` both match.  A mismatch raises
`RuntimeError` immediately, before any worker is spawned or CRIU restore
attempted.  The `model_dir` check matters because the image bakes
absolute compile-cache paths (see
[semi-p_DESIGN.md](semi-p_DESIGN.md)).

The child's `os.environ` is itself captured inside the CRIU dump and
restored verbatim by `load()`, so env vars set during cold start
(both the hard-coded trio in `vllm_child_loop` and any
`vllm_config["_env"]` applied before `from vllm import LLM`) survive
restore without further work.  `_env` in `meta.json` is consulted only
on cold-start paths (orchestrator-side `register`) and by
client-side dedup; it is *not* re-applied on `load()`.

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

**Problem:** CRIU restores every task at its *recorded* PID (via
`clone3(set_tid)`), so the restore fails if that PID is already taken:

```
Can't fork for 47619: File exists
```

Three ways it gets taken:

1. **A zombie from the dump.** The destructive dump kills the child, but
   its still-live worker never reaps it, and a zombie holds its PID.
   This one is guaranteed, not incidental — the image's own `child_pid`
   is the PID that is occupied.
2. **Concurrent restores.** Two images captured on the same node with
   their child trees alive simultaneously carry adjacent/interleaved
   PIDs, so the second restore collides with the first.
3. **An unrelated host process** happening to hold the recorded PID.

**Partial fix (retry loop):** `worker_loop` retries `_worker_criu_load`
up to 5 times with 0.5s backoff on `"File exists"`.  This only helps for
case 3, and only when the holder is short-lived.  A zombie under a live
parent never goes away, so cases 1 and 2 defeat it entirely.

**Real fix:** restore each tree into its own PID namespace, where the
recorded PIDs are always free.  See Complication 10.

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

## Complication 10: Per-Restore PID Namespace (and the tty it forced out)

**Problem:** See Complication 8 — the recorded PIDs are frequently
occupied, and CRIU 4.2 explicitly rejects `--join-ns pid:`, so we cannot
ask it to place the tree into a pre-made namespace.

**Fix: run criu inside a fresh PID namespace held open by a reaper.**
The sudo'd restore helper (`_worker_criu_load` in `worker.py`):

1. Receives the inherited pipe fd (SCM_RIGHTS, as in Complication 5),
   then `unshare(CLONE_NEWPID)` and `fork()`.
2. The host-namespace parent publishes **PID 1's host PID** to
   `reaper.pid` (in the image dir) and blocks — this keeps the namespace,
   and the `sudo` handle, alive.
3. **PID 1** unshares a mount namespace, mounts a private `/proc` (so
   criu sees the namespace's PID view), then forks `criu restore -d`
   into the namespace, records criu's exit code to `restore.rc`, and
   reaps forever. A live PID 1 is required both for the namespace to
   persist and for `clone3(set_tid)` of any non-1 PID to be permitted.
4. After criu detaches, the restored tree reparents onto PID 1. The
   worker discovers the restored root's **host** PID via the inherited
   pipe and drives the rest of the lifecycle with it (CUDA
   checkpoint/restore, `/proc` walks, and signals all use host PIDs,
   which are unaffected by the nesting) — `--pidfile` now holds the
   meaningless in-namespace PID.

Teardown SIGKILLs PID 1, which makes the kernel tear down the whole
namespace and the restored tree in one shot. An `atexit` fallback in
`worker_loop` catches Ctrl-C and unhandled exceptions, and a stale-reaper
sweep (reads a leftover `reaper.pid` and kills it) covers `SIGKILL`
deaths that cannot run cleanup.

**The tty this forced out.** A private PID namespace is incompatible
with a `--shell-job` image. The child inherited the interactive shell's
pts on fd 0, so it was captured as a shell job tied to an external
terminal. On restore inside the new namespace the session and pgrp
collapse to `1`, and CRIU's `TIOCSPGRP` on the host terminal fails:

```
tty: Restore inherited group 1
Error (criu/tty.c:689): tty: Failed to set group 1 on 0: Inappropriate ioctl for device
Error (criu/files.c:1221): Unable to open fd=0 id=0xec
```

**Fix (dump side):** the child sheds its controlling terminal at startup
(`vllm_child_loop`): point fd 0 at `/dev/null` and `os.setsid()` so the
captured tree owns its own session and holds no terminal. With no tty in
the image, `--shell-job` is unnecessary and is dropped from both `criu
dump` and `criu restore`. This matches the CRIU maintainers' guidance for
PID-namespace restore. **Images captured before this change (with a tty /
`--shell-job`) must be re-dumped.**

---

## Complication 11: Unprivileged Dump + Restore (`SEMIP_UNPRIVILEGED`)

**Problem:** the default dump and restore paths need `CAP_SYS_ADMIN` in
two places, so they only run on privileged pods:

1. CRIU's kerndat init builds a throwaway *network* namespace to probe
   kernel features; creating a netns needs `CAP_SYS_ADMIN`. Without it
   criu aborts with "Could not initialize kernel features detection" --
   on **both** dump and restore.
2. The restore path additionally wraps criu in a private PID namespace
   (Complication 10): `unshare(CLONE_NEWPID|CLONE_NEWNS)` + a private
   `/proc` mount, which also needs `CAP_SYS_ADMIN`.

Production pods are non-privileged: they grant only
`CAP_CHECKPOINT_RESTORE + CAP_SYS_PTRACE`, never `CAP_SYS_ADMIN`.

**Fix: `SEMIP_UNPRIVILEGED=1` -- one flag for both sides.**

- **Dump** (`_worker_criu_save`): append `--unprivileged`, which makes
  CRIU skip the netns kerndat probe. The dump has no namespace machinery
  of its own, so this one flag is the only change it needs; seizing the
  target still uses `CAP_SYS_PTRACE`.
- **Restore**: take `_worker_criu_load_lowcap` instead of
  `_worker_criu_load`. It runs `criu restore -d --unprivileged`
  **directly in the host PID namespace** -- no `unshare`, no reaper, no
  private `/proc`. The recorded PID is therefore the host PID, so
  `--pidfile` is authoritative (with a pipe-scan fallback), and teardown
  SIGKILLs the restored tree directly (holder
  `{"kind": "tree", "pid": host_pid}`) rather than collapsing a namespace.

Why the asymmetry (a flag on dump, a whole function on restore): the dump
had only dependency (1); the restore had (1) **and** (2), and (2) lives
in the sudo'd wrapper *around* criu, so it cannot be undone by a criu
flag -- it needs a different, wrapper-free restore procedure.

Because `-d` makes criu exit with the restore rc, the sudo process's exit
code *is* criu's rc, so this path needs none of the `reaper.pid` /
`restore.rc` side-channel files the namespace path uses. Its stdout and
stderr must go to `/dev/null` rather than pipes: with `-d` the restored
process inherits fd 1/2 and holds them open after criu exits, so a pipe
would never EOF and draining it would deadlock.

**Capability floor:** `CAP_CHECKPOINT_RESTORE + CAP_SYS_PTRACE`, no
`CAP_SYS_ADMIN`. `clone3(set_tid)` at the recorded PIDs in the host PID
namespace is authorized by `CAP_CHECKPOINT_RESTORE`, so it does not need
to write the read-only `ns_last_pid` sysctl. Validate a node with
`sudo criu check --unprivileged` (a residual read-only `ns_last_pid`
complaint from the checker is expected and does not block restore).

**Trade-off (accepted):** without the private PID namespace the recorded
PIDs must be free on the host -- at most **one live restore per node**.
Fine for a one-job-per-pod layout. A collision surfaces as CRIU
"File exists" in `restore.log`, which the existing retry loop recognizes.
Note this makes `scripts/test_weights.py`, which restores two models
concurrently, incompatible with this mode.

**Restorable image requires shed capabilities.** CRIU's `restore_creds()`
calls `capset()` to reinstate each task's recorded caps; if the image
recorded caps the low-cap node can't grant, restore fails at
`criu/pie/restorer.c` ("Unable to restore capabilities"). So in
unprivileged mode the vLLM child zeroes its caps *before* `import torch`
-- torch and vLLM spawn many background threads and CRIU records
credentials **per thread**, so dropping first is what makes every task
record an empty cap set. This is **implied by `SEMIP_UNPRIVILEGED=1`**,
not a separate flag: the worker sets an internal
`_SEMIP_CHILD_DROP_CAPS` signal only across the child's spawn, because
the worker itself must keep its caps to run `sudo criu`.

Consequently **image portability is decided at dump time**: an image
dumped without the flag records real caps and cannot be restored on a
low-cap node without re-dumping.

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
| stdin / tty        | pts captured as `--shell-job`; can't reattach in a PID ns | fd 0 → `/dev/null` + `setsid()` at child start | drop `--shell-job` |
| PID collisions     | Zombie from the destructive dump holds the recorded PID | — | Restore each tree in its own PID namespace (reaper + private /proc); retry loop as backstop |
| Privileged-only CRIU | dump + restore need `CAP_SYS_ADMIN` (netns kerndat probe; restore PID namespace) | `--unprivileged` (`SEMIP_UNPRIVILEGED=1`) skips the netns probe; caps shed in the child before `import torch` | lowcap path: `criu restore -d --unprivileged` in the host PID ns (no `unshare`), tree-kill teardown |
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
