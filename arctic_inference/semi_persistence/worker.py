"""Worker process for a single Instance.

Each worker is spawned from the main process.  On "init", it spawns the
vLLM child via mp.get_context("spawn").  Checkpoint/restore run in the
worker via CUDA driver ctypes.  CRIU save/load enables dumping the
child process tree to disk and restoring it with new PIDs.

Command protocol:  (cmd, kwargs)
Result protocol:   (cmd, elapsed, error, info)
"""
import json, os, shutil, signal, subprocess, sys, time, ctypes, threading, queue, struct

import pynvml
import torch.multiprocessing as mp

import semip_logging


# ---------------------------------------------------------------------------
# Dead-child diagnostics
# ---------------------------------------------------------------------------


def _diagnose_dead_child(child_pid):
    """Return a one-line forensic summary of a (probably) dead child.

    Distinguishes SIGKILL (signal 9 -- usually kernel OOM-killer),
    SIGSEGV (11 -- typically a CUDA / native crash), SIGABRT (6 --
    Python ``assert`` or C ``abort``) and clean exits, and reports the
    process state ("Z" zombie, "R" running, etc.) plus VmRSS at death.
    Best-effort: any failure to read ``/proc`` or reap the zombie is
    swallowed and reported as ``unknown``.
    """
    parts = []

    # Process state from /proc/<pid>/status -- captures whether the
    # child is a zombie awaiting reap, frozen, etc.  May not exist if
    # the kernel already cleaned it up by the time we read.
    state = "unknown"
    rss_kib = None
    try:
        with open(f"/proc/{child_pid}/status", "r") as _sf:
            for _line in _sf:
                if _line.startswith("State:"):
                    # Format: "State:\tZ (zombie)"
                    state = _line.split(":", 1)[1].strip()
                elif _line.startswith("VmRSS:"):
                    rss_kib = int(_line.split()[1])
    except (FileNotFoundError, PermissionError, ValueError):
        pass
    parts.append(f"state={state}")
    if rss_kib is not None:
        parts.append(f"rss={rss_kib / 1024:.1f}MiB")

    # Reap the zombie (if any) and decode the exit reason.  WNOHANG
    # so we never block when the child is somehow still running (e.g.
    # frozen via cuda-checkpoint -- in which case we won't reap and
    # just record that fact).
    exit_desc = "no_reap"
    try:
        _wpid, _wstatus = os.waitpid(child_pid, os.WNOHANG)
        if _wpid == 0:
            exit_desc = "running_or_frozen"
        elif os.WIFSIGNALED(_wstatus):
            _sig = os.WTERMSIG(_wstatus)
            try:
                _name = signal.Signals(_sig).name
            except ValueError:
                _name = f"sig{_sig}"
            _hint = ""
            if _sig == signal.SIGKILL:
                _hint = " (likely kernel OOM-killer)"
            elif _sig == signal.SIGSEGV:
                _hint = " (native/CUDA crash)"
            elif _sig == signal.SIGABRT:
                _hint = " (assert/abort)"
            exit_desc = f"killed_by={_name}({_sig}){_hint}"
        elif os.WIFEXITED(_wstatus):
            exit_desc = f"exited_code={os.WEXITSTATUS(_wstatus)}"
        else:
            exit_desc = f"raw_status=0x{_wstatus:x}"
    except ChildProcessError:
        exit_desc = "already_reaped"
    except OSError as _e:
        exit_desc = f"waitpid_err={type(_e).__name__}"
    parts.append(exit_desc)

    return ", ".join(parts)

# ---------------------------------------------------------------------------
# CUDA driver API bindings (checkpoint / restore)
# ---------------------------------------------------------------------------

_CU_CHECKPOINT_ALREADY_DONE = 401

_cu_bindings = None
_cu_bindings_pid = None
_cu_lock = threading.Lock()


def _check_cu(name, ret, *, ignore=None):
    if ret != 0 and ret != ignore:
        raise RuntimeError(f"{name} failed with CUresult={ret}")


def _get_cu():
    """Return CUDA driver bindings, lazily initializing per-process."""
    global _cu_bindings, _cu_bindings_pid
    with _cu_lock:
        if _cu_bindings_pid != os.getpid():
            lib = ctypes.CDLL("libcuda.so")

            lib.cuInit.argtypes = [ctypes.c_uint]
            lib.cuInit.restype = ctypes.c_int

            lib.cuCheckpointProcessLock.argtypes = [ctypes.c_int, ctypes.c_void_p]
            lib.cuCheckpointProcessLock.restype = ctypes.c_int

            lib.cuCheckpointProcessCheckpoint.argtypes = [ctypes.c_int, ctypes.c_void_p]
            lib.cuCheckpointProcessCheckpoint.restype = ctypes.c_int

            lib.cuCheckpointProcessUnlock.argtypes = [ctypes.c_int, ctypes.c_void_p]
            lib.cuCheckpointProcessUnlock.restype = ctypes.c_int

            lib.cuCheckpointProcessRestore.argtypes = [ctypes.c_int, ctypes.c_void_p]
            lib.cuCheckpointProcessRestore.restype = ctypes.c_int

            _check_cu("cuInit", lib.cuInit(0))
            _cu_bindings = lib
            _cu_bindings_pid = os.getpid()
    return _cu_bindings


def _get_descendant_pids(pid):
    """Return PIDs of all descendant processes, leaves first (bottom-up)."""
    import psutil
    try:
        proc = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return []
    children = proc.children(recursive=True)
    filtered = []
    for c in children:
        try:
            if "resource_tracker" in " ".join(c.cmdline()):
                continue
        except (psutil.NoSuchProcess, psutil.ZombieProcess, psutil.AccessDenied):
            pass
        filtered.append(c.pid)
    filtered.reverse()
    return filtered


def _kill_process_tree(pid):
    """SIGKILL a process and all its descendants (leaves first)."""
    import signal as _sig
    for desc_pid in _get_descendant_pids(pid):
        try:
            os.kill(desc_pid, _sig.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        os.kill(pid, _sig.SIGKILL)
    except ProcessLookupError:
        pass


_is_root = os.geteuid() == 0


def _run_cuda_checkpoint(action, pid, ignore_state_err=False, device_map=None):
    """Run sudo cuda-checkpoint --action <action> --pid <pid>."""
    cmd = ["sudo", "cuda-checkpoint", "--action", action, "--pid", str(pid)]
    if device_map:
        cmd.extend(["--device-map", device_map])
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        err_msg = (r.stderr or r.stdout or "")
        if ignore_state_err and "present state" in err_msg.lower():
            return
        raise RuntimeError(f"cuda-checkpoint {action}({pid}) failed: {err_msg}")


def _worker_checkpoint(child_pid, unlock=True):
    """Checkpoint the vLLM child and all its GPU-holding descendants.

    If unlock=False, leaves processes in CUDA 'checkpointed' state
    (for a subsequent CRIU dump — the CRIU CUDA plugin will skip the
    redundant lock+checkpoint cycle when it sees the process is already
    checkpointed).

    Uses driver API directly when running as root, falls back to the
    cuda-checkpoint CLI via sudo otherwise.
    """
    descendant_pids = _get_descendant_pids(child_pid)
    all_pids = descendant_pids + [child_pid]
    if _is_root:
        cu = _get_cu()
        for pid in all_pids:
            _check_cu(f"Lock({pid})", cu.cuCheckpointProcessLock(pid, None))
            _check_cu(f"Checkpoint({pid})", cu.cuCheckpointProcessCheckpoint(pid, None))
            if unlock:
                _check_cu(f"Unlock({pid})", cu.cuCheckpointProcessUnlock(pid, None),
                          ignore=_CU_CHECKPOINT_ALREADY_DONE)
    else:
        for pid in all_pids:
            _run_cuda_checkpoint("lock", pid)
            _run_cuda_checkpoint("checkpoint", pid)
            if unlock:
                _run_cuda_checkpoint("unlock", pid, ignore_state_err=True)
    return all_pids


def _gpu_uuids():
    """Return list of GPU UUID strings (e.g. 'GPU-<uuid>') via NVML."""
    pynvml.nvmlInit()
    uuids = []
    for i in range(pynvml.nvmlDeviceGetCount()):
        u = pynvml.nvmlDeviceGetUUID(pynvml.nvmlDeviceGetHandleByIndex(i))
        if isinstance(u, bytes):
            u = u.decode()
        uuids.append(u)
    return uuids


def _build_full_device_map(old_gpu, new_gpu):
    """Build full GPU device map string for cuda-checkpoint --device-map.

    cuda-checkpoint requires a bijective mapping of ALL visible GPUs,
    not just the pair being swapped.
    Format: oldUuid1=newUuid1,oldUuid2=newUuid2,...
    """
    uuids = _gpu_uuids()
    if not (0 <= old_gpu < len(uuids) and 0 <= new_gpu < len(uuids)):
        raise ValueError(
            f"GPU index out of range: old_gpu={old_gpu}, new_gpu={new_gpu}, "
            f"visible GPUs=[0..{len(uuids) - 1}]")
    pairs = []
    for i, uuid in enumerate(uuids):
        if i == old_gpu:
            pairs.append(f"{uuid}={uuids[new_gpu]}")
        elif i == new_gpu:
            pairs.append(f"{uuid}={uuids[old_gpu]}")
        else:
            pairs.append(f"{uuid}={uuid}")
    return ",".join(pairs)


def _build_restore_args(old_gpu, new_gpu):
    """Build a CUcheckpointRestoreArgs ctypes buffer for GPU migration.

    Layout (64-bit):
      gpuPairs*      (8 bytes pointer)
      gpuPairsCount  (4 bytes, unsigned int)
      reserved       (44 bytes, zeroed)
      reserved1      (8 bytes, zeroed)
    Each CUcheckpointGpuPair is oldUuid(16) + newUuid(16) = 32 bytes.
    """
    uuids_str = _gpu_uuids()
    if not (0 <= old_gpu < len(uuids_str) and 0 <= new_gpu < len(uuids_str)):
        raise ValueError(
            f"GPU index out of range: old_gpu={old_gpu}, new_gpu={new_gpu}, "
            f"visible GPUs=[0..{len(uuids_str) - 1}]")
    uuids = [bytes.fromhex(u.replace("GPU-", "").replace("-", ""))
             for u in uuids_str]

    pairs_data = bytearray()
    for i in range(len(uuids)):
        old_uuid = uuids[i]
        if i == old_gpu:
            new_uuid = uuids[new_gpu]
        elif i == new_gpu:
            new_uuid = uuids[old_gpu]
        else:
            new_uuid = uuids[i]
        pairs_data += old_uuid + new_uuid

    pairs_buf = (ctypes.c_char * len(pairs_data))(*pairs_data)
    pairs_ptr = ctypes.cast(pairs_buf, ctypes.c_void_p)

    args_data = bytearray(64)
    struct.pack_into("<Q", args_data, 0, pairs_ptr.value)
    struct.pack_into("<I", args_data, 8, len(uuids))
    args_buf = (ctypes.c_char * 64)(*args_data)
    return args_buf, pairs_buf


def _worker_restore(pids, old_gpu=None, new_gpu=None):
    """Restore CUDA context on pids.

    State machine: checkpointed → restore → locked → unlock → running

    Uses driver API directly when running as root, falls back to the
    cuda-checkpoint CLI via sudo otherwise.
    """
    if _is_root:
        cu = _get_cu()
        args_ptr = None
        _kept_alive = None
        if old_gpu is not None and new_gpu is not None and old_gpu != new_gpu:
            args_buf, pairs_buf = _build_restore_args(old_gpu, new_gpu)
            args_ptr = ctypes.cast(args_buf, ctypes.c_void_p)
            _kept_alive = (args_buf, pairs_buf)
        for pid in reversed(pids):
            _check_cu(f"Restore({pid})", cu.cuCheckpointProcessRestore(pid, args_ptr),
                      ignore=_CU_CHECKPOINT_ALREADY_DONE)
            _check_cu(f"Unlock({pid})", cu.cuCheckpointProcessUnlock(pid, None),
                      ignore=_CU_CHECKPOINT_ALREADY_DONE)
    else:
        device_map = None
        if old_gpu is not None and new_gpu is not None and old_gpu != new_gpu:
            device_map = _build_full_device_map(old_gpu, new_gpu)
        for pid in reversed(pids):
            _run_cuda_checkpoint("restore", pid, device_map=device_map)
            _run_cuda_checkpoint("unlock", pid, ignore_state_err=True)


# ---------------------------------------------------------------------------
# CRIU save / load (process image to/from disk)
# ---------------------------------------------------------------------------

def _resolve_fd_resource(pid, fd):
    """Read /proc/<pid>/fd/<fd> to determine the CRIU resource identifier."""
    link = os.readlink(f"/proc/{pid}/fd/{fd}")
    return link


def _worker_criu_save(child_pid, image_dir, pipe_fd, pipe_resource, rank,
                      meta_extra=None):
    """Dump the vLLM child process tree to disk via CRIU (destructive).

    The child process is killed after a successful dump.  The on-disk
    image is later restored via load().
    """
    # Clear any stale dump in the target directory so leftover files from a
    # prior criu_dump() (which CRIU may not overwrite) cannot corrupt the new
    # image.  Files from a previously aborted dump may be root-owned, so fall
    # back to `sudo rm -rf` (sudo is already required for criu dump).
    if os.path.exists(image_dir):
        try:
            shutil.rmtree(image_dir)
        except PermissionError:
            subprocess.run(["sudo", "rm", "-rf", image_dir], check=True)
    os.makedirs(image_dir, exist_ok=True)

    external_unix = []
    nvidia_fds = {}
    proc_fd_dir = f"/proc/{child_pid}/fd"
    for fd_name in os.listdir(proc_fd_dir):
        try:
            link = os.readlink(f"{proc_fd_dir}/{fd_name}")
            if link.startswith("socket:["):
                ino = link.split("[")[1].rstrip("]")
                external_unix.append(f"unix[{ino}]")
            elif "/dev/nvidia" in link:
                nvidia_fds[int(fd_name)] = link
        except OSError:
            pass

    cmd = [
        "sudo",
        "criu", "dump",
        "-t", str(child_pid),
        "-D", image_dir,
        "-o", "dump.log",
        # No --shell-job: the child detaches from the controlling terminal
        # at startup (fd 0 -> /dev/null + setsid), so no tty enters the
        # image.  A tty would be unreattachable inside the private PID
        # namespace the restore path uses.
        "--tcp-close",
        "--ext-unix-sk",
        "--link-remap",
        "--libdir", "/usr/lib/criu/empty",
        "-v4",
    ]
    for ext in external_unix:
        cmd.extend(["--external", ext])
    result = subprocess.run(cmd, capture_output=True, text=True)
    subprocess.run(["sudo", "chown", "-R", f"{os.getuid()}:{os.getgid()}", image_dir],
                   capture_output=True)
    if result.returncode != 0:
        detail = result.stderr or result.stdout or "(no output)"
        log_path = os.path.join(image_dir, "dump.log")
        if os.path.exists(log_path):
            with open(log_path) as f:
                detail += "\n--- dump.log ---\n" + f.read()[-2000:]
        raise RuntimeError(
            f"criu dump failed (rc={result.returncode}): {detail}")

    meta = {
        "child_pid": child_pid,
        "pipe_fd": pipe_fd,
        "pipe_resource": pipe_resource,
        "nvidia_fds": {str(k): v for k, v in nvidia_fds.items()},
        "rank": rank,
    }
    if meta_extra:
        meta.update(meta_extra)
    with open(os.path.join(image_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return meta


def _find_pid_by_pipe(pipe_inode):
    """Scan /proc to find the PID that holds a socket with the given inode.

    Skips the calling process's own pid: ``multiprocessing.Pipe()`` is built
    on ``socketpair()``, so both endpoints share the same socket inode.  The
    worker keeps the parent end and would otherwise be returned (it has the
    smaller pid, and ``/proc`` enumeration is pid-sorted) instead of the
    actual peer we are trying to locate.
    """
    target = f"socket:[{pipe_inode}]"
    self_pid = os.getpid()
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        if int(entry) == self_pid:
            continue
        fd_dir = f"/proc/{entry}/fd"
        try:
            for fd_name in os.listdir(fd_dir):
                try:
                    link = os.readlink(f"{fd_dir}/{fd_name}")
                    if link == target:
                        return int(entry)
                except OSError:
                    pass
        except (OSError, PermissionError):
            pass
    return None


# CRIU restores every task at its *recorded* PID (via clone3(set_tid)).
# Two images captured on the same node with their child trees alive at
# the same time carry adjacent/interleaved PIDs, and a destructive dump
# leaves the killed child as a zombie under its still-live worker, which
# keeps that PID occupied.  Either way the restore fails with
# "Can't fork for <pid>: File exists" (EEXIST) on whatever TID is already
# taken.
#
# The fix is to give each restore its own PID namespace so the recorded
# PIDs are free.  CRIU 4.2 can't ``--join-ns pid:`` an existing PID
# namespace ("join-ns pid namespace not supported"), so instead we run
# criu itself *inside* a fresh PID namespace: a privileged helper does
# ``unshare(CLONE_NEWPID)`` + ``fork()`` so its child is PID 1, that PID 1
# unshares a mount namespace and mounts a private /proc (so criu sees the
# namespace's PID view), forks criu ``restore -d``, records criu's exit
# code, then reaps forever so the namespace (and the detached restored
# tree, reparented onto it) stays alive.  Everything the rest of the
# worker touches keys off the restored root's *host* PID (found via the
# inherited pipe), which is unaffected by the nested namespace.  The
# holder tuple is ``(pid1_host_pid, popen)``; SIGKILLing PID 1 destroys
# the namespace and the whole restored tree in one shot.


def _kill_pidns_holder(holder, log=None):
    """Tear down a restore's PID-namespace holder ``(pid1_host_pid, popen)``.

    SIGKILLing PID 1 of the namespace makes the kernel kill every task in
    it (the restored tree), so this doubles as the restored-tree cleanup.
    """
    if not holder:
        return
    host_pid, proc = holder
    try:
        subprocess.run(["sudo", "kill", "-9", str(host_pid)],
                       capture_output=True)
    except Exception:
        pass
    try:
        proc.wait(timeout=5)
    except Exception:
        try:
            proc.kill()
        except OSError:
            pass
    if log is not None:
        log.info("  PID-namespace holder torn down (reaper host_pid=%s)",
                 host_pid)


def _worker_criu_load(image_dir, new_pipe_fd):
    """Restore a vLLM child process tree from a CRIU image on disk.

    Restores into a dedicated PID namespace (criu runs *inside* a fresh
    namespace held open by a PID 1 reaper) so the image's recorded PIDs
    can't collide with a concurrently-restored sibling's tree or with a
    zombie left behind by the dump.  Returns ``(host_pid, meta, holder)``
    where ``host_pid`` is the restored root's *host-visible* PID
    (discovered by scanning /proc for the inherited pipe -- the
    ``--pidfile`` reports the in-namespace PID, which is meaningless on
    the host) and ``holder`` is ``(pid1_host_pid, popen)`` to hand to
    ``_kill_pidns_holder`` at teardown.
    """
    import fcntl

    meta_path = os.path.join(image_dir, "meta.json")
    with open(meta_path) as f:
        meta = json.load(f)

    pipe_resource = meta["pipe_resource"]
    pidfile = os.path.join(image_dir, "restored.pid")
    for _stale in [pidfile, os.path.join(image_dir, "restore.log")]:
        if os.path.exists(_stale):
            os.remove(_stale)

    flags = fcntl.fcntl(new_pipe_fd, fcntl.F_GETFD)
    fcntl.fcntl(new_pipe_fd, fcntl.F_SETFD, flags & ~fcntl.FD_CLOEXEC)

    pipe_inode = os.readlink(f"/proc/self/fd/{new_pipe_fd}")
    if pipe_inode.startswith("socket:["):
        pipe_inode = pipe_inode.split("[")[1].rstrip("]")

    # sudo closes all FDs >= 3, so we pass the pipe fd to the child via
    # a Unix domain socket (SCM_RIGHTS) bound to a temp path that sudo
    # can connect to.
    import socket as _socket, tempfile, array, threading

    sock_path = os.path.join(tempfile.gettempdir(),
                             f"criu_fd_{os.getpid()}_{new_pipe_fd}.sock")
    if os.path.exists(sock_path):
        os.remove(sock_path)

    srv = _socket.socket(_socket.AF_UNIX, _socket.SOCK_STREAM)
    srv.bind(sock_path)
    os.chmod(sock_path, 0o777)
    srv.listen(1)

    def _send_fd():
        conn, _ = srv.accept()
        conn.sendmsg(
            [b"\x00"],
            [(_socket.SOL_SOCKET, _socket.SCM_RIGHTS,
              array.array("i", [new_pipe_fd]))]
        )
        conn.close()
        srv.close()

    sender = threading.Thread(target=_send_fd, daemon=True)
    sender.start()

    criu_argv = [
        "criu", "restore",
        "-D", image_dir,
        "-o", "restore.log",
        # No --shell-job: images captured by the updated dump path own
        # their session and hold no controlling terminal, so there is no
        # external tty to reattach -- which is what would otherwise break
        # restore inside the private PID namespace.
        "--tcp-close",
        "--inherit-fd", f"fd[{new_pipe_fd}]:{pipe_resource}",
        "--inherit-fd", "fd[1]:stdout",
        "--inherit-fd", "fd[2]:stderr",
        "--pidfile", pidfile,
        "--link-remap",
        "-d",
        "-v4",
    ]

    # Runtime side-channels (on shared storage, visible from inside the
    # helper's private mount namespace and from the host worker alike):
    #   reaper.pid -- PID 1's host PID, published by the host-ns parent
    #   restore.rc -- criu's exit code, published by PID 1 once criu exits
    reaper_pidfile = os.path.join(image_dir, "reaper.pid")
    rc_path = os.path.join(image_dir, "restore.rc")
    # A prior run that died abnormally (SIGKILL, crash) may have left its
    # reaper -- and thus its namespace + restored tree -- alive.  If a
    # stale reaper.pid points at a live process, kill it before reusing
    # this image so we don't accumulate orphaned namespaces.
    if os.path.exists(reaper_pidfile):
        try:
            _stale_reaper = int(open(reaper_pidfile).read().strip())
            if _stale_reaper > 1 and os.path.exists(f"/proc/{_stale_reaper}"):
                subprocess.run(["sudo", "kill", "-9", str(_stale_reaper)],
                               capture_output=True)
        except (OSError, ValueError):
            pass
    for _stale in (reaper_pidfile, rc_path):
        if os.path.exists(_stale):
            os.remove(_stale)

    # Helper (run as root via sudo).  sudo strips fds >= 3, so it first
    # re-receives the inherited pipe fd over the SCM_RIGHTS socket, then
    # unshares a PID namespace and forks: the host-ns parent publishes
    # PID 1's host PID and blocks; PID 1 unshares a mount namespace with a
    # private /proc, forks criu ``restore -d`` into the new namespace,
    # records criu's exit code, and reaps forever so the namespace (and
    # the detached restored tree) survives.
    helper_script = (
        "import os, socket, array, ctypes, signal, sys, time\n"
        "libc = ctypes.CDLL('libc.so.6', use_errno=True)\n"
        "CLONE_NEWPID = 0x20000000\n"
        "CLONE_NEWNS = 0x00020000\n"
        "MS_REC = 0x4000\n"
        "MS_PRIVATE = 0x40000\n"
        "s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)\n"
        f"s.connect({sock_path!r})\n"
        "msg, ancdata, _, _ = s.recvmsg(1, socket.CMSG_LEN(4))\n"
        "received = None\n"
        "for cl, ct, cd in ancdata:\n"
        "    if cl == socket.SOL_SOCKET and ct == socket.SCM_RIGHTS:\n"
        "        fds = array.array('i'); fds.frombytes(cd); received = fds[0]\n"
        "s.close()\n"
        f"os.dup2(received, {new_pipe_fd})\n"
        f"if received != {new_pipe_fd}: os.close(received)\n"
        "if libc.unshare(CLONE_NEWPID) != 0:\n"
        "    e = ctypes.get_errno()\n"
        "    sys.stderr.write('unshare(CLONE_NEWPID): %s\\n' % os.strerror(e))\n"
        "    os._exit(3)\n"
        "pid1 = os.fork()\n"
        "if pid1 > 0:\n"
        # host-ns parent: fd belongs to the restored tree now, drop our
        # copy so the pipe scan uniquely finds the restored root; publish
        # PID 1's host pid; block so this process (and the sudo handle)
        # lives as long as the namespace.
        f"    os.close({new_pipe_fd})\n"
        f"    open({reaper_pidfile!r}, 'w').write(str(pid1) + '\\n')\n"
        "    try:\n"
        "        os.waitpid(pid1, 0)\n"
        "    except OSError:\n"
        "        pass\n"
        "    os._exit(0)\n"
        "os.setsid()\n"
        "if libc.unshare(CLONE_NEWNS) == 0:\n"
        "    libc.mount(b'none', b'/', None, MS_REC | MS_PRIVATE, None)\n"
        "    libc.mount(b'proc', b'/proc', b'proc', 0, None)\n"
        "criu_pid = os.fork()\n"
        "if criu_pid == 0:\n"
        f"    os.execvp({criu_argv[0]!r}, {criu_argv!r})\n"
        "    os._exit(127)\n"
        f"os.close({new_pipe_fd})\n"
        "_, status = os.waitpid(criu_pid, 0)\n"
        "rc = os.waitstatus_to_exitcode(status)\n"
        "try:\n"
        f"    open({rc_path!r}, 'w').write(str(rc) + '\\n')\n"
        "except OSError:\n"
        "    pass\n"
        "signal.signal(signal.SIGTERM, lambda *a: os._exit(0))\n"
        "while True:\n"
        "    try:\n"
        "        os.waitpid(-1, 0)\n"
        "    except ChildProcessError:\n"
        "        time.sleep(0.2)\n"
        "    except OSError:\n"
        "        time.sleep(0.2)\n"
    )
    cmd = ["sudo", "python3", "-c", helper_script]

    holder = None
    # From here on, any failure must tear the namespace down (which also
    # kills anything CRIU partially restored into it) so we don't leak a
    # namespace / orphan tree per attempt.
    try:
        # stdout -> /dev/null: criu logs to restore.log (via -o) and we
        # signal completion through files, so nothing must be drained from
        # stdout; the restored process transiently inherits fd 1 there
        # (rebind_log repoints it right after) and an undrained pipe could
        # otherwise fill and block it.  Keep stderr for helper diagnostics.
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                stderr=subprocess.PIPE, text=True)

        # Wait for PID 1's host pid and criu's exit code to appear.  criu
        # restore of a large model tree can take a while; poll generously.
        deadline = time.time() + 300
        reaper_pid = None
        rc = None
        while time.time() < deadline:
            if reaper_pid is None and os.path.exists(reaper_pidfile):
                try:
                    reaper_pid = int(open(reaper_pidfile).read().strip())
                    holder = (reaper_pid, proc)
                except (OSError, ValueError):
                    reaper_pid = None
            if os.path.exists(rc_path):
                try:
                    rc = int(open(rc_path).read().strip())
                    break
                except (OSError, ValueError):
                    rc = None
            if proc.poll() is not None and reaper_pid is None:
                # Helper died before even publishing PID 1 (e.g. unshare
                # failed); surface its stderr.
                break
            time.sleep(0.1)

        sender.join(timeout=2)
        try:
            os.remove(sock_path)
        except OSError:
            pass
        subprocess.run(["sudo", "chown", "-R", f"{os.getuid()}:{os.getgid()}", image_dir],
                       capture_output=True)

        if rc is None:
            _err = ""
            try:
                if proc.poll() is not None:
                    _err = (proc.stderr.read() or "")[-500:]
            except Exception:
                pass
            raise RuntimeError(
                f"criu restore did not complete (reaper_pid={reaper_pid!r}, "
                f"helper stderr={_err!r})")
        if rc != 0:
            detail = f"rc={rc}"
            log_path = os.path.join(image_dir, "restore.log")
            if os.path.exists(log_path):
                with open(log_path) as f:
                    detail += "\n--- restore.log ---\n" + f.read()[-2000:]
            raise RuntimeError(f"criu restore failed ({detail})")

        # The restored root lives in the private namespace, so the
        # ``--pidfile`` value is its in-namespace PID -- meaningless on the
        # host.  Find its host PID by the inherited pipe instead.
        new_pid = _find_pid_by_pipe(pipe_inode)
        if new_pid is None or new_pid == os.getpid():
            raise RuntimeError(
                f"failed to discover CRIU-restored root host pid "
                f"(pipe scan returned {new_pid!r})")
    except BaseException:
        _kill_pidns_holder(holder)
        raise

    return new_pid, meta, holder


# ---------------------------------------------------------------------------
# Child thread -- communicates with the vLLM child process via pipe
# ---------------------------------------------------------------------------

def _child_thread(instance_id, rank, child_pid, pipe,
                  child_queue, result_queue, completed_counter,
                  initial_state="alive"):
    """Thread that owns the single child.  Pulls commands from child_queue,
    executes them serially, puts results on result_queue.

    Generate commands are fire-and-forget on the pipe.  The child sends
    back ("generate_done", ...) when done (new async child) or
    ("generate", ...) immediately (old CRIU-restored child with sync
    llm.generate).  The worker accepts both.
    """
    log = semip_logging.worker(instance_id, rank)

    def _emit_result(cmd, elapsed, error, info):
        result_queue.put((cmd, elapsed, error, info))
        with completed_counter.get_lock():
            completed_counter.value += 1

    _pending_generates = 0
    # Mirrors the child's `_paused` flag.  Set on a successful `pause`
    # ack, cleared on a successful `resume` ack.  While true, the child
    # has frozen its step loop so no `generate_done` will arrive: we
    # turn `_drain_pipe_generates` into a no-op so that subsequent
    # synchronous commands (`unpin`, `sleep`, `cuda_checkpoint`, ...)
    # do not deadlock waiting on completions that will not come until
    # `resume()` re-engages the engine via prefill.
    _worker_paused = False

    def _handle_pipe_result(result):
        """Handle a single result tuple from the pipe.  Returns True if
        it was a generate completion, False otherwise (caller should
        handle it)."""
        nonlocal _pending_generates
        if isinstance(result, tuple) and len(result) == 4:
            if result[0] in ("generate_done", "generate"):
                _pending_generates -= 1
                _emit_result("generate", result[1], result[2], result[3])
                return True
        return False

    def _drain_pipe_generates():
        nonlocal _pending_generates
        if _worker_paused:
            return
        while _pending_generates > 0:
            try:
                result = pipe.recv()
            except (BrokenPipeError, ConnectionResetError, EOFError):
                while _pending_generates > 0:
                    _emit_result("generate", 0.0,
                                 "child process died during generate", {})
                    _pending_generates -= 1
                break
            _handle_pipe_result(result)

    def _recv_sync():
        """Receive a synchronous command response, transparently consuming
        any generate completions that arrive first."""
        while True:
            result = pipe.recv()
            if not _handle_pipe_result(result):
                return result

    def _get_next_command():
        """Get next command, polling pipe for generate results while waiting."""
        nonlocal _pending_generates
        if _pending_generates == 0:
            return child_queue.get()
        while True:
            while _pending_generates > 0 and pipe.poll(0):
                try:
                    result = pipe.recv()
                except (BrokenPipeError, ConnectionResetError, EOFError):
                    while _pending_generates > 0:
                        _emit_result("generate", 0.0,
                                     "child process died during generate", {})
                        _pending_generates -= 1
                    break
                _handle_pipe_result(result)
            try:
                return child_queue.get(timeout=0.01)
            except queue.Empty:
                if _pending_generates == 0:
                    return child_queue.get()

    state = initial_state
    checkpointed_pids = (
        _get_descendant_pids(child_pid) + [child_pid]
        if initial_state == "checkpointed" else None
    )
    # When cuda_restore fails the child's CUDA driver state is left in
    # an inconsistent (locked / partially-restored) state.  Any
    # subsequent pipe-bound op (repin, generate, sleep, ...) would
    # deadlock the child in the driver.  Latch a "broken" flag so we
    # auto-fail those instead of forwarding them, until a later
    # cuda_restore succeeds (or the worker is torn down).
    cuda_broken = False
    cuda_broken_reason = None

    while True:
        cmd, kwargs = _get_next_command()
        log.info(">>> %s", cmd)

        if cmd == "exit":
            if state == "alive":
                _drain_pipe_generates()
                pipe.send(("exit", {}))
                while True:
                    result = pipe.recv()
                    if not _handle_pipe_result(result):
                        break
            pipe.close()
            _kill_process_tree(child_pid)
            log.info("exited")
            break

        if cmd == "cuda_checkpoint":
            _drain_pipe_generates()
            t0 = time.perf_counter()
            error = None
            info = {}
            try:
                checkpointed_pids = _worker_checkpoint(child_pid, unlock=False)
                log.info("  checkpointed pids: %s", checkpointed_pids)
                state = "checkpointed"
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
            elapsed = time.perf_counter() - t0
            log.info("<<< cuda_checkpoint %s (%.3fs)",
                     'OK' if error is None else 'FAILED', elapsed)
            _emit_result(cmd, elapsed, error, info)
            continue

        if cmd == "cuda_restore":
            t0 = time.perf_counter()
            error = None
            target_gpu = kwargs["gpu"]
            info = {}
            try:
                if state == "alive":
                    checkpointed_pids = _get_descendant_pids(child_pid) + [child_pid]
                    log.info("  process is alive, checkpointing before restore...")
                    if _is_root:
                        cu = _get_cu()
                        for _p in checkpointed_pids:
                            _check_cu(f"Lock({_p})", cu.cuCheckpointProcessLock(_p, None))
                            _check_cu(f"Checkpoint({_p})", cu.cuCheckpointProcessCheckpoint(_p, None))
                    else:
                        for _p in checkpointed_pids:
                            _run_cuda_checkpoint("lock", _p)
                            _run_cuda_checkpoint("checkpoint", _p)
                    state = "checkpointed"
                if checkpointed_pids is None:
                    raise RuntimeError("restore called but no checkpointed PIDs stored")
                _all_settled = True
                for _wp in checkpointed_pids:
                    _settled = False
                    for _wi in range(50):
                        try:
                            with open(f"/proc/{_wp}/status") as _sf:
                                _st = _sf.read()
                            if "State:\tS" in _st or "State:\tT" in _st:
                                _settled = True
                                break
                        except Exception:
                            break
                        time.sleep(0.1)
                    if not _settled:
                        _all_settled = False
                if _all_settled:
                    _worker_restore(checkpointed_pids,
                                    old_gpu=rank,
                                    new_gpu=target_gpu)
                else:
                    log.info("  process still running after CRIU restore, skipping CUDA restore")
                log.info("  restored pids: %s", checkpointed_pids)
                info["gpu"] = target_gpu
                rank = target_gpu
                log.set_gpu(target_gpu)
                checkpointed_pids = None
                state = "alive"
                cuda_broken = False
                cuda_broken_reason = None
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
                cuda_broken = True
                cuda_broken_reason = f"cuda_restore failed: {error}"
            elapsed = time.perf_counter() - t0
            log.info("<<< cuda_restore %s (%.3fs)",
                     'OK' if error is None else 'FAILED', elapsed)
            _emit_result(cmd, elapsed, error, info)
            continue

        if cmd == "criu_dump":
            _drain_pipe_generates()
            t0 = time.perf_counter()
            error = None
            info = {}
            try:
                image_dir = kwargs["filename"]
                pipe.send(("get_pipe_fd", {}))
                fd_result = _recv_sync()
                _fd_cmd, _fd_elapsed, _fd_error, fd_info = fd_result
                if _fd_error is not None:
                    raise RuntimeError(f"get_pipe_fd failed: {_fd_error}")
                child_pipe_fd = fd_info["pipe_fd"]
                log.info("  child pipe fd: %s", child_pipe_fd)

                pipe.send(("prepare_criu_dump", {"pipe_fd": child_pipe_fd}))
                prep_result = _recv_sync()
                _pr_cmd, _pr_elapsed, _pr_error, pr_info = prep_result
                if _pr_error is not None:
                    log.warning("  prepare_criu_dump failed: %s", _pr_error)
                else:
                    log.info("  prepare_criu_dump: fds=%s, unmapped=%s",
                             pr_info.get('closed_fds', []),
                             pr_info.get('unmapped', []))

                pipe_resource = _resolve_fd_resource(child_pid, child_pipe_fd)

                meta = _worker_criu_save(
                    child_pid, image_dir, child_pipe_fd, pipe_resource, rank,
                    meta_extra=kwargs.get("meta_extra"),
                )
                info["image_dir"] = image_dir
                info["meta"] = meta
                state = "saved"
                log.info("  CRIU saved to %s (child killed by dump)", image_dir)
            except Exception as e:
                import traceback; traceback.print_exc()
                error = f"{type(e).__name__}: {e}"
            elapsed = time.perf_counter() - t0
            log.info("<<< criu_dump %s (%.3fs)",
                     'OK' if error is None else 'FAILED', elapsed)
            _emit_result(cmd, elapsed, error, info)
            try:
                pipe.close()
            except OSError:
                pass
            break

        if cmd == "teardown":
            _drain_pipe_generates()
            t0 = time.perf_counter()
            error = None
            info = {}
            try:
                pipe.close()
                _kill_process_tree(child_pid)
                state = "removed"
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
            elapsed = time.perf_counter() - t0
            log.info("<<< teardown %s (%.3fs)",
                     'OK' if error is None else 'FAILED', elapsed)
            _emit_result(cmd, elapsed, error, info)
            break

        if cmd == "generate":
            if cuda_broken:
                log.warning("<<< generate FAILED (cuda_broken: %s)",
                            cuda_broken_reason)
                _emit_result(cmd, 0.0,
                             f"skipped: {cuda_broken_reason}", {})
                continue
            try:
                pipe.send((cmd, kwargs))
                _pending_generates += 1
            except (BrokenPipeError, ConnectionResetError, EOFError):
                _alive = os.path.exists(f"/proc/{child_pid}")
                _diag = _diagnose_dead_child(child_pid)
                log.error("<<< generate FAILED (child pipe broken, "
                          "pid=%s alive=%s, %s)",
                          child_pid, _alive, _diag)
                _emit_result(cmd, 0.0, "child process died", {})
            continue

        if cuda_broken:
            log.warning("<<< %s FAILED (cuda_broken: %s)",
                        cmd, cuda_broken_reason)
            _emit_result(cmd, 0.0,
                         f"skipped: {cuda_broken_reason}", {})
            continue

        try:
            # `pause` is the command that must skip the drain even
            # while `_worker_paused` is False: its whole purpose is
            # to stop the very generates we'd otherwise be waiting
            # on.  `resume` arrives while already paused (no-op
            # drain), so its skip is redundant but harmless.
            if cmd not in ("pause", "resume"):
                _drain_pipe_generates()
            pipe.send((cmd, kwargs))
            result = _recv_sync()
        except (BrokenPipeError, ConnectionResetError, EOFError) as _pipe_err:
            _alive = os.path.exists(f"/proc/{child_pid}")
            _diag = _diagnose_dead_child(child_pid)
            log.error("<<< %s FAILED (child pipe broken, "
                      "pid=%s alive=%s, %s)",
                      cmd, child_pid, _alive, _diag)
            _emit_result(cmd, 0.0, "child process died", {})
            _orphaned = 0
            while not child_queue.empty():
                try:
                    _ocmd, _okw = child_queue.get_nowait()
                    _emit_result(_ocmd, 0.0, "child process died", {})
                    _orphaned += 1
                except queue.Empty:
                    break
            if _orphaned:
                log.info("  flushed %d orphaned commands", _orphaned)
            break
        # Track child pause state so `_drain_pipe_generates` can no-op
        # for the duration of the pause/save/restore/resume cycle.
        _result_cmd, _, _result_error, _ = result
        if _result_error is None:
            if _result_cmd == "pause":
                _worker_paused = True
            elif _result_cmd == "resume":
                _worker_paused = False
        _emit_result(*result)

    log.info("child thread done")


# ---------------------------------------------------------------------------
# Worker main loop
# ---------------------------------------------------------------------------

def worker_loop(instance_id, rank, cmd_queue, result_queue, completed_counter):
    """Main loop for a per-Instance worker process."""
    semip_logging.init_process()
    log = semip_logging.worker(instance_id, rank)
    # Route everything this process emits (worker.N records, prints,
    # tracebacks) into the shared per-instance file.  The parent
    # truncated it at Instance() construction; we just append.
    semip_logging.redirect_stdio_to_instance_file(instance_id)

    child_pid = None
    child_proc = None
    child_queue = None
    child_thread_obj = None
    # Set when a restore stands up a private PID namespace for the child
    # tree; SIGKILLing its PID 1 at teardown destroys the namespace and
    # the whole restored tree in one shot.
    ns_holder = None

    # Fallback cleanup: teardown/exit clear ns_holder after killing it, so
    # this only fires when the worker dies abnormally (Ctrl-C, unhandled
    # exception).  The closure reads the *current* ns_holder at exit time.
    # SIGKILL can't run this; that case is covered by the stale-reaper
    # sweep in _worker_criu_load on the next restore of the same image.
    import atexit
    atexit.register(lambda: _kill_pidns_holder(ns_holder, log))

    log.info("started")

    while True:
        cmd, kwargs = cmd_queue.get()
        log.info(">>> %s", cmd)

        if cmd == "init":
            from vllm_child import vllm_child_loop

            vllm_config = kwargs["vllm_config"]

            pipe_parent, pipe_child = mp.Pipe()

            spawn_ctx = mp.get_context("spawn")
            child_proc = spawn_ctx.Process(
                target=vllm_child_loop,
                args=(pipe_child, instance_id, rank),
            )
            child_proc.start()
            pipe_child.close()
            child_pid = child_proc.pid

            child_queue = queue.Queue()
            child_thread_obj = threading.Thread(
                target=_child_thread,
                args=(instance_id, rank, child_pid, pipe_parent,
                      child_queue, result_queue, completed_counter),
                daemon=True,
            )
            child_thread_obj.start()

            child_queue.put((cmd, kwargs))
            continue

        if cmd == "criu_restore":
            t0 = time.perf_counter()
            error = None
            info = {}
            try:
                image_dir = kwargs["filename"]

                max_retries = 5
                for _attempt in range(max_retries):
                    pipe_parent, pipe_child = mp.Pipe()
                    new_pipe_fd = pipe_child.fileno()
                    try:
                        new_pid, meta, _holder = _worker_criu_load(
                            image_dir, new_pipe_fd)
                        _kill_pidns_holder(ns_holder, log)
                        ns_holder = _holder
                        pipe_child.close()
                        break
                    except RuntimeError as exc:
                        # Kill any orphaned process tree from the failed restore
                        _pipe_ino = os.readlink(f"/proc/self/fd/{new_pipe_fd}")
                        if _pipe_ino.startswith("socket:["):
                            _pipe_ino = _pipe_ino.split("[")[1].rstrip("]")
                        _orphan = _find_pid_by_pipe(_pipe_ino)
                        if _orphan:
                            log.warning("  killing orphan pid=%s from failed restore",
                                        _orphan)
                            _kill_process_tree(_orphan)
                        pipe_child.close()
                        pipe_parent.close()
                        if "File exists" in str(exc) and _attempt < max_retries - 1:
                            log.warning("  PID collision, retrying (%d/%d)",
                                        _attempt + 1, max_retries)
                            time.sleep(0.5)
                            continue
                        raise

                child_pid = new_pid
                child_proc = None
                old_rank = meta.get("rank", rank)

                # CRIU restored fd 1/2 onto the path that was open at dump
                # time, which encodes the instance_id the model had then --
                # not necessarily the one we have now.  Tell the child to
                # re-dup2 onto /tmp/inst{instance_id}.log before any other
                # command flows through the pipe.  Safe to do here because
                # _child_thread hasn't been started yet, so the worker
                # still owns the pipe.
                pipe_parent.send(("rebind_log",
                                  {"instance_id": instance_id}))
                ack = pipe_parent.recv()
                if (not isinstance(ack, tuple) or len(ack) != 4
                        or ack[0] != "rebind_log" or ack[2] is not None):
                    raise RuntimeError(f"rebind_log failed: {ack!r}")
                log.info("  rebound child stdio to %s",
                         ack[3].get("path"))

                child_queue = queue.Queue()
                child_thread_obj = threading.Thread(
                    target=_child_thread,
                    args=(instance_id, old_rank, child_pid, pipe_parent,
                          child_queue, result_queue, completed_counter,
                          "checkpointed"),
                    daemon=True,
                )
                child_thread_obj.start()

                info["pid"] = new_pid
                info["rank"] = old_rank
                info["image_dir"] = image_dir
                info["child_log_path"] = semip_logging.instance_log_path(
                    instance_id)
                log.info("  CRIU restored pid=%s (checkpointed, awaiting restore)",
                         new_pid)
            except Exception as e:
                import traceback; traceback.print_exc()
                error = f"{type(e).__name__}: {e}"
            elapsed = time.perf_counter() - t0
            log.info("<<< criu_restore %s (%.3fs)",
                     'OK' if error is None else 'FAILED', elapsed)
            result_queue.put(("criu_restore", elapsed, error, info))
            with completed_counter.get_lock():
                completed_counter.value += 1
            continue

        if cmd == "teardown":
            if child_queue is not None:
                child_queue.put(("teardown", {}))
            if child_thread_obj is not None:
                child_thread_obj.join(timeout=30)
            if child_proc is not None:
                child_proc.join(timeout=5)
                if child_proc.is_alive():
                    log.warning("child_proc still alive after join, force-killing")
                    _kill_process_tree(child_proc.pid)
                    child_proc.join(timeout=5)
            _kill_pidns_holder(ns_holder, log)
            ns_holder = None
            break

        if cmd == "exit":
            if child_queue is not None:
                child_queue.put(("exit", {}))
            if child_thread_obj is not None:
                child_thread_obj.join(timeout=30)
            if child_proc is not None:
                child_proc.join(timeout=5)
                if child_proc.is_alive():
                    log.warning("child_proc still alive after join, force-killing")
                    _kill_process_tree(child_proc.pid)
                    child_proc.join(timeout=5)
            _kill_pidns_holder(ns_holder, log)
            ns_holder = None
            result_queue.put(("exit", 0.0, None, {}))
            break

        if child_queue is not None:
            child_queue.put((cmd, kwargs))
        else:
            log.error("ERROR: no child for cmd=%s", cmd)
            result_queue.put((cmd, 0.0, f"no child initialized", {}))
            with completed_counter.get_lock():
                completed_counter.value += 1

    log.info("exiting")
