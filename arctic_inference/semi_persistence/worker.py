"""Worker process for a single Instance.

Each worker is spawned from the main process.  On "init", it spawns the
vLLM child via mp.get_context("spawn").  Checkpoint/restore run in the
worker via CUDA driver ctypes.  CRIU save/load enables dumping the
child process tree to disk and restoring it with new PIDs.

Command protocol:  (cmd, kwargs)
Result protocol:   (cmd, elapsed, error, info)
"""
import json, os, subprocess, sys, time, ctypes, threading, queue, struct

import pynvml
import torch.multiprocessing as mp

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
        "--shell-job",
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
    """Scan /proc to find the PID that holds a socket with the given inode."""
    target = f"socket:[{pipe_inode}]"
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
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


def _worker_criu_load(image_dir, new_pipe_fd):
    """Restore a vLLM child process tree from a CRIU image on disk.

    Discovers the host PID by scanning /proc for the process holding
    the inherited pipe fd.
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
        "--shell-job",
        "--tcp-close",
        "--inherit-fd", f"fd[{new_pipe_fd}]:{pipe_resource}",
        "--inherit-fd", "fd[1]:stdout",
        "--inherit-fd", "fd[2]:stderr",
        "--pidfile", pidfile,
        "--link-remap",
        "-d",
        "-v4",
    ]
    helper_script = (
        "import os, socket, array\n"
        f"s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)\n"
        f"s.connect({sock_path!r})\n"
        f"msg, ancdata, _, _ = s.recvmsg(1, socket.CMSG_LEN(4))\n"
        f"for cl, ct, cd in ancdata:\n"
        f"    if cl == socket.SOL_SOCKET and ct == socket.SCM_RIGHTS:\n"
        f"        fds = array.array('i'); fds.frombytes(cd)\n"
        f"        received = fds[0]\n"
        f"s.close()\n"
        f"os.dup2(received, {new_pipe_fd})\n"
        f"if received != {new_pipe_fd}: os.close(received)\n"
        f"os.execvp({criu_argv[0]!r}, {criu_argv!r})\n"
    )
    cmd = ["sudo", "python3", "-c", helper_script]
    result = subprocess.run(
        cmd, capture_output=True, text=True,
    )
    sender.join(timeout=2)
    try:
        os.remove(sock_path)
    except OSError:
        pass
    subprocess.run(["sudo", "chown", "-R", f"{os.getuid()}:{os.getgid()}", image_dir],
                   capture_output=True)
    if result.returncode != 0:
        detail = result.stderr or result.stdout or "(no output)"
        log_path = os.path.join(image_dir, "restore.log")
        if os.path.exists(log_path):
            with open(log_path) as f:
                detail += "\n--- restore.log ---\n" + f.read()[-2000:]
        raise RuntimeError(
            f"criu restore failed (rc={result.returncode}): {detail}")

    new_pid = _find_pid_by_pipe(pipe_inode)
    if new_pid is None:
        with open(pidfile) as f:
            new_pid = int(f.read().strip())

    return new_pid, meta


# ---------------------------------------------------------------------------
# Child thread -- communicates with the vLLM child process via pipe
# ---------------------------------------------------------------------------

def _child_thread(rank, child_pid, pipe,
                  child_queue, result_queue, completed_counter,
                  initial_state="alive"):
    """Thread that owns the single child.  Pulls commands from child_queue,
    executes them serially, puts results on result_queue.

    Generate commands are fire-and-forget on the pipe.  The child sends
    back ("generate_done", ...) when done (new async child) or
    ("generate", ...) immediately (old CRIU-restored child with sync
    llm.generate).  The worker accepts both.
    """

    def _tlog(msg):
        print(f"[worker{rank}] [{time.strftime('%H:%M:%S')}] {msg}",
              flush=True)

    def _emit_result(cmd, elapsed, error, info):
        result_queue.put((cmd, elapsed, error, info))
        with completed_counter.get_lock():
            completed_counter.value += 1

    _pending_generates = 0

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

    while True:
        cmd, kwargs = _get_next_command()
        _tlog(f">>> {cmd}")

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
            _tlog("exited")
            break

        if cmd == "checkpoint_cuda":
            _drain_pipe_generates()
            t0 = time.perf_counter()
            error = None
            info = {}
            try:
                checkpointed_pids = _worker_checkpoint(child_pid, unlock=False)
                _tlog(f"  checkpointed pids: {checkpointed_pids}")
                state = "checkpointed"
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
            elapsed = time.perf_counter() - t0
            _tlog(f"<<< checkpoint_cuda {'OK' if error is None else 'FAILED'} ({elapsed:.3f}s)")
            _emit_result(cmd, elapsed, error, info)
            continue

        if cmd == "restore_cuda":
            t0 = time.perf_counter()
            error = None
            target_gpu = kwargs["gpu"]
            info = {}
            try:
                if state == "alive":
                    checkpointed_pids = _get_descendant_pids(child_pid) + [child_pid]
                    _tlog(f"  process is alive, checkpointing before restore...")
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
                    _tlog(f"  process still running after CRIU restore, skipping CUDA restore")
                _tlog(f"  restored pids: {checkpointed_pids}")
                info["gpu"] = target_gpu
                rank = target_gpu
                checkpointed_pids = None
                state = "alive"
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
            elapsed = time.perf_counter() - t0
            _tlog(f"<<< restore_cuda {'OK' if error is None else 'FAILED'} ({elapsed:.3f}s)")
            _emit_result(cmd, elapsed, error, info)
            continue

        if cmd == "save_image":
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
                _tlog(f"  child pipe fd: {child_pipe_fd}")

                pipe.send(("prepare_criu_dump", {"pipe_fd": child_pipe_fd}))
                prep_result = _recv_sync()
                _pr_cmd, _pr_elapsed, _pr_error, pr_info = prep_result
                if _pr_error is not None:
                    _tlog(f"  WARNING: prepare_criu_dump failed: {_pr_error}")
                else:
                    _tlog(f"  prepare_criu_dump: fds={pr_info.get('closed_fds', [])}, "
                          f"unmapped={pr_info.get('unmapped', [])}")

                pipe_resource = _resolve_fd_resource(child_pid, child_pipe_fd)

                meta = _worker_criu_save(
                    child_pid, image_dir, child_pipe_fd, pipe_resource, rank,
                    meta_extra=kwargs.get("meta_extra"),
                )
                info["image_dir"] = image_dir
                info["meta"] = meta
                state = "saved"
                _tlog(f"  CRIU saved to {image_dir} (child killed by dump)")
            except Exception as e:
                import traceback; traceback.print_exc()
                error = f"{type(e).__name__}: {e}"
            elapsed = time.perf_counter() - t0
            _tlog(f"<<< save_image {'OK' if error is None else 'FAILED'} ({elapsed:.3f}s)")
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
            _tlog(f"<<< teardown {'OK' if error is None else 'FAILED'} ({elapsed:.3f}s)")
            _emit_result(cmd, elapsed, error, info)
            break

        if cmd == "generate":
            try:
                pipe.send((cmd, kwargs))
                _pending_generates += 1
            except (BrokenPipeError, ConnectionResetError, EOFError):
                _alive = os.path.exists(f"/proc/{child_pid}")
                _tlog(f"<<< generate FAILED (child pipe broken, pid={child_pid} alive={_alive})")
                _emit_result(cmd, 0.0, "child process died", {})
            continue

        try:
            _drain_pipe_generates()
            pipe.send((cmd, kwargs))
            result = _recv_sync()
        except (BrokenPipeError, ConnectionResetError, EOFError) as _pipe_err:
            _alive = os.path.exists(f"/proc/{child_pid}")
            _tlog(f"<<< {cmd} FAILED (child pipe broken, pid={child_pid} alive={_alive})")
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
                _tlog(f"  flushed {_orphaned} orphaned commands")
            break
        _emit_result(*result)

    _tlog("child thread done")


# ---------------------------------------------------------------------------
# Worker main loop
# ---------------------------------------------------------------------------

def worker_loop(rank, cmd_queue, result_queue, completed_counter):
    """Main loop for a per-Instance worker process."""
    child_pid = None
    child_proc = None
    child_queue = None
    child_thread_obj = None

    def _wlog(msg):
        print(f"[worker{rank}] [{time.strftime('%H:%M:%S')}] (pid={os.getpid()}) {msg}", flush=True)

    _wlog("started")

    while True:
        cmd, kwargs = cmd_queue.get()
        _wlog(f">>> {cmd}")

        if cmd == "init":
            from vllm_child import vllm_child_loop

            vllm_config = kwargs["vllm_config"]

            pipe_parent, pipe_child = mp.Pipe()

            spawn_ctx = mp.get_context("spawn")
            child_proc = spawn_ctx.Process(
                target=vllm_child_loop,
                args=(pipe_child, rank),
            )
            child_proc.start()
            pipe_child.close()
            child_pid = child_proc.pid

            child_queue = queue.Queue()
            child_thread_obj = threading.Thread(
                target=_child_thread,
                args=(rank, child_pid, pipe_parent,
                      child_queue, result_queue, completed_counter),
                daemon=True,
            )
            child_thread_obj.start()

            child_queue.put((cmd, kwargs))
            continue

        if cmd == "load_image":
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
                        new_pid, meta = _worker_criu_load(image_dir, new_pipe_fd)
                        pipe_child.close()
                        break
                    except RuntimeError as exc:
                        # Kill any orphaned process tree from the failed restore
                        _pipe_ino = os.readlink(f"/proc/self/fd/{new_pipe_fd}")
                        if _pipe_ino.startswith("socket:["):
                            _pipe_ino = _pipe_ino.split("[")[1].rstrip("]")
                        _orphan = _find_pid_by_pipe(_pipe_ino)
                        if _orphan:
                            _wlog(f"  killing orphan pid={_orphan} from failed restore")
                            _kill_process_tree(_orphan)
                        pipe_child.close()
                        pipe_parent.close()
                        if "File exists" in str(exc) and _attempt < max_retries - 1:
                            _wlog(f"  PID collision, retrying ({_attempt + 1}/{max_retries})")
                            time.sleep(0.5)
                            continue
                        raise

                child_pid = new_pid
                child_proc = None
                old_rank = meta.get("rank", rank)

                child_queue = queue.Queue()
                child_thread_obj = threading.Thread(
                    target=_child_thread,
                    args=(old_rank, child_pid, pipe_parent,
                          child_queue, result_queue, completed_counter,
                          "checkpointed"),
                    daemon=True,
                )
                child_thread_obj.start()

                info["pid"] = new_pid
                info["rank"] = old_rank
                info["image_dir"] = image_dir
                _wlog(f"  CRIU restored pid={new_pid} (checkpointed, awaiting restore)")
            except Exception as e:
                import traceback; traceback.print_exc()
                error = f"{type(e).__name__}: {e}"
            elapsed = time.perf_counter() - t0
            _wlog(f"<<< load_image {'OK' if error is None else 'FAILED'} ({elapsed:.3f}s)")
            result_queue.put(("load_image", elapsed, error, info))
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
                    _wlog("child_proc still alive after join, force-killing")
                    _kill_process_tree(child_proc.pid)
                    child_proc.join(timeout=5)
            break

        if cmd == "exit":
            if child_queue is not None:
                child_queue.put(("exit", {}))
            if child_thread_obj is not None:
                child_thread_obj.join(timeout=30)
            if child_proc is not None:
                child_proc.join(timeout=5)
                if child_proc.is_alive():
                    _wlog("child_proc still alive after join, force-killing")
                    _kill_process_tree(child_proc.pid)
                    child_proc.join(timeout=5)
            result_queue.put(("exit", 0.0, None, {}))
            break

        if child_queue is not None:
            child_queue.put((cmd, kwargs))
        else:
            _wlog(f"ERROR: no child for cmd={cmd}")
            result_queue.put((cmd, 0.0, f"no child initialized", {}))
            with completed_counter.get_lock():
                completed_counter.value += 1

    _wlog("exiting")
