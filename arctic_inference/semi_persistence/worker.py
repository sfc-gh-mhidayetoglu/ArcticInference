"""Worker process for a single Instance.

Each worker is forked from the main process.  On "init", it spawns the
vLLM child via mp.get_context("spawn").  Checkpoint/restore run in the
worker via CUDA driver ctypes.

Command protocol:  (cmd, kwargs)
Result protocol:   (cmd, elapsed, error, info)
"""
import json, os, sys, time, ctypes, threading, queue, struct
import glob as _glob

import torch.multiprocessing as mp

# ---------------------------------------------------------------------------
# Architecture / weight helpers
# ---------------------------------------------------------------------------

_ARCH_FIELDS = (
    "architectures", "hidden_size", "num_hidden_layers",
    "num_attention_heads", "num_key_value_heads", "intermediate_size",
    "head_dim", "vocab_size",
)


def _read_architecture(model_path):
    """Read config.json and return a hashable architecture key."""
    cfg_path = os.path.join(model_path, "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    vals = []
    for field in _ARCH_FIELDS:
        v = cfg.get(field)
        vals.append(tuple(v) if isinstance(v, list) else v)
    return tuple(vals)


def _weight_footprint(model_path):
    """Compute total weight bytes by reading safetensors headers (no data I/O)."""
    d = model_path.rstrip("/")
    files = sorted(_glob.glob(f"{d}/model-*.safetensors"))
    if not files:
        files = sorted(_glob.glob(f"{d}/model.safetensors"))
    total = 0
    for p in files:
        with open(p, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
        offsets = [info["data_offsets"][1]
                   for name, info in header.items() if name != "__metadata__"]
        if offsets:
            total += max(offsets) - min(
                info["data_offsets"][0]
                for name, info in header.items() if name != "__metadata__"
            )
    return total


def _shard_layout(model_path):
    """Read safetensors headers and return per-shard info.

    Returns list of (shard_path, data_offset, data_size, tensors) where
    tensors is [(name, start, end, dtype, shape), ...].
    """
    d = model_path.rstrip("/")
    files = sorted(_glob.glob(f"{d}/model-*.safetensors"))
    if not files:
        files = sorted(_glob.glob(f"{d}/model.safetensors"))

    shards = []
    for p in files:
        with open(p, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
        tensors = [(name, info["data_offsets"][0], info["data_offsets"][1],
                     info["dtype"], info["shape"])
                    for name, info in header.items() if name != "__metadata__"]
        if not tensors:
            continue
        lo = min(s for _, s, _, _, _ in tensors)
        hi = max(e for _, _, e, _, _ in tensors)
        shards.append((p, lo, hi - lo, tensors))
    return shards


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
    """Return CUDA driver bindings, re-initializing after fork."""
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

            lib.cuCheckpointProcessRestore.argtypes = [ctypes.c_int, ctypes.c_void_p]
            lib.cuCheckpointProcessRestore.restype = ctypes.c_int

            lib.cuCheckpointProcessUnlock.argtypes = [ctypes.c_int, ctypes.c_void_p]
            lib.cuCheckpointProcessUnlock.restype = ctypes.c_int

            lib.cuDeviceGetUuid.argtypes = [ctypes.c_void_p, ctypes.c_int]
            lib.cuDeviceGetUuid.restype = ctypes.c_int

            lib.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
            lib.cuDeviceGetCount.restype = ctypes.c_int

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
    children = [c for c in children if "resource_tracker" not in " ".join(c.cmdline())]
    children.reverse()
    return [c.pid for c in children]


class CUuuid(ctypes.Structure):
    _fields_ = [("bytes", ctypes.c_char * 16)]


class CUcheckpointGpuPair(ctypes.Structure):
    _fields_ = [
        ("oldUuid", CUuuid),
        ("newUuid", CUuuid),
    ]


_PTR_SIZE = ctypes.sizeof(ctypes.c_void_p)


class CUcheckpointRestoreArgs(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("gpuPairs", ctypes.POINTER(CUcheckpointGpuPair)),
        ("gpuPairsCount", ctypes.c_uint),
        ("reserved", ctypes.c_char * (52 - _PTR_SIZE)),
        ("reserved1", ctypes.c_uint64),
    ]


def _get_device_uuid(cu, ordinal):
    """Get the CUuuid for a GPU by device ordinal."""
    uuid = CUuuid()
    _check_cu(f"cuDeviceGetUuid({ordinal})",
              cu.cuDeviceGetUuid(ctypes.byref(uuid), ordinal))
    return uuid


def _get_device_count(cu):
    count = ctypes.c_int(0)
    _check_cu("cuDeviceGetCount", cu.cuDeviceGetCount(ctypes.byref(count)))
    return count.value


def _make_restore_args(cu, old_gpu, new_gpu):
    """Build CUcheckpointRestoreArgs with GPU pair for cross-GPU restore.

    Every GPU visible to CUDA must be listed. The mapping must be a
    valid permutation (bijective): old_gpu swaps with new_gpu, all
    others map to themselves.
    """
    dev_count = _get_device_count(cu)
    pairs = (CUcheckpointGpuPair * dev_count)()
    for i in range(dev_count):
        pairs[i].oldUuid = _get_device_uuid(cu, i)
        if i == old_gpu:
            pairs[i].newUuid = _get_device_uuid(cu, new_gpu)
        elif i == new_gpu:
            pairs[i].newUuid = _get_device_uuid(cu, old_gpu)
        else:
            pairs[i].newUuid = _get_device_uuid(cu, i)

    args = CUcheckpointRestoreArgs()
    ctypes.memset(ctypes.byref(args), 0, ctypes.sizeof(args))
    args.gpuPairsCount = dev_count
    args.gpuPairs = pairs
    return args, pairs


def _worker_checkpoint(child_pid):
    """Checkpoint the vLLM child and all its GPU-holding descendants."""
    cu = _get_cu()
    descendant_pids = _get_descendant_pids(child_pid)
    all_pids = descendant_pids + [child_pid]
    for pid in all_pids:
        _check_cu(f"Lock({pid})", cu.cuCheckpointProcessLock(pid, None))
        _check_cu(f"Checkpoint({pid})", cu.cuCheckpointProcessCheckpoint(pid, None))
        _check_cu(f"Unlock({pid})", cu.cuCheckpointProcessUnlock(pid, None),
                  ignore=_CU_CHECKPOINT_ALREADY_DONE)
    return all_pids


def _worker_restore(pids, old_gpu=None, new_gpu=None):
    """Restore processes in top-down order (reverse of checkpoint order).

    If old_gpu and new_gpu are set, remaps the checkpoint from old_gpu
    to new_gpu using CUcheckpointGpuPair UUID mapping.
    """
    cu = _get_cu()
    restore_arg = None
    _pairs_ref = None
    if old_gpu is not None and new_gpu is not None and old_gpu != new_gpu:
        restore_arg, _pairs_ref = _make_restore_args(cu, old_gpu, new_gpu)
    for pid in reversed(pids):
        if restore_arg is not None:
            cu.cuCheckpointProcessRestore.argtypes = [ctypes.c_int, ctypes.POINTER(CUcheckpointRestoreArgs)]
            _check_cu(f"Restore({pid})",
                      cu.cuCheckpointProcessRestore(pid, ctypes.byref(restore_arg)))
            cu.cuCheckpointProcessRestore.argtypes = [ctypes.c_int, ctypes.c_void_p]
        else:
            _check_cu(f"Restore({pid})",
                      cu.cuCheckpointProcessRestore(pid, None))
        _check_cu(f"Unlock({pid})", cu.cuCheckpointProcessUnlock(pid, None))


# ---------------------------------------------------------------------------
# Child thread -- communicates with the vLLM child process via pipe
# ---------------------------------------------------------------------------

def _child_thread(rank, child_pid, pipe, arch,
                  child_queue, result_queue, completed_counter):
    """Thread that owns the single child.  Pulls commands from child_queue,
    executes them serially, puts results on result_queue."""

    def _tlog(msg):
        print(f"[worker{rank}] [{time.strftime('%H:%M:%S')}] {msg}",
              flush=True)

    def _emit_result(cmd, elapsed, error, info):
        result_queue.put((cmd, elapsed, error, info))
        with completed_counter.get_lock():
            completed_counter.value += 1

    state = "alive"
    checkpointed_pids = None

    while True:
        cmd, kwargs = child_queue.get()
        _tlog(f">>> {cmd}")

        if cmd == "exit":
            if state == "alive":
                pipe.send(("exit", {}))
                pipe.recv()
            pipe.close()
            import signal as _sig
            try:
                os.kill(child_pid, _sig.SIGKILL)
            except ProcessLookupError:
                pass
            _tlog("exited")
            break

        if cmd == "checkpoint":
            t0 = time.perf_counter()
            error = None
            info = {"arch": arch}
            try:
                checkpointed_pids = _worker_checkpoint(child_pid)
                _tlog(f"  checkpointed pids: {checkpointed_pids}")
                state = "checkpointed"
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
            elapsed = time.perf_counter() - t0
            _tlog(f"<<< checkpoint {'OK' if error is None else 'FAILED'} ({elapsed:.3f}s)")
            _emit_result(cmd, elapsed, error, info)
            continue

        if cmd == "restore":
            t0 = time.perf_counter()
            error = None
            target_gpu = kwargs["gpu"]
            info = {"arch": arch}
            try:
                if checkpointed_pids is None:
                    raise RuntimeError("restore called but no checkpointed PIDs stored")
                _worker_restore(checkpointed_pids,
                                old_gpu=rank,
                                new_gpu=target_gpu)
                _tlog(f"  restored pids: {checkpointed_pids}")
                info["gpu"] = target_gpu
                rank = target_gpu
                checkpointed_pids = None
                state = "alive"
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
            elapsed = time.perf_counter() - t0
            _tlog(f"<<< restore {'OK' if error is None else 'FAILED'} ({elapsed:.3f}s)")
            _emit_result(cmd, elapsed, error, info)
            continue

        if cmd == "teardown":
            t0 = time.perf_counter()
            error = None
            info = {"arch": arch}
            try:
                if state == "alive":
                    pipe.send(("exit", {}))
                    pipe.recv()
                pipe.close()
                import signal as _sig
                os.kill(child_pid, _sig.SIGKILL)
                state = "removed"
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
            elapsed = time.perf_counter() - t0
            _tlog(f"<<< teardown {'OK' if error is None else 'FAILED'} ({elapsed:.3f}s)")
            _emit_result(cmd, elapsed, error, info)
            break

        pipe.send((cmd, kwargs))
        result = pipe.recv()
        _emit_result(*result)

    _tlog("child thread done")


# ---------------------------------------------------------------------------
# Worker main loop
# ---------------------------------------------------------------------------

def worker_loop(rank, cmd_queue, result_queue, completed_counter,
                use_odirect=True):
    """Main loop for a per-Instance worker process.

    Forked from the main process.  Spawns the vLLM child via
    mp.get_context("spawn") on "init".  Checkpoint/restore run in the
    child thread via CUDA driver ctypes.
    """
    child_pid = None
    child_proc = None
    child_queue = None
    child_thread_obj = None
    arch = None

    def _wlog(msg):
        print(f"[worker{rank}] [{time.strftime('%H:%M:%S')}] (pid={os.getpid()}) {msg}", flush=True)

    _wlog("started")

    while True:
        cmd, kwargs = cmd_queue.get()
        _wlog(f">>> {cmd}")

        if cmd == "wait_for":
            from instance import _counter_registry
            counter = _counter_registry[kwargs["instance_id"]]
            target = kwargs["target"]
            _wlog(f"wait_for target={target} (current={counter.value})")
            while counter.value < target:
                time.sleep(0.01)
            _wlog(f"wait_for satisfied ({counter.value}/{target})")
            continue

        if cmd == "init":
            from vllm_child import vllm_child_loop

            vllm_config = kwargs["vllm_config"]
            model_path = vllm_config["model"]
            arch = _read_architecture(model_path)

            pipe_parent, pipe_child = mp.Pipe()

            spawn_ctx = mp.get_context("spawn")
            child_proc = spawn_ctx.Process(
                target=vllm_child_loop,
                args=(pipe_child, rank, use_odirect, arch),
            )
            child_proc.start()
            pipe_child.close()
            child_pid = child_proc.pid

            child_queue = queue.Queue()
            child_thread_obj = threading.Thread(
                target=_child_thread,
                args=(rank, child_pid, pipe_parent, arch,
                      child_queue, result_queue, completed_counter),
                daemon=True,
            )
            child_thread_obj.start()

            child_queue.put((cmd, kwargs))
            continue

        if cmd == "teardown":
            if child_queue is not None:
                child_queue.put(("teardown", {}))
            if child_thread_obj is not None:
                child_thread_obj.join(timeout=30)
            if child_proc is not None:
                child_proc.join(timeout=30)
            break

        if cmd == "exit":
            if child_queue is not None:
                child_queue.put(("exit", {}))
            if child_thread_obj is not None:
                child_thread_obj.join(timeout=30)
            if child_proc is not None:
                child_proc.join(timeout=30)
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
