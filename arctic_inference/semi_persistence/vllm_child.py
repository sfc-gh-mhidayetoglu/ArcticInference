"""vLLM child process loop.

Spawned (not forked) by the worker process.  Owns CUDA and vLLM.
Reads (cmd, kwargs) from a pipe, puts results on result_queue.

On attach, allocates its own pinned CPU memory via torch pin_memory.
On detach, frees it.  stage loads shards into the pinned buffer.
h2d copies to GPU.  scatter places weights into model params.
"""
import sys, os, time, ctypes, json, struct
import glob as _glob
from concurrent.futures import ThreadPoolExecutor

import torch


SAFETENSORS_DTYPE_MAP = {
    "F16": torch.float16, "BF16": torch.bfloat16,
    "F32": torch.float32, "F64": torch.float64,
    "F8_E4M3": torch.float8_e4m3fn,
    "I8": torch.int8, "I16": torch.int16,
    "I32": torch.int32, "I64": torch.int64,
    "U8": torch.uint8, "BOOL": torch.bool,
}


def _shard_layout(model_path):
    """Read safetensors headers and return per-shard info.

    Returns list of (shard_path, data_offset, data_size, tensors) where
    tensors is [(name, start, end, dtype, shape), ...].
    """
    d = model_path.rstrip("/")
    files = sorted(_glob.glob(f"{d}/model-*.safetensors"))
    if not files:
        files = sorted(_glob.glob(f"{d}/model.safetensors*"))
    files = [f for f in files if not f.endswith(".index.json")]

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


def _scatter_into_model(model, buf_gpu, index):
    """Load weights from GPU staging buffer into model using vLLM's own
    weight loading logic (handles fused params, sharding, name mapping)."""

    def _weight_iter():
        for name in sorted(index):
            offset, length, dtype_str, shape = index[name]
            dt = SAFETENSORS_DTYPE_MAP[dtype_str]
            tensor = buf_gpu[offset:offset + length].view(dt).reshape(shape)
            yield name, tensor

    loaded = model.load_weights(_weight_iter())

    total_params = len(dict(model.named_parameters()))
    print(f"[scatter] loaded {len(loaded)}/{total_params} params "
          f"(index has {len(index)} tensors)", flush=True)

    return loaded


def _readinto_until_full(path, view, lo, size):
    with open(path, "rb") as f:
        f.seek(lo)
        while size > 0:
            n = f.readinto(view)
            if n == 0:
                raise RuntimeError(f"short read: {path}")
            size -= n
            view = view[n:]


def _load_shard(shard_path, buf_np, buf_offset, data_offset, data_size,
                use_odirect):
    """Load one shard file into its pinned fragment.

    data_offset is relative to the start of the data section in the
    safetensors file.  We add the header prefix (8-byte length + JSON
    header) to get the absolute file offset.
    """
    with open(shard_path, "rb") as fh:
        header_len = struct.unpack("<Q", fh.read(8))[0]
    abs_offset = 8 + header_len + data_offset

    if use_odirect:
        from kvikio import CuFile
        dst = buf_np[buf_offset:buf_offset + data_size]
        with CuFile(shard_path, "r") as f:
            f.read(dst, size=data_size, file_offset=abs_offset)
    else:
        view = memoryview(buf_np[buf_offset:buf_offset + data_size])
        _readinto_until_full(shard_path, view, abs_offset, data_size)


def _read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def vllm_child_loop(pipe_conn, rank, use_odirect, arch):
    """Runs in a spawned child process: owns CUDA and vLLM.

    Reads (cmd, kwargs) from pipe_conn.  Sends result tuples back
    through the same pipe; the worker's child thread relays them
    to the fork-context result_queue.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    # Run EngineCore in-process to avoid IPC serialization of GPU tensors
    # during scatter (apply_model). With multiprocessing=1, the closure
    # capturing buf_gpu gets pickled over ZMQ, which fails for models
    # >4 GiB (msgspec Ext limit) and is slow even for small models (~16s
    # vs 0.2s). With =0, apply_model calls the function directly.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    _devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(_devnull, 2)
    os.close(_devnull)

    torch.cuda.set_device(0)

    llm = None
    model_path = None
    buf_gpu = None
    index = None
    gpu_index = None

    pinned_buf = None
    fragment_info = None

    def _clog(msg):
        print(f"[gpu{rank}] [{time.strftime('%H:%M:%S', time.localtime())}] {msg}", flush=True)

    while True:
        try:
            cmd, kwargs = pipe_conn.recv()
        except EOFError:
            break

        if cmd == "exit":
            _clog("exit")
            pinned_buf = None
            fragment_info = None
            pipe_conn.send("exit_ack")
            break

        _clog(f">>> {cmd}")
        t0 = time.perf_counter()
        error = None
        info = {}

        try:
            if cmd == "init":
                vllm_config = dict(kwargs["vllm_config"])
                vllm_config["load_format"] = "dummy"
                vllm_config["enable_sleep_mode"] = True
                model_path = vllm_config["model"]

                from vllm import LLM
                llm = LLM(**vllm_config)
                info["pid"] = os.getpid()

            elif cmd == "attach":
                attach_path = kwargs.get("model_path") or model_path
                if attach_path is None:
                    raise RuntimeError("attach requires model_path (call init first or pass explicitly)")

                shards = _shard_layout(attach_path)
                total_size = sum(size for _, _, size, _ in shards)

                pinned_buf = torch.empty(total_size, dtype=torch.uint8,
                                         pin_memory=True)

                buf_offset = 0
                fragment_info = []
                for shard_path, data_offset, data_size, tensors in shards:
                    fragment_info.append((shard_path, buf_offset, data_offset, data_size))
                    buf_offset += data_size

                info["pinned_bytes"] = total_size
                _clog(f"  allocated {total_size / 2**30:.2f} GiB pinned memory")

            elif cmd == "detach":
                if pinned_buf is not None:
                    total = pinned_buf.numel()
                    pinned_buf = None
                    fragment_info = None
                    _clog(f"  freed {total / 2**30:.2f} GiB pinned memory")

            elif cmd == "sleep":
                llm.sleep(level=2)
                torch.cuda.synchronize(0)
                torch.cuda.empty_cache()

            elif cmd == "stage":
                stage_path = kwargs.get("data_path") or model_path

                if fragment_info is None:
                    raise RuntimeError("stage requires attach first")
                if not fragment_info:
                    _clog("  no fragments to stage (dummy weights?), skipping")
                    info["bytes"] = 0
                    elapsed = time.perf_counter() - t0
                    status = "OK"
                    _clog(f"<<< {cmd} {status} ({elapsed:.3f}s)")
                    info["arch"] = arch
                    pipe_conn.send((cmd, elapsed, None, info))
                    continue
                if pinned_buf is None:
                    raise RuntimeError("pinned_buf not set")

                buf_np = pinned_buf.numpy()

                if use_odirect:
                    try:
                        import kvikio.defaults
                        kvikio.defaults.set("num_threads", max(len(fragment_info), 1))
                        kvikio.defaults.set("task_size", 1 * 1024 * 1024)
                    except ImportError:
                        _clog("  kvikio not available, falling back to regular I/O")
                        use_odirect = False

                shard_files = sorted(
                    f for f in os.listdir(stage_path)
                    if f.startswith("model-") and f.endswith(".safetensors")
                )
                if not shard_files:
                    shard_files = sorted(
                        f for f in os.listdir(stage_path)
                        if f.startswith("model.safetensors") and not f.endswith(".index.json")
                    )

                frag_by_name = {}
                for shard_path, buf_offset, data_offset, data_size in fragment_info:
                    fname = os.path.basename(shard_path)
                    frag_by_name[fname] = (shard_path, buf_offset, data_offset, data_size)

                with ThreadPoolExecutor(max_workers=len(fragment_info)) as ex:
                    futures = []
                    for fname in shard_files:
                        if fname not in frag_by_name:
                            continue
                        sp, bo, do, ds = frag_by_name[fname]
                        stage_shard = os.path.join(stage_path, fname)
                        futures.append(ex.submit(
                            _load_shard, stage_shard, buf_np, bo, do, ds,
                            use_odirect))
                    for fut in futures:
                        fut.result()

                total_bytes = 0
                index = {}
                for shard_path, buf_offset, data_offset, data_size in fragment_info:
                    fname = os.path.basename(shard_path)
                    h = _read_header(os.path.join(stage_path, fname))
                    lo = data_offset
                    for name, tinfo in h.items():
                        if name == "__metadata__":
                            continue
                        start, end = tinfo["data_offsets"]
                        L = end - start
                        index[name] = (buf_offset + (start - lo), L,
                                       tinfo["dtype"], tinfo["shape"])
                    total_bytes += data_size

                info["bytes"] = total_bytes

            elif cmd == "wake_up_weights":
                llm.wake_up(tags=["weights"])

            elif cmd == "h2d":
                total_bytes = sum(ds for _, _, _, ds in fragment_info)
                info["bytes"] = total_bytes

                torch.cuda.synchronize(0)
                buf_gpu = torch.empty(total_bytes, dtype=torch.uint8,
                                      device="cuda:0")

                cpu_to_gpu = {}
                gpu_offset = 0
                for shard_path, buf_offset, data_offset, data_size in fragment_info:
                    src = pinned_buf[buf_offset:buf_offset + data_size]
                    dst = buf_gpu[gpu_offset:gpu_offset + data_size]
                    dst.copy_(src, non_blocking=True)
                    cpu_to_gpu[buf_offset] = gpu_offset
                    gpu_offset += data_size

                torch.cuda.synchronize(0)

                gpu_index = {}
                for name, (cpu_off, length, dtype_str, shape) in index.items():
                    for shard_path, buf_offset, data_offset, data_size in fragment_info:
                        if buf_offset <= cpu_off < buf_offset + data_size:
                            gpu_base = cpu_to_gpu[buf_offset]
                            gpu_off = gpu_base + (cpu_off - buf_offset)
                            gpu_index[name] = (gpu_off, length, dtype_str, shape)
                            break

            elif cmd == "scatter":
                def _scatter(model):
                    return _scatter_into_model(model, buf_gpu, gpu_index)
                llm.apply_model(_scatter)
                del _scatter
                buf_gpu.storage().resize_(0)
                del buf_gpu
                buf_gpu = None
                torch.cuda.empty_cache()

            elif cmd == "wake_up_kv_cache":
                llm.wake_up(tags=["kv_cache"])

            elif cmd == "generate":
                from vllm import SamplingParams
                sp = SamplingParams(**kwargs["sampling_params"])
                outputs = llm.generate(kwargs["prompts"], sp)
                info["outputs"] = [
                    [o.text for o in req.outputs] for req in outputs
                ]

            else:
                error = f"unknown command: {cmd}"

        except Exception as e:
            import traceback
            traceback.print_exc()
            error = f"{type(e).__name__}: {e}"

        elapsed = time.perf_counter() - t0
        status = "OK" if error is None else "FAILED"
        _clog(f"<<< {cmd} {status} ({elapsed:.3f}s)")
        info["arch"] = arch
        pipe_conn.send((cmd, elapsed, error, info))
