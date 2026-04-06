"""vLLM child process loop.

Spawned by the worker process.  Owns CUDA and vLLM.
Reads (cmd, kwargs) from a pipe, puts results on result_queue.

Init loads real weights (load_format=auto) so that vLLM runs
process_weights_after_loading and produces its internal kernel format
(Marlin-packed for GPTQ, cutlass layout for FP8, plain tensors for
BF16).

Attach allocates pinned CPU memory sized to model.named_parameters().
Stage snapshots the post-processed GPU parameters into the pinned
buffer.  h2d copies the pinned buffer to a GPU staging buffer.
Scatter copies directly into model parameters by name.
"""
import ctypes, os, sys, time

import torch

_cudart = ctypes.CDLL("libcudart.so")
_cudart.cudaHostUnregister.argtypes = [ctypes.c_void_p]
_cudart.cudaHostUnregister.restype = ctypes.c_int
_cudart.cudaHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
_cudart.cudaHostRegister.restype = ctypes.c_int


def _unpin_buffer(buf):
    ret = _cudart.cudaHostUnregister(ctypes.c_void_p(buf.data_ptr()))
    if ret != 0:
        raise RuntimeError(f"cudaHostUnregister failed with cudaError={ret}")


def _repin_buffer(buf):
    ret = _cudart.cudaHostRegister(
        ctypes.c_void_p(buf.data_ptr()),
        ctypes.c_size_t(buf.numel() * buf.element_size()),
        ctypes.c_uint(0),
    )
    if ret != 0:
        raise RuntimeError(f"cudaHostRegister failed with cudaError={ret}")


def vllm_child_loop(pipe_conn, rank):
    """Runs in a spawned child process: owns CUDA and vLLM."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["USE_LIBUV"] = "0"

    torch.cuda.set_device(0)

    llm = None
    pinned_buf = None
    index = None       # {name: (offset, nbytes, dtype, shape)}
    buf_gpu = None
    gpu_index = None

    def _clog(msg):
        try:
            print(f"[gpu{rank}] [{time.strftime('%H:%M:%S', time.localtime())}] {msg}", flush=True)
        except OSError:
            pass

    _stdout_fixed = False

    while True:
        try:
            cmd, kwargs = pipe_conn.recv()
        except EOFError:
            break

        if not _stdout_fixed:
            try:
                sys.stdout.write("")
                sys.stdout.flush()
            except OSError:
                _devnull = open(os.devnull, "w")
                sys.stdout = _devnull
                sys.stderr = _devnull
                _stdout_fixed = True

        if cmd == "exit":
            _clog("exit")
            pinned_buf = None
            pipe_conn.send("exit_ack")
            break

        _clog(f">>> {cmd}")
        t0 = time.perf_counter()
        error = None
        info = {}

        try:
            if cmd == "init":
                vllm_config = dict(kwargs["vllm_config"])
                vllm_config["enable_sleep_mode"] = True

                from vllm import LLM
                llm = LLM(**vllm_config)
                info["pid"] = os.getpid()

            elif cmd == "attach":
                if llm is None:
                    raise RuntimeError("attach requires init first")

                def _compute_layout(model):
                    layout = []
                    for name, param in model.named_parameters():
                        d = param.data
                        layout.append((name, d.nbytes, d.dtype, tuple(d.shape)))
                    return layout

                layout = llm.apply_model(_compute_layout)[0]
                total_size = sum(nbytes for _, nbytes, _, _ in layout)

                pinned_buf = torch.empty(total_size, dtype=torch.uint8)

                index = {}
                offset = 0
                for name, nbytes, dtype, shape in layout:
                    index[name] = (offset, nbytes, dtype, shape)
                    offset += nbytes

                info["pinned_bytes"] = total_size
                _clog(f"  allocated {total_size / 2**30:.2f} GiB pinned memory "
                      f"({len(layout)} params)")

            elif cmd == "detach":
                if pinned_buf is not None:
                    total = pinned_buf.numel()
                    pinned_buf = None
                    index = None
                    _clog(f"  freed {total / 2**30:.2f} GiB pinned memory")

            elif cmd == "unpin":
                if pinned_buf is None:
                    raise RuntimeError("unpin requires attach first")
                _unpin_buffer(pinned_buf)
                _clog(f"  unpinned {pinned_buf.numel() / 2**30:.2f} GiB")

            elif cmd == "repin":
                if pinned_buf is None:
                    raise RuntimeError("repin requires attach first")
                _repin_buffer(pinned_buf)
                _clog(f"  repinned {pinned_buf.numel() / 2**30:.2f} GiB")

            elif cmd == "sleep":
                llm.sleep(level=2)
                torch.cuda.synchronize(0)
                torch.cuda.empty_cache()

            elif cmd == "stage":
                if pinned_buf is None:
                    raise RuntimeError("stage requires attach first")
                if index is None:
                    raise RuntimeError("no index (call attach first)")

                _pinned = pinned_buf
                _index = index

                def _stage_weights(model):
                    for name, param in model.named_parameters():
                        offset, nbytes, dtype, shape = _index[name]
                        src = param.data.contiguous().reshape(-1).view(torch.uint8)
                        _pinned[offset:offset + nbytes].copy_(src, non_blocking=True)
                    torch.cuda.synchronize()

                llm.apply_model(_stage_weights)

                total_bytes = pinned_buf.numel()
                info["bytes"] = total_bytes
                _clog(f"  staged {len(index)} params "
                      f"({total_bytes / 2**30:.2f} GiB)")

            elif cmd == "wake_up_weights":
                llm.wake_up(tags=["weights"])

            elif cmd == "h2d":
                if pinned_buf is None or index is None:
                    raise RuntimeError("h2d requires attach+stage first")

                total_bytes = pinned_buf.numel()
                info["bytes"] = total_bytes

                torch.cuda.synchronize(0)
                buf_gpu = torch.empty(total_bytes, dtype=torch.uint8,
                                      device="cuda:0")
                buf_gpu.copy_(pinned_buf, non_blocking=True)
                torch.cuda.synchronize(0)

                gpu_index = index

            elif cmd == "scatter":
                if buf_gpu is None or gpu_index is None:
                    raise RuntimeError("scatter requires h2d first")

                _buf = buf_gpu
                _gi = gpu_index

                def _scatter_direct(model):
                    params = dict(model.named_parameters())
                    loaded = 0
                    for name, (offset, nbytes, dtype, shape) in _gi.items():
                        src = _buf[offset:offset + nbytes].view(dtype).reshape(shape)
                        params[name].data.copy_(src)
                        loaded += 1
                    return loaded

                result = llm.apply_model(_scatter_direct)
                _clog(f"  scattered {result[0]}/{len(gpu_index)} params")

                buf_gpu.storage().resize_(0)
                del buf_gpu
                buf_gpu = None
                gpu_index = None
                torch.cuda.empty_cache()

            elif cmd == "wake_up_kv_cache":
                llm.wake_up(tags=["kv_cache"])

            elif cmd == "get_pipe_fd":
                info["pipe_fd"] = pipe_conn.fileno()

            elif cmd == "prepare_criu_dump":
                closed_fds = []
                unmapped = []
                destroyed_pg = False
                pid = os.getpid()

                try:
                    import torch.distributed as dist
                    if dist.is_initialized():
                        dist.destroy_process_group()
                        destroyed_pg = True
                except Exception as _e:
                    _clog(f"  prepare_criu_dump: dist teardown error: {_e}")

                if destroyed_pg:
                    store_names = ("pt_tcpstore", "pt_nccl_watchdg",
                                   "pt_nccl_heartbt")
                    for _attempt in range(50):
                        alive = []
                        for tid_name in os.listdir(f"/proc/{pid}/task"):
                            try:
                                comm = open(
                                    f"/proc/{pid}/task/{tid_name}/comm"
                                ).read().strip()
                                if any(comm.startswith(s)
                                       for s in store_names):
                                    alive.append(f"{tid_name}({comm})")
                            except (OSError, ValueError):
                                pass
                        if not alive:
                            _clog(f"  prepare_criu_dump: store threads "
                                  f"exited after {_attempt} polls")
                            break
                        time.sleep(0.05)
                    else:
                        _clog(f"  prepare_criu_dump: WARNING store threads "
                              f"still alive: {alive}")

                pipe_fd = kwargs.get("pipe_fd", -1)
                devnull = os.open(os.devnull, os.O_RDWR)
                for std_fd in (1, 2):
                    os.dup2(devnull, std_fd)
                os.close(devnull)

                keep_prefixes = ("/dev/nvidia", "/dev/shm", "anon_inode:",
                                 "socket:", "pipe:")
                for fd_name in sorted(os.listdir(f"/proc/{pid}/fd"),
                                      key=int):
                    try:
                        fd_int = int(fd_name)
                        if fd_int == pipe_fd or fd_int <= 2:
                            continue
                        link = os.readlink(f"/proc/{pid}/fd/{fd_name}")
                        if any(link.startswith(p) for p in keep_prefixes):
                            continue
                        os.close(fd_int)
                        closed_fds.append(fd_int)
                    except (OSError, ValueError):
                        pass

                libc = ctypes.CDLL("libc.so.6")
                with open(f"/proc/{pid}/maps") as f:
                    for line in f:
                        if "io_uring" in line:
                            addr_range = line.split()[0]
                            start_s, end_s = addr_range.split("-")
                            start = int(start_s, 16)
                            length = int(end_s, 16) - start
                            libc.munmap(ctypes.c_void_p(start),
                                        ctypes.c_size_t(length))
                            unmapped.append(f"0x{start:x}")

                remaining_threads = []
                for tid_name in os.listdir(f"/proc/{pid}/task"):
                    try:
                        comm = open(
                            f"/proc/{pid}/task/{tid_name}/comm"
                        ).read().strip()
                        if comm != "python":
                            remaining_threads.append(f"{tid_name}({comm})")
                    except (OSError, ValueError):
                        pass
                info["closed_fds"] = closed_fds
                info["unmapped"] = unmapped
                info["destroyed_pg"] = destroyed_pg
                info["remaining_threads"] = remaining_threads
                _clog(f"  prepare_criu_dump: fds={closed_fds}, "
                      f"unmapped={unmapped}, destroyed_pg={destroyed_pg}, "
                      f"remaining_threads={remaining_threads}")

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
        pipe_conn.send((cmd, elapsed, error, info))
