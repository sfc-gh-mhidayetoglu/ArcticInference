"""vLLM child process loop.

Spawned by the worker process.  Owns CUDA and vLLM.
Reads (cmd, kwargs) from a pipe, puts results on result_queue.

Init loads real weights (load_format=auto) so that vLLM runs
process_weights_after_loading and produces its internal kernel format
(Marlin-packed for GPTQ, cutlass layout for FP8, plain tensors for
BF16).

The generate path drives LLMEngine directly via add_request() + step()
instead of using the blocking LLM.generate().  This allows the child
to accept new generate requests (and other commands) while the engine
is actively decoding, enabling concurrent request handling without
asyncio or extra threads.

Attach allocates pinned CPU memory sized to model.named_parameters().
Stage snapshots the post-processed GPU parameters into the pinned
buffer.  plan_load_weights walks the param index once and caches a
chunk plan (chunk_lo, chunk_hi, members) bounded by max_buffer_bytes.
load_weights then loops over the cached plan: per chunk, copy a
slice of pinned CPU into a single reused GPU staging buffer and
scatter into model parameters by name.  If no plan is cached,
load_weights falls back to a single-chunk path.
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
    """Runs in a spawned child process: owns CUDA and vLLM.

    The main loop has two modes:
    - **Idle**: blocks on pipe_conn.recv() (zero CPU).
    - **Active** (engine has unfinished requests): alternates between
      engine.step() and non-blocking pipe_conn.poll() so new generate
      requests can be submitted mid-decode.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["USE_LIBUV"] = "0"

    torch.cuda.set_device(0)

    llm = None
    engine = None
    pinned_buf = None
    index = None       # {name: (offset, nbytes, dtype, shape)}
    chunk_plan = None  # list[(chunk_lo, chunk_hi, members)] from plan_load_weights
    chunk_size = None  # int; size of the GPU staging buffer for load_weights

    _active_reqs = {}     # req_id -> {"t0", "engine_ids", "finished"}
    _engine_to_req = {}   # engine_request_id -> req_id
    _next_engine_id = 0
    _deferred_cmds = []   # non-generate commands received during drain

    def _clog(msg):
        try:
            print(f"[gpu{rank}] [{time.strftime('%H:%M:%S', time.localtime())}] {msg}", flush=True)
        except OSError:
            pass

    def _alloc_engine_id():
        nonlocal _next_engine_id
        eid = f"req-{_next_engine_id}"
        _next_engine_id += 1
        return eid

    def _submit_generate(req_id, prompts, sampling_params_dict):
        from vllm import SamplingParams
        sp = SamplingParams(**sampling_params_dict)
        engine_ids = []
        for prompt in prompts:
            eid = _alloc_engine_id()
            engine.add_request(eid, prompt, sp)
            _engine_to_req[eid] = req_id
            engine_ids.append(eid)
        _active_reqs[req_id] = {
            "t0": time.perf_counter(),
            "engine_ids": engine_ids,
            "finished": {},
        }
        _clog(f"  submitted {len(prompts)} prompts for req_id={req_id}")

    def _process_step_outputs(step_outputs):
        for output in step_outputs:
            if not output.finished:
                continue
            eid = output.request_id
            req_id = _engine_to_req.pop(eid, None)
            if req_id is None:
                continue
            entry = _active_reqs.get(req_id)
            if entry is None:
                continue
            entry["finished"][eid] = output

            if len(entry["finished"]) == len(entry["engine_ids"]):
                ordered = [entry["finished"][e] for e in entry["engine_ids"]]
                info = {
                    "req_id": req_id,
                    "outputs": [[o.text for o in r.outputs] for r in ordered],
                    "prompt_tokens": sum(
                        len(r.prompt_token_ids) for r in ordered
                    ),
                    "completion_tokens": sum(
                        len(o.token_ids) for r in ordered for o in r.outputs
                    ),
                }
                elapsed = time.perf_counter() - entry["t0"]
                del _active_reqs[req_id]
                _clog(f"<<< generate req_id={req_id} OK ({elapsed:.3f}s)")
                pipe_conn.send(("generate_done", elapsed, None, info))

    def _drain_engine():
        while engine is not None and engine.has_unfinished_requests():
            _process_step_outputs(engine.step())

    def _handle_command(cmd, kwargs):
        nonlocal llm, engine, pinned_buf, index, chunk_plan, chunk_size

        error = None
        info = {}

        try:
            if cmd == "init":
                vllm_config = dict(kwargs["vllm_config"])
                vllm_config["enable_sleep_mode"] = True

                from vllm import LLM
                llm = LLM(**vllm_config)
                engine = llm.llm_engine
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

                info["pinned_cpu_bytes"] = total_size
                _clog(f"  allocated {total_size / 2**30:.2f} GiB pinned memory "
                      f"({len(layout)} params)")

            elif cmd == "detach":
                if pinned_buf is not None:
                    total = pinned_buf.numel()
                    pinned_buf = None
                    index = None
                    chunk_plan = None
                    chunk_size = None
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
                _drain_engine()
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

            elif cmd == "plan_load_weights":
                if pinned_buf is None or index is None:
                    raise RuntimeError(
                        "plan_load_weights requires attach first")

                total_bytes = pinned_buf.numel()
                mb = kwargs.get("max_buffer_bytes")
                cs = total_bytes if mb is None else min(int(mb), total_bytes)

                plan = []
                cur = []
                cur_lo = 0
                for name, (off, nbytes, dtype, shape) in index.items():
                    if nbytes > cs:
                        raise RuntimeError(
                            f"param {name} ({nbytes}B) exceeds "
                            f"chunk_size ({cs}B)")
                    if cur and (off + nbytes - cur_lo) > cs:
                        cur_hi = cur[-1][1] + cur[-1][2]
                        plan.append((cur_lo, cur_hi, cur))
                        cur = []
                        cur_lo = off
                    cur.append((name, off, nbytes, dtype, shape))
                if cur:
                    cur_hi = cur[-1][1] + cur[-1][2]
                    plan.append((cur_lo, cur_hi, cur))

                chunk_plan = plan
                chunk_size = cs
                info["n_chunks"] = len(plan)
                info["chunk_size"] = cs
                _clog(f"  planned {len(plan)} chunks of <= "
                      f"{cs / 2**30:.2f} GiB "
                      f"(total {total_bytes / 2**30:.2f} GiB)")

            elif cmd == "load_weights":
                if pinned_buf is None or index is None:
                    raise RuntimeError(
                        "load_weights requires attach+stage first")

                total_bytes = pinned_buf.numel()
                info["bytes"] = total_bytes

                # Use cached chunk plan if planned; otherwise fall back
                # to a single-chunk plan equivalent to the prior path.
                if chunk_plan is None:
                    plan = [(0, total_bytes,
                             [(n, o, nb, dt, sh)
                              for n, (o, nb, dt, sh) in index.items()])]
                    cs = total_bytes
                else:
                    plan = chunk_plan
                    cs = chunk_size

                # Drain any pending GPU work before the (potentially
                # large) staging-buffer alloc so the cumem allocator
                # settles on a clean contiguous block.
                torch.cuda.synchronize(0)
                buf_gpu = torch.empty(cs, dtype=torch.uint8,
                                      device="cuda:0")
                for chunk_lo, chunk_hi, members in plan:
                    n = chunk_hi - chunk_lo
                    buf_gpu[:n].copy_(pinned_buf[chunk_lo:chunk_hi],
                                      non_blocking=True)
                    torch.cuda.synchronize(0)

                    _members = members
                    _lo = chunk_lo
                    _buf = buf_gpu

                    def _scatter(model):
                        params = dict(model.named_parameters())
                        for name, off, nbytes, dtype, shape in _members:
                            start = off - _lo
                            src = (_buf[start:start + nbytes]
                                   .view(dtype).reshape(shape))
                            params[name].data.copy_(src)
                        return len(_members)

                    llm.apply_model(_scatter)
                    torch.cuda.synchronize(0)

                _clog(f"  loaded {len(index)} params in "
                      f"{len(plan)} chunk(s) "
                      f"(chunk<= {cs / 2**30:.2f} GiB, "
                      f"total {total_bytes / 2**30:.2f} GiB)")

                # Free staging buffer through PyTorch's caching allocator
                # so block metadata stays consistent across CRIU cycles.
                buf_gpu.storage().resize_(0)
                del buf_gpu
                torch.cuda.empty_cache()

            elif cmd == "wake_up_kv_cache":
                llm.wake_up(tags=["kv_cache"])

            elif cmd == "get_pipe_fd":
                info["pipe_fd"] = pipe_conn.fileno()

            elif cmd == "prepare_criu_dump":
                _drain_engine()

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

                import glob as _criu_glob
                for _sem in _criu_glob.glob("/dev/shm/sem.*"):
                    try:
                        os.remove(_sem)
                    except OSError:
                        pass

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

            else:
                error = f"unknown command: {cmd}"

        except Exception as e:
            import traceback
            traceback.print_exc()
            error = f"{type(e).__name__}: {e}"

        return error, info

    # -- Main loop --------------------------------------------------------------

    # After a CRIU restore the original stdout/stderr fds are stale and the
    # first write raises OSError.  Redirect to a per-rank log file (instead
    # of /dev/null) so that any traceback/_clog from a restored child is
    # still captured for post-mortem debugging.
    _stdout_fixed = False
    _child_log_path = f"/tmp/vllm_child_rank{rank}.log"

    # The current in-flight command, captured outside the per-iteration
    # scope so the fatal-error reporter below can blame the right cmd.
    cmd = None

    try:
        while True:
            if engine is None and llm is not None:
                engine = llm.llm_engine

            has_active = (engine is not None
                          and engine.has_unfinished_requests())

            if has_active:
                _process_step_outputs(engine.step())
                if not pipe_conn.poll(0):
                    continue

            if _deferred_cmds:
                cmd, kwargs = _deferred_cmds.pop(0)
            else:
                try:
                    cmd, kwargs = pipe_conn.recv()
                except EOFError:
                    break

            if not _stdout_fixed:
                try:
                    sys.stdout.write("")
                    sys.stdout.flush()
                except OSError:
                    _logfp = open(_child_log_path, "a", buffering=1)
                    sys.stdout = _logfp
                    sys.stderr = _logfp
                    _stdout_fixed = True
                    _clog(f"stdout/stderr redirected to {_child_log_path} "
                          f"after CRIU restore")

            if cmd == "exit":
                _drain_engine()
                _clog("exit")
                pinned_buf = None
                pipe_conn.send("exit_ack")
                break

            _clog(f">>> {cmd}")

            if cmd == "generate":
                req_id = kwargs.get("req_id")
                if req_id is None:
                    req_id = f"auto-{_next_engine_id}"
                try:
                    _submit_generate(req_id, kwargs["prompts"],
                                     kwargs["sampling_params"])
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    pipe_conn.send(("generate_done", 0.0,
                                    f"{type(e).__name__}: {e}",
                                    {"req_id": req_id}))
                # Drain any additional generate commands already on the pipe
                # so they get added to the engine before the first step().
                while pipe_conn.poll(0):
                    try:
                        cmd2, kwargs2 = pipe_conn.recv()
                    except EOFError:
                        break
                    if cmd2 == "generate":
                        rid2 = kwargs2.get("req_id",
                                           f"auto-{_next_engine_id}")
                        try:
                            _submit_generate(rid2, kwargs2["prompts"],
                                             kwargs2["sampling_params"])
                        except Exception as e2:
                            import traceback
                            traceback.print_exc()
                            pipe_conn.send(("generate_done", 0.0,
                                            f"{type(e2).__name__}: {e2}",
                                            {"req_id": rid2}))
                    else:
                        _clog(f">>> {cmd2} (deferred)")
                        _deferred_cmds.append((cmd2, kwargs2))
                continue

            t0 = time.perf_counter()
            error, info = _handle_command(cmd, kwargs)
            elapsed = time.perf_counter() - t0
            status = "OK" if error is None else "FAILED"
            _clog(f"<<< {cmd} {status} ({elapsed:.3f}s)")
            pipe_conn.send((cmd, elapsed, error, info))

    except BaseException as _fatal:
        # Last-resort reporter: any unhandled exception in the main loop
        # (including KeyboardInterrupt, SystemExit) gets a final error
        # frame on the pipe so the worker can attribute the failure to a
        # specific cmd instead of just seeing "child pipe broken".  Both
        # the traceback and the offending cmd are logged to the per-rank
        # log file via _clog so the post-mortem survives a CRIU restore.
        import traceback as _tb
        _trace = _tb.format_exc()
        _clog(f"FATAL in main loop (cmd={cmd}): "
              f"{type(_fatal).__name__}: {_fatal}")
        _clog(_trace)
        try:
            pipe_conn.send((
                cmd if cmd is not None else "__fatal__",
                0.0,
                f"FATAL {type(_fatal).__name__}: {_fatal}",
                {"traceback": _trace, "cmd": cmd},
            ))
        except Exception:
            pass
        raise
