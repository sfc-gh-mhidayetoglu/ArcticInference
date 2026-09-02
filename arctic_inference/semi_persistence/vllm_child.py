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
buffer.  plan_restore_weights walks the param index once and caches
a chunk plan (chunk_lo, chunk_hi, members) bounded by max_buffer_bytes.
restore_weights then loops over the cached plan: per chunk, copy a
slice of pinned CPU into a single reused GPU staging buffer and
scatter into model parameters by name.  If no plan is cached,
restore_weights falls back to a single-chunk path.
"""
import ctypes, json, os, shutil, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

import torch

import semip_logging

def _truncate_for_display(value, limit=200):
    """Truncate strings (or strings inside a list/tuple) to ``limit`` chars,
    appending ``...(<n> chars)`` when the original exceeds ``limit``.
    """
    if isinstance(value, str):
        if len(value) > limit:
            return f"{value[:limit]}...({len(value)} chars)"
        return value
    if isinstance(value, (list, tuple)):
        out = [_truncate_for_display(v, limit) for v in value]
        return out if isinstance(value, list) else tuple(out)
    return value


# Shard size / parallelism for save_weights + load_weights disk I/O.
_WEIGHTS_SHARD_BYTES = 2 * 2**30   # 2 GiB per shard
_WEIGHTS_IO_WORKERS = 8            # thread pool size for shard I/O

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


def vllm_child_loop(pipe_conn, instance_id, rank):
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

    semip_logging.init_process()
    log = semip_logging.child(instance_id, rank)
    # First-run path: route this process's stdout/stderr to the shared
    # per-instance log file from the very first byte.  CRIU later dumps
    # fd 1/2 as regular-file references to this path.
    #
    # Across runs the path baked into the image can be stale: if this
    # model was instance N when it was dumped, the restored child will
    # re-open /tmp/instN.log even when the orchestrator has now placed
    # it at a different instance_id (so its output silently leaks into
    # another instance's log).  To fix that, the worker sends a
    # ``rebind_log`` command immediately after CRIU restore (handled in
    # ``_handle_command`` below), which dup2s fd 1/2 onto the file
    # matching the *current* instance_id and rebuilds the log adapter.
    _child_log_path = semip_logging.redirect_stdio_to_instance_file(
        instance_id)

    torch.cuda.set_device(0)

    llm = None
    engine = None
    pinned_buf = None
    pinned_via_attach_pinned = False  # True iff buffer was alloc'd with pin_memory=True
    index = None       # {name: (offset, nbytes, dtype, shape)}
    chunk_plan = None  # list[(chunk_lo, chunk_hi, members)] from plan_restore_weights
    chunk_size = None  # int; size of the GPU staging buffer for restore_weights

    _active_reqs = {}     # req_id -> {"t0", "engine_ids", "finished"}
    _engine_to_req = {}   # engine_request_id -> req_id
    _next_engine_id = 0
    _deferred_cmds = []   # non-generate commands received during drain

    # Pause/resume state.  `_paused` gates the engine.step() call in
    # the main loop, and is also the single switch that routes
    # generate-while-paused submits: `_submit_generate` parks them
    # in `_saved_requests` (skipping the engine entirely) iff
    # `_paused` is True.  `_saved_requests` is populated by both
    # `pause` (which snapshots active requests and aborts them in
    # the engine) and post-pause `_submit_generate` (which
    # synthesises a never-stepped record), then drained on `resume`
    # via `engine.add_request` for each entry.  All of this lives
    # in plain Python state so CRIU dumps and restores it for free
    # across cuda_checkpoint/cuda_restore cycles, and keeping the
    # engine untouched while paused makes the path robust to any
    # pipe interleaving of pause / sleep / cuda_checkpoint /
    # generate that the orchestrator's "Walking down past `up`
    # while paused" rule permits.
    #
    # `_dormant` is a separate, *defensive* flag that brackets the
    # span where the vLLM engine is unsafe to mutate because
    # `llm.sleep(level=2)` discarded its KV cache and possibly
    # `cuda-checkpoint` froze the entire CUDA context.  Set True at
    # the bottom of the `sleep` handler, False at the bottom of the
    # `wake_up_kv_cache` handler.  `_submit_generate` checks it
    # BEFORE `_paused` and, when True, sends back a `generate_done`
    # ack carrying a `RuntimeError("generate against dormant
    # engine")` instead of touching the engine -- so any race that
    # slips past the orchestrator's Phase-2 eviction sentinel
    # (`Orchestrator._evict_for_phase2`) surfaces as a loud, fail-
    # fast future exception in `_on_generate_done` instead of a
    # silent hang inside `engine.step()` on a torn-down executor.
    # Defense in depth: with the sentinel intact this branch is
    # unreachable in normal operation; the historical record of the
    # `_engine_dormant` / `_paused` unification (commit `ad74086`)
    # is in `orchestrator_DESIGN.md` "Eviction-mid-generate
    # dormant-engine wedge".
    _paused = False
    _dormant = False
    _saved_requests = []

    def _alloc_engine_id():
        nonlocal _next_engine_id
        eid = f"req-{_next_engine_id}"
        _next_engine_id += 1
        return eid

    def _submit_generate(req_id, prompts, sampling_params_dict):
        if _dormant and not _paused:
            # Defense-in-depth fail-fast: the orchestrator should
            # never enqueue a generate cmd onto an engine that has
            # been put to sleep without a corresponding pause (the
            # Phase-2 eviction sentinel in
            # ``Orchestrator._evict_for_phase2`` gates this).  If
            # that gate ever has a hole, abort with a loud error
            # ack instead of silently hanging inside
            # ``engine.step()`` on a torn-down executor.  Routes
            # through the demuxer's standard error path
            # (``error is not None`` on the result tuple), which
            # latches the error, decrements ``_pending_count``
            # cleanly, and surfaces to ``Orchestrator.
            # _on_generate_done`` as the ``error`` arg so the
            # in-flight ``done_event.set()`` happens with
            # ``q_rec["state"]="error"``.
            err = RuntimeError(
                f"generate req_id={req_id} arrived against dormant "
                f"engine (sleep without prior pause); orchestrator "
                f"sentinel breach -- see orchestrator_DESIGN.md "
                f"'Eviction-mid-generate dormant-engine wedge'")
            log.error(
                "  _dormant fail-fast: rejecting req_id=%s "
                "(prompts=%s)  -- %s",
                req_id, _truncate_for_display(list(prompts)), err)
            pipe_conn.send((
                "generate_done", 0.0, err, {"req_id": req_id}))
            return
        if _paused:
            # Single rule: while paused, the vLLM engine sees no
            # scheduler mutations from this child.  Park the request
            # in `_saved_requests` and let the next `resume` reload
            # it; `pause` already did the same for whatever was
            # in-flight at pause-time, so on resume the deferred
            # entries and the pause-snapshotted entries flow back
            # into the engine through one code path.
            #
            # This keeps the engine untouched for the entire dormant
            # span -- `llm.sleep` discards cumem-allocated KV blocks
            # and `cuda-checkpoint` (during `cuda_checkpoint`)
            # freezes the CUDA context, so any `engine.add_request`
            # / `engine.abort_request` call inside that window would
            # either enqueue into a scheduler that can never `step`
            # or block on a torn-down executor.  It is also
            # order-independent w.r.t. pipe interleavings of
            # generate/sleep/checkpoint/etc. while paused.
            #
            # `prompt_token_ids: []` (not None) matches the shape
            # `_snapshot_active_into_saved` produces for an empty
            # per-eid state via its `list(... or [])` clause, so
            # the resume branch's `len(prompt_tids)` works and its
            # `if prompt_tids:` test falls through to the
            # `elif i < len(prompts_orig)` re-prefill branch.
            _saved_requests.append({
                "req_id": req_id,
                "t0": time.perf_counter(),
                "first_token_ts": None,
                "prompts": list(prompts),
                "sampling_params": dict(sampling_params_dict),
                "eids": [{"prompt_token_ids": [],
                          "output_token_ids": [],
                          "output_text": ""}
                         for _ in prompts],
            })
            log.info("  submitted req_id=%s  prompts=%s  "
                     "(deferred to _saved_requests; paused)",
                     req_id, _truncate_for_display(list(prompts)))
            return

        from vllm import SamplingParams
        sp = SamplingParams(**sampling_params_dict)
        engine_ids = []
        for prompt in prompts:
            eid = _alloc_engine_id()
            engine.add_request(eid, prompt, sp)
            _engine_to_req[eid] = req_id
            engine_ids.append(eid)
        # `per_eid` tracks the latest cumulative engine output per
        # sub-request so that `pause` can snapshot the current state
        # without poking engine internals.  Updated in
        # `_process_step_outputs` on every step.
        per_eid = {eid: {"prompt_token_ids": None,
                         "output_token_ids": [],
                         "output_text": ""} for eid in engine_ids}
        _active_reqs[req_id] = {
            "t0": time.perf_counter(),
            "engine_ids": engine_ids,
            "finished": {},
            "prompts": list(prompts),
            "first_token_ts": None,
            "sampling_params": dict(sampling_params_dict),
            "per_eid": per_eid,
        }
        log.info("  submitted req_id=%s  prompts=%s",
                 req_id, _truncate_for_display(list(prompts)))

    def _process_step_outputs(step_outputs):
        for output in step_outputs:
            eid = output.request_id
            req_id = _engine_to_req.get(eid)
            if req_id is None:
                continue
            entry = _active_reqs.get(req_id)
            if entry is None:
                continue

            # First-token detection: stamp on the first step that
            # produced any decoded tokens for any sub-request of this
            # req_id.  output_kind defaults to CUMULATIVE so token_ids
            # is the running total -- non-empty iff at least one token
            # has been generated.
            if entry["first_token_ts"] is None and any(
                    o.token_ids for o in output.outputs):
                entry["first_token_ts"] = time.perf_counter()

            # Per-eid cumulative snapshot used by `pause`.  This must
            # happen on every step (not just the finishing one)
            # because pause can be invoked mid-decode.  We track only
            # the n=1 case (outputs[0]).
            per_eid_state = entry.get("per_eid", {}).get(eid)
            if per_eid_state is not None:
                if (per_eid_state["prompt_token_ids"] is None
                        and output.prompt_token_ids):
                    per_eid_state["prompt_token_ids"] = list(
                        output.prompt_token_ids)
                if output.outputs:
                    per_eid_state["output_token_ids"] = list(
                        output.outputs[0].token_ids)
                    per_eid_state["output_text"] = output.outputs[0].text

            if not output.finished:
                continue
            _engine_to_req.pop(eid, None)
            entry["finished"][eid] = output

            if len(entry["finished"]) == len(entry["engine_ids"]):
                ordered = [entry["finished"][e] for e in entry["engine_ids"]]

                # If this entry was resumed via `resume`, fold
                # pre-pause output back into the reported view
                # so the caller sees seamless continuation.
                pre_completion = entry.get("pre_pause_completion")
                pre_text = entry.get("pre_pause_text")
                orig_prompt_tokens = entry.get("original_prompt_tokens")

                if pre_completion is not None:
                    eid_index = {e: i for i, e in enumerate(entry["engine_ids"])}
                    outputs = [
                        [pre_text[eid_index[r.request_id]] + o.text
                         for o in r.outputs]
                        for r in ordered]
                    completion_tokens = sum(
                        len(o.token_ids)
                        + pre_completion[eid_index[r.request_id]]
                        for r in ordered for o in r.outputs)
                    prompt_tokens = sum(orig_prompt_tokens)
                else:
                    outputs = [[o.text for o in r.outputs] for r in ordered]
                    prompt_tokens = sum(
                        len(r.prompt_token_ids) for r in ordered)
                    completion_tokens = sum(
                        len(o.token_ids) for r in ordered for o in r.outputs)
                cached_tokens = sum(
                    (r.num_cached_tokens or 0) for r in ordered)
                finish_reasons = sorted({
                    o.finish_reason for r in ordered for o in r.outputs
                    if o.finish_reason is not None
                })
                info = {
                    "req_id": req_id,
                    "outputs": outputs,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "num_cached_tokens": cached_tokens,
                    "finish_reasons": finish_reasons,
                }

                t_done = time.perf_counter()
                elapsed = t_done - entry["t0"]
                first_token_ts = entry["first_token_ts"]
                ttft = (first_token_ts - entry["t0"]
                        if first_token_ts is not None else None)
                decode_time = (t_done - first_token_ts
                               if first_token_ts is not None else None)
                tpot_ms = (decode_time * 1000.0 / (completion_tokens - 1)
                           if (decode_time is not None
                               and completion_tokens > 1) else None)
                gen_tput = (completion_tokens / elapsed
                            if elapsed > 0 else 0.0)

                info["ttft_s"] = ttft
                info["decode_s"] = decode_time
                info["tpot_ms"] = tpot_ms
                info["gen_tput_tok_s"] = gen_tput

                prompts = entry.get("prompts")
                del _active_reqs[req_id]
                log.info(
                    "<<< generate req_id=%s OK (%.3fs)  "
                    "prompt_tokens=%s  completion_tokens=%s  "
                    "cached_tokens=%s  finish=%s  "
                    "prompt=%s  output=%s",
                    req_id, elapsed,
                    prompt_tokens, completion_tokens,
                    cached_tokens, finish_reasons,
                    _truncate_for_display(prompts),
                    _truncate_for_display(outputs),
                )
                log.info(
                    "    perf  ttft=%s  decode=%s  tpot=%s  "
                    "gen_tput=%.1f tok/s",
                    f"{ttft * 1000:.1f}ms" if ttft is not None else "n/a",
                    (f"{decode_time:.3f}s"
                     if decode_time is not None else "n/a"),
                    (f"{tpot_ms:.2f}ms"
                     if tpot_ms is not None else "n/a"),
                    gen_tput,
                )
                pipe_conn.send(("generate_done", elapsed, None, info))

    def _drain_engine():
        while engine is not None and engine.has_unfinished_requests():
            _process_step_outputs(engine.step())

    def _snapshot_active_into_saved() -> tuple[int, int]:
        """Snapshot every active sub-request into ``_saved_requests`` and
        abort it in the engine.  Mirrors the pause-time snapshot path,
        factored out so callers other than ``pause`` (e.g. a ``sleep``
        arriving on a paused engine that picked up a generate after
        pause) can preserve those requests for the next ``resume``
        instead of silently draining them to completion.

        Returns ``(saved_count, aborted_eid_count)``.  Safe to call when
        ``_active_reqs`` is empty (returns ``(0, 0)``).  Caller is
        responsible for any state flag updates (``_paused``) and for
        emitting the appropriate log line; this helper only touches the
        ledgers.
        """
        saved = []
        for req_id, entry in list(_active_reqs.items()):
            sp_dict = entry.get("sampling_params") or {}
            n_branch = sp_dict.get("n", 1)
            if n_branch != 1:
                raise RuntimeError(
                    f"snapshot with n={n_branch} not supported "
                    "(n=1 only)")
            eids_data = []
            for eid in entry["engine_ids"]:
                per_eid_state = entry["per_eid"].get(eid, {})
                eids_data.append({
                    "prompt_token_ids": list(
                        per_eid_state.get("prompt_token_ids") or []),
                    "output_token_ids": list(
                        per_eid_state.get("output_token_ids") or []),
                    "output_text":
                        per_eid_state.get("output_text", ""),
                })
            saved.append({
                "req_id": req_id,
                "t0": entry["t0"],
                "first_token_ts": entry["first_token_ts"],
                "prompts": list(entry.get("prompts") or []),
                "sampling_params": dict(sp_dict),
                "eids": eids_data,
            })

        all_eids = [eid
                    for entry in _active_reqs.values()
                    for eid in entry["engine_ids"]]
        if all_eids:
            try:
                engine.abort_request(all_eids)
            except Exception as _e:
                log.warning("  snapshot: abort_request failed: %s", _e)

        _saved_requests.extend(saved)
        _active_reqs.clear()
        _engine_to_req.clear()
        return len(saved), len(all_eids)

    def _handle_command(cmd, kwargs):
        nonlocal llm, engine, pinned_buf, pinned_via_attach_pinned
        nonlocal index, chunk_plan, chunk_size
        nonlocal _paused, _dormant
        nonlocal log, _child_log_path

        error = None
        info = {}

        try:
            if cmd == "init":
                vllm_config = dict(kwargs["vllm_config"])
                vllm_config["enable_sleep_mode"] = True

                # Per-model env vars: vllm_config["_env"] is a reserved
                # mapping applied to os.environ before vLLM is imported,
                # so flags vLLM reads at import time take effect.  The
                # trio set at the top of vllm_child_loop is reserved
                # (CUDA isolation + in-process EngineCore + libuv off);
                # silently drop any attempt to override it from _env.
                _RESERVED_ENV = {
                    "CUDA_VISIBLE_DEVICES",
                    "VLLM_ENABLE_V1_MULTIPROCESSING",
                    "USE_LIBUV",
                }
                for k, v in (vllm_config.pop("_env", None) or {}).items():
                    if k in _RESERVED_ENV:
                        log.warning(
                            "ignoring reserved env key in _env: %s", k)
                        continue
                    os.environ[k] = str(v)

                # Force vLLM plugins (e.g. arctic_inference) to load
                # *before* `LLM(**vllm_config)` so plugin-installed
                # EngineArgs fields like `ulysses_sequence_parallel_size`
                # are present when EngineArgs is instantiated from
                # vllm_config.  vLLM normally loads plugins itself
                # during LLMEngine construction, but that fires too
                # late for plugins that extend the EngineArgs dataclass.
                # NB: `vllm.plugins` is a submodule, not auto-attached
                # to the `vllm` package on `import vllm` -- use the
                # `from vllm.plugins import ...` form.
                from vllm.plugins import load_general_plugins
                load_general_plugins()
                from vllm import LLM
                llm = LLM(**vllm_config)
                engine = llm.llm_engine
                info["pid"] = os.getpid()

                # Opt this worker out of arctic_inference's level-2
                # sleep/wake fast paths.  We restore main and drafter
                # params from a host-side pinned buffer ourselves
                # (stage / restore_weights), so:
                #   - skip the disk reload of the main model on wake_up
                #   - skip the per-sleep CPU snapshot of drafter
                #     ``named_parameters()`` (drafter ``named_buffers()``
                #     are still snapshotted; sub-MB)
                # Default arctic behavior is preserved for other users
                # because the flags are read via ``getattr(..., False)``.
                #
                # Note: ``GPUModelRunnerPatch.reload_weights`` is not
                # gated -- semi-persistence never calls
                # ``model_runner.reload_weights`` (the patched
                # ``Worker.wake_up`` reaches the unpatched original via
                # ``GPUModelRunnerPatch._orig_reload_weights``), so the
                # drafter-load augmentation in that path is never hit
                # from this child.
                def _enable_semi_persistence_flags(self):
                    # ``self`` here is the ``WorkerWrapperBase`` driver
                    # (see ``UniProcExecutor.collective_rpc`` ->
                    # ``run_method(self.driver_worker, ...)``).  Plain
                    # ``self.X = ...`` writes onto the wrapper; the
                    # wrapper only forwards ``__getattr__`` to
                    # ``self.worker``, so arctic's patched
                    # ``Worker.wake_up`` (where ``self`` is the real
                    # ``Worker``) would never see these flags and would
                    # fall back to the disk reload.  Write through to
                    # ``self.worker`` so the gating actually fires.
                    self.worker._skip_main_reload_on_wake = True
                    self.worker._skip_drafter_param_snapshot = True

                llm.collective_rpc(_enable_semi_persistence_flags)

            elif cmd == "attach" or cmd == "attach_pinned":
                if llm is None:
                    raise RuntimeError(f"{cmd} requires init first")

                # Walk both the main model and the drafter (if it has a
                # ``.model`` attribute -- Eagle / Medusa / DraftModel /
                # ArcticProposer).  Non-model drafters (Ngram / Suffix)
                # have ``drafter.model is None`` and are skipped, so the
                # layout collapses to main params only -- no behavior
                # change vs. ``apply_model(_compute_layout)``.
                #
                # Names are namespaced so ``stage`` and ``restore_weights``
                # can dispatch each entry back to the right tensor table:
                #   "main:p:<name>"     -> main.named_parameters()[name]
                #   "drafter:p:<name>"  -> drafter.model.named_parameters()[name]
                # Buffers (``named_buffers()``) are intentionally not
                # staged -- main buffers ride stock vLLM's
                # ``_sleep_saved_buffers`` snapshot, drafter buffers
                # ride arctic's ``_save_module_state`` snapshot.
                def _compute_layout_full(self):
                    layout = []
                    main = self.model_runner.model
                    for name, p in main.named_parameters():
                        d = p.data
                        layout.append((f"main:p:{name}", d.nbytes,
                                       d.dtype, tuple(d.shape)))
                    drafter = getattr(self.model_runner, "drafter", None)
                    dm = (getattr(drafter, "model", None)
                          if drafter is not None else None)
                    if dm is not None:
                        for name, p in dm.named_parameters():
                            d = p.data
                            layout.append((f"drafter:p:{name}", d.nbytes,
                                           d.dtype, tuple(d.shape)))
                    return layout

                layout = llm.collective_rpc(_compute_layout_full)[0]
                total_size = sum(nbytes for _, nbytes, _, _ in layout)

                # attach_pinned: pin_memory=True routes through PyTorch's
                # CachingHostAllocator (cudaHostAlloc), so the buffer stays
                # pinned for its entire lifetime and skips the per-cycle
                # repin()/unpin() pattern at the cost of ~34 ms/GiB extra
                # inside cuCheckpointProcess{Checkpoint,Restore}.
                if cmd == "attach_pinned":
                    pinned_buf = torch.empty(total_size, dtype=torch.uint8,
                                             pin_memory=True)
                    pinned_via_attach_pinned = True
                else:
                    pinned_buf = torch.empty(total_size, dtype=torch.uint8)
                    pinned_via_attach_pinned = False

                index = {}
                offset = 0
                for name, nbytes, dtype, shape in layout:
                    index[name] = (offset, nbytes, dtype, shape)
                    offset += nbytes

                info["pinned_cpu_bytes"] = total_size
                _label = ("pinned host memory (pin_memory=True)"
                          if pinned_via_attach_pinned else "pinned memory")
                log.info("  allocated %.2f GiB %s (%d params)",
                         total_size / 2**30, _label, len(layout))

            elif cmd == "detach":
                if pinned_buf is not None:
                    total = pinned_buf.numel()
                    pinned_buf = None
                    pinned_via_attach_pinned = False
                    index = None
                    chunk_plan = None
                    chunk_size = None
                    log.info("  freed %.2f GiB pinned memory", total / 2**30)

            elif cmd == "unpin":
                if pinned_buf is None:
                    raise RuntimeError("unpin requires attach first")
                if pinned_via_attach_pinned:
                    raise RuntimeError(
                        "unpin not allowed on attach_pinned() buffer; "
                        "memory stays pinned for the buffer's lifetime")
                _unpin_buffer(pinned_buf)
                log.info("  unpinned %.2f GiB", pinned_buf.numel() / 2**30)

            elif cmd == "repin":
                if pinned_buf is None:
                    raise RuntimeError("repin requires attach first")
                if pinned_via_attach_pinned:
                    raise RuntimeError(
                        "repin not allowed on attach_pinned() buffer; "
                        "memory stays pinned for the buffer's lifetime")
                _repin_buffer(pinned_buf)
                log.info("  repinned %.2f GiB", pinned_buf.numel() / 2**30)

            elif cmd == "sleep":
                # Invariant: while `_paused` is True, `_active_reqs`
                # is empty -- `pause` snapshot-and-aborts whatever
                # was in flight at pause time and post-pause submits
                # go straight to `_saved_requests` via
                # `_submit_generate`, never touching the engine.
                # `_drain_engine` therefore has no scheduled work to
                # step through here.  If we are not paused, the
                # drain just runs the engine to completion as in a
                # normal cold sleep.
                _drain_engine()
                llm.sleep(level=2)
                torch.cuda.synchronize(0)
                torch.cuda.empty_cache()
                # Set _dormant AFTER llm.sleep so the fail-fast in
                # _submit_generate only fires once the engine is
                # actually torn down.  The flag is a defensive net
                # against the eviction-mid-generate wedge described
                # in orchestrator_DESIGN.md; the orchestrator's
                # Phase-2 sentinel is the primary gate.
                _dormant = True

            elif cmd == "stage":
                if pinned_buf is None:
                    raise RuntimeError("stage requires attach first")
                if index is None:
                    raise RuntimeError("no index (call attach first)")

                _pinned = pinned_buf
                _index = index

                # Mirror image of ``_compute_layout_full`` in attach:
                # build a unified ``name -> tensor`` table that covers
                # main params + drafter params (drafter only if it has
                # a ``.model``).  Then drive copies off ``_index`` so
                # the source set lines up exactly with what attach
                # measured.
                def _stage_weights(self):
                    main = self.model_runner.model
                    drafter = getattr(self.model_runner, "drafter", None)
                    dm = (getattr(drafter, "model", None)
                          if drafter is not None else None)
                    sources = {f"main:p:{n}": p.data
                               for n, p in main.named_parameters()}
                    if dm is not None:
                        for n, p in dm.named_parameters():
                            sources[f"drafter:p:{n}"] = p.data
                    for name, (offset, nbytes, dtype, shape) in _index.items():
                        src = (sources[name].contiguous().reshape(-1)
                               .view(torch.uint8))
                        _pinned[offset:offset + nbytes].copy_(
                            src, non_blocking=True)
                    torch.cuda.synchronize()

                llm.collective_rpc(_stage_weights)

                total_bytes = pinned_buf.numel()
                info["bytes"] = total_bytes
                log.info("  staged %d params (%.2f GiB)",
                         len(index), total_bytes / 2**30)

            elif cmd == "wake_up_weights":
                llm.wake_up(tags=["weights"])

            elif cmd == "plan_restore_weights":
                if pinned_buf is None or index is None:
                    raise RuntimeError(
                        "plan_restore_weights requires attach first")

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
                log.info("  planned %d chunks of <= %.2f GiB (total %.2f GiB)",
                         len(plan), cs / 2**30, total_bytes / 2**30)

            elif cmd == "restore_weights":
                if pinned_buf is None or index is None:
                    raise RuntimeError(
                        "restore_weights requires attach+stage first")

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

                    # Mirror image of ``_compute_layout_full`` /
                    # ``_stage_weights``: build a unified
                    # ``name -> tensor`` table for main params + drafter
                    # params, then dispatch each chunk member to the
                    # right destination by namespaced name.
                    def _scatter(self):
                        main = self.model_runner.model
                        drafter = getattr(self.model_runner, "drafter", None)
                        dm = (getattr(drafter, "model", None)
                              if drafter is not None else None)
                        targets = {f"main:p:{n}": p.data
                                   for n, p in main.named_parameters()}
                        if dm is not None:
                            for n, p in dm.named_parameters():
                                targets[f"drafter:p:{n}"] = p.data
                        for name, off, nbytes, dtype, shape in _members:
                            start = off - _lo
                            src = (_buf[start:start + nbytes]
                                   .view(dtype).reshape(shape))
                            targets[name].copy_(src)
                        return len(_members)

                    llm.collective_rpc(_scatter)
                    torch.cuda.synchronize(0)

                log.info("  loaded %d params in %d chunk(s) "
                         "(chunk<= %.2f GiB, total %.2f GiB)",
                         len(index), len(plan),
                         cs / 2**30, total_bytes / 2**30)

                # Free staging buffer through PyTorch's caching allocator
                # so block metadata stays consistent across CRIU cycles.
                buf_gpu.storage().resize_(0)
                del buf_gpu
                torch.cuda.empty_cache()

            elif cmd == "wake_up_kv_cache":
                llm.wake_up(tags=["kv_cache"])
                # Clear _dormant AFTER wake_up so the engine is
                # actually back up before _submit_generate stops
                # short-circuiting.  Pairs with the set in the
                # `sleep` handler.
                _dormant = False

            elif cmd == "pause":
                if engine is None:
                    raise RuntimeError("pause requires init first")

                was_paused = _paused
                _paused = True

                # Snapshot every active sub-request and abort it in
                # the engine so the upcoming `unpin` / `sleep` /
                # `cuda_checkpoint` runs against an empty scheduler.
                # Pending `generate_done` messages are deferred until
                # `resume` re-adds the requests via prefill.
                saved_count, aborted_count = _snapshot_active_into_saved()

                info["paused"] = True
                info["was_paused"] = was_paused
                info["saved"] = saved_count
                log.info("  pause: saved %d req_id(s) "
                         "(%d sub-requests aborted, was_paused=%s)",
                         saved_count, aborted_count, was_paused)

            elif cmd == "resume":
                if engine is None:
                    raise RuntimeError("resume requires init first")

                was_paused = _paused

                from vllm import SamplingParams
                from vllm.inputs import TokensPrompt

                restored = 0
                synthesized = 0
                for record in _saved_requests:
                    req_id = record["req_id"]
                    sp_dict = dict(record["sampling_params"] or {})
                    n_branch = sp_dict.get("n", 1)
                    if n_branch != 1:
                        raise RuntimeError(
                            f"resume with n={n_branch} not supported "
                            "(n=1 only)")
                    original_max = sp_dict.get("max_tokens")
                    eids_data = record["eids"]
                    prompts_orig = record["prompts"]

                    new_engine_ids = []
                    pre_pause_completion = []
                    pre_pause_text = []
                    original_prompt_tokens = []
                    all_finished_outputs = []

                    for i, eid_data in enumerate(eids_data):
                        prompt_tids = eid_data["prompt_token_ids"]
                        output_tids = eid_data["output_token_ids"]
                        output_text = eid_data["output_text"]
                        original_prompt_tokens.append(len(prompt_tids))
                        all_finished_outputs.append(output_text)

                        remaining = (original_max - len(output_tids)
                                     if original_max is not None else None)
                        if (remaining is not None and remaining <= 0):
                            # Already at max_tokens pre-pause; skip
                            # re-submission and synthesize the result.
                            continue

                        if prompt_tids:
                            full_token_ids = prompt_tids + list(output_tids)
                            prompt_obj = TokensPrompt(
                                prompt_token_ids=full_token_ids)
                        elif i < len(prompts_orig):
                            prompt_obj = prompts_orig[i]
                        else:
                            log.warning(
                                "  resume: req_id=%s eid#%d has no "
                                "prompt_token_ids and no original "
                                "prompt; skipping", req_id, i)
                            continue

                        sp_kwargs = dict(sp_dict)
                        if remaining is not None:
                            sp_kwargs["max_tokens"] = remaining
                        sp = SamplingParams(**sp_kwargs)

                        new_eid = _alloc_engine_id()
                        engine.add_request(new_eid, prompt_obj, sp)
                        _engine_to_req[new_eid] = req_id
                        new_engine_ids.append(new_eid)
                        pre_pause_completion.append(len(output_tids))
                        pre_pause_text.append(output_text)

                    if not new_engine_ids:
                        # Every branch was already finished pre-pause;
                        # emit a synthetic generate_done so the
                        # original waiter unblocks.
                        completion_tokens = sum(
                            len(d["output_token_ids"]) for d in eids_data)
                        prompt_tokens = sum(original_prompt_tokens)
                        synth_info = {
                            "req_id": req_id,
                            "outputs": [[t] for t in all_finished_outputs],
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "num_cached_tokens": 0,
                            "finish_reasons": ["length"],
                            "ttft_s": None,
                            "decode_s": None,
                            "tpot_ms": None,
                            "gen_tput_tok_s": 0.0,
                        }
                        pipe_conn.send(("generate_done", 0.0, None, synth_info))
                        synthesized += 1
                        log.info("  resume: req_id=%s synthesized "
                                 "(all branches at max_tokens pre-pause)",
                                 req_id)
                        continue

                    new_per_eid = {
                        new_eid: {"prompt_token_ids": None,
                                   "output_token_ids": [],
                                   "output_text": ""}
                        for new_eid in new_engine_ids
                    }
                    _active_reqs[req_id] = {
                        "t0": record["t0"],
                        "engine_ids": new_engine_ids,
                        "finished": {},
                        "prompts": list(prompts_orig),
                        "first_token_ts": record["first_token_ts"],
                        "sampling_params": dict(sp_dict),
                        "per_eid": new_per_eid,
                        "pre_pause_completion": pre_pause_completion,
                        "pre_pause_text": pre_pause_text,
                        "original_prompt_tokens": original_prompt_tokens,
                    }
                    restored += 1

                _saved_requests.clear()
                _paused = False

                info["paused"] = False
                info["was_paused"] = was_paused
                info["restored"] = restored
                info["synthesized"] = synthesized
                log.info("  resume: restored=%d synthesized=%d "
                         "(was_paused=%s)",
                         restored, synthesized, was_paused)

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
                    log.warning("  prepare_criu_dump: dist teardown error: %s", _e)

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
                            log.info("  prepare_criu_dump: store threads "
                                     "exited after %d polls", _attempt)
                            break
                        time.sleep(0.05)
                    else:
                        log.warning("  prepare_criu_dump: store threads "
                                    "still alive: %s", alive)

                pipe_fd = kwargs.get("pipe_fd", -1)
                # stdout/stderr were already pointed at the per-instance
                # log file by redirect_stdio_to_instance_file at process
                # startup, so CRIU dumps them as regular files pointing
                # to that path -- restoring re-opens the same path and
                # the restored child keeps logging there.

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
                log.info("  prepare_criu_dump: fds=%s, unmapped=%s, "
                         "destroyed_pg=%s, remaining_threads=%s",
                         closed_fds, unmapped, destroyed_pg,
                         remaining_threads)

            elif cmd == "rebind_log":
                # Sent by the worker right after CRIU restore when the
                # current instance_id differs from the one baked into the
                # image: re-dup2 stdout/stderr onto /tmp/inst{new_id}.log
                # and rebuild the log adapter so subsequent records carry
                # the correct i{N} scope.
                new_id = kwargs["instance_id"]
                _child_log_path = semip_logging.redirect_stdio_to_instance_file(
                    new_id)
                log = semip_logging.child(new_id, rank)
                info["instance_id"] = new_id
                info["path"] = _child_log_path

            elif cmd == "save_weights":
                if pinned_buf is None or index is None:
                    raise RuntimeError(
                        "save_weights requires attach+stage first")

                weights_dir = kwargs["weights_dir"]
                shard_bytes = int(kwargs.get("shard_bytes")
                                  or _WEIGHTS_SHARD_BYTES)
                workers_req = int(kwargs.get("io_workers")
                                  or _WEIGHTS_IO_WORKERS)
                total = pinned_buf.numel()
                mv = memoryview(pinned_buf.numpy())

                # Recreate the dir clean so stale shards from a prior
                # (possibly larger) model can't linger.  Aborted runs may
                # leave root-owned files -> fall back to `sudo rm -rf`
                # (mirrors the criu_dump dir-clear in worker.py).
                if os.path.exists(weights_dir):
                    try:
                        shutil.rmtree(weights_dir)
                    except PermissionError:
                        subprocess.run(["sudo", "rm", "-rf", weights_dir],
                                       check=True)
                os.makedirs(weights_dir, exist_ok=True)

                ranges = []
                _lo = 0
                _i = 0
                while _lo < total:
                    _hi = min(_lo + shard_bytes, total)
                    ranges.append((_i, _lo, _hi))
                    _lo = _hi
                    _i += 1
                workers = max(1, min(workers_req, len(ranges)))

                def _write_shard(i, lo, hi):
                    fd = os.open(
                        os.path.join(weights_dir, f"shard_{i:04d}.bin"),
                        os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
                    try:
                        pos = lo
                        while pos < hi:            # os.write may short-write
                            pos += os.write(fd, mv[pos:hi])
                        os.fsync(fd)
                    finally:
                        os.close(fd)
                    return hi - lo

                t0 = time.perf_counter()
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    for fu in [ex.submit(_write_shard, *r) for r in ranges]:
                        fu.result()               # re-raise worker errors
                dt = time.perf_counter() - t0

                model_name = getattr(
                    getattr(engine, "model_config", None), "model", None)
                manifest = {
                    "total_bytes": total,
                    "n_params": len(index),
                    "model": model_name,
                    "shard_bytes": shard_bytes,
                    "shards": [{"name": f"shard_{i:04d}.bin",
                                "offset": lo, "nbytes": hi - lo}
                               for (i, lo, hi) in ranges],
                    "layout": [[name, off, nbytes, str(dtype), list(shape)]
                               for name, (off, nbytes, dtype, shape)
                               in index.items()],
                }
                with open(os.path.join(weights_dir, "weights_meta.json"),
                          "w") as f:
                    json.dump(manifest, f)

                info["weights_dir"] = weights_dir
                info["bytes"] = total
                info["n_shards"] = len(ranges)
                log.info("  saved weights: %d shard(s), %.2f GiB in %.2fs "
                         "(%.2f GiB/s)", len(ranges), total / 2**30, dt,
                         (total / 2**30) / dt if dt > 0 else 0.0)

            elif cmd == "load_weights":
                if pinned_buf is None or index is None:
                    raise RuntimeError("load_weights requires attach first")

                weights_dir = kwargs["weights_dir"]
                meta_path = os.path.join(weights_dir, "weights_meta.json")
                if not os.path.isfile(meta_path):
                    raise RuntimeError(
                        f"no weights manifest at {meta_path}; "
                        f"run save_weights first")
                with open(meta_path) as f:
                    manifest = json.load(f)

                total = pinned_buf.numel()
                if int(manifest["total_bytes"]) != total:
                    raise RuntimeError(
                        f"weights size mismatch: manifest "
                        f"{manifest['total_bytes']}B but attached buffer "
                        f"{total}B (config/model changed?)")

                # Strict layout check against the freshly-attached index:
                # (name, offset, nbytes) must line up exactly.
                cur_layout = [[name, off, nbytes]
                              for name, (off, nbytes, dtype, shape)
                              in index.items()]
                man_layout = [[row[0], row[1], row[2]]
                              for row in manifest.get("layout", [])]
                if man_layout and man_layout != cur_layout:
                    raise RuntimeError(
                        "weights layout mismatch between manifest and "
                        "attached model (param order/sizes differ)")

                mv = memoryview(pinned_buf.numpy())
                shards = manifest["shards"]
                workers = max(1, min(int(kwargs.get("io_workers")
                                         or _WEIGHTS_IO_WORKERS), len(shards)))

                def _read_shard(s):
                    lo = int(s["offset"])
                    n = int(s["nbytes"])
                    dst = mv[lo:lo + n]
                    got = 0
                    with open(os.path.join(weights_dir, s["name"]),
                              "rb", buffering=0) as f:
                        while got < n:             # readinto may short-read
                            r = f.readinto(dst[got:])
                            if r == 0:
                                break
                            got += r
                    if got != n:
                        raise RuntimeError(
                            f"{s['name']}: read {got} != {n}")
                    return n

                t0 = time.perf_counter()
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    for fu in [ex.submit(_read_shard, s) for s in shards]:
                        fu.result()               # re-raise worker errors
                dt = time.perf_counter() - t0

                info["bytes"] = total
                info["n_shards"] = len(shards)
                log.info("  loaded weights: %d shard(s), %.2f GiB in %.2fs "
                         "(%.2f GiB/s)", len(shards), total / 2**30, dt,
                         (total / 2**30) / dt if dt > 0 else 0.0)

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
    # of /dev/null) so that any traceback/log from a restored child is
    # still captured for post-mortem debugging.
    _stdout_fixed = False

    # The current in-flight command, captured outside the per-iteration
    # scope so the fatal-error reporter below can blame the right cmd.
    cmd = None

    try:
        while True:
            if engine is None and llm is not None:
                engine = llm.llm_engine

            has_active = (engine is not None
                          and engine.has_unfinished_requests()
                          and not _paused)

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
                    semip_logging.rebind_stdout()
                    log.info("stdout/stderr redirected to %s after CRIU restore",
                             _child_log_path)

            if cmd == "exit":
                _drain_engine()
                log.info("exit")
                pinned_buf = None
                pipe_conn.send("exit_ack")
                break

            log.info(">>> %s", cmd)

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
                        log.info(">>> %s (deferred)", cmd2)
                        _deferred_cmds.append((cmd2, kwargs2))
                continue

            t0 = time.perf_counter()
            error, info = _handle_command(cmd, kwargs)
            elapsed = time.perf_counter() - t0
            status = "OK" if error is None else "FAILED"
            log.info("<<< %s %s (%.3fs)", cmd, status, elapsed)
            pipe_conn.send((cmd, elapsed, error, info))

    except BaseException as _fatal:
        # Last-resort reporter: any unhandled exception in the main loop
        # (including KeyboardInterrupt, SystemExit) gets a final error
        # frame on the pipe so the worker can attribute the failure to a
        # specific cmd instead of just seeing "child pipe broken".  Both
        # the traceback and the offending cmd are logged to the per-rank
        # log file via log so the post-mortem survives a CRIU restore.
        import traceback as _tb
        _trace = _tb.format_exc()
        log.error("FATAL in main loop (cmd=%s): %s: %s",
                  cmd, type(_fatal).__name__, _fatal)
        log.error("%s", _trace)
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
