"""Lightweight orchestrator for managing multiple vLLM instances by model ID.

State machine with a single ordered ladder:

    saved  <-->  checkpoint  <-->  sleep  <-->  up  -->  running (transient)

move(model_id, target) walks up or down the ladder.
generate() auto-transitions to 'up', then up -> running -> up.
remove() auto-transitions to 'saved', then deletes.
"""
from __future__ import annotations

import collections
import json
import math
import os
import shutil
import subprocess
import threading
import time

import pynvml
from contextlib import contextmanager
from concurrent.futures import Future, ThreadPoolExecutor
from http.server import HTTPServer
from typing import Any

from instance import Instance
from slots import Slot, Slots
from state_server import init_t0, start_state_server

import sys as _sys

try:
    _terminal = os.fdopen(os.dup(_sys.stderr.fileno()), "w")
except (OSError, AttributeError):
    _terminal = None

_console_stream = None


def _console(msg: str) -> None:
    out = _console_stream or _terminal or _sys.stderr
    out.write(msg + "\n")
    out.flush()


def _discover_gpu_ids() -> list[int]:
    """Return GPU device indices via NVML without initializing CUDA."""
    pynvml.nvmlInit()
    return list(range(pynvml.nvmlDeviceGetCount()))


_STATES = ["saved", "checkpoint", "sleep", "up"]


def _pick_level(gpu_memory_utilization: float) -> int:
    """Pick the smallest slot *level* (largest GPU fraction) that satisfies *util*.

    A level-L slot covers ``1 / 2**(L-1)`` of a GPU.  We pick the largest
    such fraction that still fits the model's ``gpu_memory_utilization``,
    which is the smallest L with ``1 / 2**(L-1) >= util``:

        util in (0.5,   1.0] -> L1   (whole GPU)
        util in (0.25,  0.5] -> L2   (half GPU)
        util in (0.125, 0.25]-> L3   (quarter GPU)
        util in (0.0625,0.125]->L4   (eighth GPU)
        ...

    Boundary points fall into the lower-L (larger-fraction) bucket, e.g.
    ``util == 0.5`` -> ``L1``.  Degenerate / non-positive ``util`` returns
    ``L1`` (whole GPU).
    """
    if gpu_memory_utilization <= 0.0 or gpu_memory_utilization >= 1.0:
        return 1
    return max(1, int(math.ceil(math.log2(1.0 / gpu_memory_utilization))))


class Orchestrator:
    """Static registry that maps human-readable model IDs to Instances.

    States (ordered ladder):
        saved      -- image on disk, no process, no GPU
        checkpoint -- image on disk + live CRIU process in memory, no GPU
        sleep      -- CUDA context restored on a GPU; slot held until
                      `running -> up` releases it, or `sleep -> checkpoint`
                      tears the context down
        up         -- image on disk + live process + GPU held + weights on GPU
        running    -- transient sub-state of 'up' during generate();
                      always holds a slot, therefore unevictable
    """

    _registry: dict[str, Any] = {}
    _futures: dict[str, Future] = {}
    _gpu_ids: list[int] = []
    _image_cache: str | None = None
    _pool: ThreadPoolExecutor | None = None
    _state_server: HTTPServer | None = None

    _request_log: list[dict] = []
    _request_counter: int = 0
    _request_lock: threading.Lock = threading.Lock()

    _generate_queue: collections.deque = collections.deque()
    _generate_futures: list[Future] = []
    _last_generate_future: dict[str, Future] = {}
    _generate_locks: dict[str, threading.Lock] = {}
    _inflight: dict[str, list] = {}  # model_id -> [(req_id, req_record, Event)]
    _waiter_active: dict[str, bool] = {}

    @staticmethod
    def _get_generate_lock(model_id: str) -> threading.Lock:
        lock = Orchestrator._generate_locks.get(model_id)
        if lock is None:
            lock = threading.Lock()
            Orchestrator._generate_locks[model_id] = lock
        return lock

    @staticmethod
    def _print_states() -> None:
        """Print a compact one-line summary of all model states."""
        parts = []
        for mid, entry in Orchestrator._registry.items():
            s = entry["state"]
            gpu = entry.get("gpu")
            tag = f"{mid}[{s}:gpu{gpu}]" if gpu is not None else f"{mid}[{s}]"
            parts.append(tag)
        _console("  models: " + "  ".join(parts) if parts else "  models: (none)")

    @staticmethod
    def _set_state(model_id: str, state: str) -> None:
        """Set a model's state and print the system-wide state summary."""
        now = time.perf_counter()
        prev_state = Orchestrator._registry[model_id].get("state")
        Orchestrator._registry[model_id]["state"] = state
        # ``wait`` is a transient sub-state of ``checkpoint`` that
        # publishes only while ``Slots.allocate`` is blocked.  The
        # ``checkpoint -> wait`` flip resets the timer so the user can
        # see how long the slot wait has been queued, but the reverse
        # ``wait -> checkpoint`` leg returns to the same idle CPU tier
        # we were already in -- preserve the original ``state_since``
        # there so the CPU row's age keeps climbing instead of bouncing
        # back to 0 once a slot becomes available.
        if not (prev_state == "wait" and state == "checkpoint"):
            Orchestrator._registry[model_id]["state_since"] = now
        # Mirror this transition into any in-flight request bound to
        # *model_id* so the dashboard can render a precise per-request
        # state timeline.  Deduped against the trailing entry.
        try:
            with Orchestrator._request_lock:
                for rec in Orchestrator._request_log:
                    if (rec.get("model_id") == model_id
                            and rec.get("t_done") is None):
                        log = rec.setdefault("state_log", [])
                        if not log or log[-1][1] != state:
                            log.append((now, state))
        except Exception:
            # State tracking is observational; never let it block a
            # real state transition.
            pass
        Orchestrator._print_states()

    _timing = threading.local()

    @staticmethod
    @contextmanager
    def _locks_ordered(*model_ids: str):
        """Acquire per-model locks in sorted *model_id* order (deadlock-safe for pairs)."""
        ids = sorted(set(model_ids))
        if not ids:
            yield
            return
        if len(ids) == 1:
            with Orchestrator._registry[ids[0]]["_lock"]:
                yield
            return
        if len(ids) == 2:
            with Orchestrator._registry[ids[0]]["_lock"]:
                with Orchestrator._registry[ids[1]]["_lock"]:
                    yield
            return
        raise RuntimeError("at most two distinct model locks per _locks_ordered call")

    @staticmethod
    def _image_dir_for(model_id: str) -> str:
        """Return the image-cache directory for a given model_id."""
        safe_name = model_id.replace("/", "--")
        return os.path.join(Orchestrator._image_cache, safe_name)

    # ------------------------------------------------------------------
    # init
    # ------------------------------------------------------------------

    @staticmethod
    def init(image_cache: str = "/data-fast/image-cache",
             gpus: list[int] | None = None,
             dashboard_port: int = 8157) -> None:
        """Discover GPUs, scan image cache, and populate registry.

        Each subdirectory of *image_cache* that contains a ``meta.json``
        is registered in **saved** state (image on disk, no live process).

        *dashboard_port* starts an HTTP server for the dashboard on that
        port.  Set to 0 to disable.
        """
        Orchestrator._gpu_ids = gpus if gpus is not None else _discover_gpu_ids()
        if Slots._inited:
            # Hard-reset: bypass leak assertions (tests may re-init).
            with Slots._cv:
                Slots._pools.clear()
                Slots._live.clear()
                Slots._last_used.clear()
                Slots._waiters.clear()
                Slots._inited = False
        Slots.init(Orchestrator._gpu_ids)
        Orchestrator._image_cache = image_cache
        os.makedirs(image_cache, exist_ok=True)

        import glob as _glob
        for _pattern in [
            "/tmp/torchinductor_root/**/*.ghost",
            "/tmp/__triton_launcher.*.ghost",
            "/dev/shm/*.ghost",
            "/dev/shm/link_remap.*",
        ]:
            for _f in _glob.glob(_pattern, recursive=True):
                try:
                    os.remove(_f)
                except OSError:
                    subprocess.run(["sudo", "rm", "-f", _f], capture_output=True)

        Orchestrator._pool = ThreadPoolExecutor()

        Orchestrator._registry = {}
        Orchestrator._futures = {}
        Orchestrator._generate_queue = collections.deque()
        Orchestrator._generate_futures = []
        Orchestrator._last_generate_future = {}
        Orchestrator._generate_locks = {}
        Orchestrator._inflight = {}
        Orchestrator._waiter_active = {}
        with Orchestrator._request_lock:
            Orchestrator._request_log = []
            Orchestrator._request_counter = 0

        for entry_name in sorted(os.listdir(image_cache)):
            image_dir = os.path.join(image_cache, entry_name)
            meta_path = os.path.join(image_dir, "meta.json")
            if not os.path.isfile(meta_path):
                continue
            with open(meta_path) as f:
                meta = json.load(f)
            vllm_config = meta.get("vllm_config", {})
            model_id = entry_name
            run_level = _pick_level(
                vllm_config.get("gpu_memory_utilization", 0.7))
            Orchestrator._registry[model_id] = {
                "state": "saved",
                "instance": None,
                "gpu": None,
                "slot": None,
                "level": run_level,
                "vllm_config": vllm_config,
                "image_dir": image_dir,
                "pinned_cpu_bytes": meta.get(
                    "pinned_cpu_bytes", meta.get("pinned_bytes", 0)),
                "total_gpu_bytes": meta.get("total_gpu_bytes", 0),
                "_lock": threading.RLock(),
            }
            pinned = Orchestrator._registry[model_id]["pinned_cpu_bytes"]
            print(f"[orchestrator] discovered {model_id}  "
                  f"model={vllm_config.get('model', '?')}  "
                  f"pinned={pinned / 2**30:.1f} GiB  image={image_dir}")

        if Orchestrator._state_server is not None:
            try:
                Orchestrator._state_server.shutdown()
            except Exception:
                pass
            Orchestrator._state_server = None

        if dashboard_port:
            Orchestrator._state_server = start_state_server(dashboard_port)
            print(f"[orchestrator] dashboard server on port {dashboard_port}")

        n = len(Orchestrator._registry)
        print(f"[orchestrator] init  image_cache={image_cache}  "
              f"gpus={Orchestrator._gpu_ids}  models={n}")
        _console(f"\nOrchestrator  image_cache={image_cache}  "
                 f"gpus={Orchestrator._gpu_ids}  discovered {n} saved models")

        # Anchor the dashboard's relative-time clock here so per-state
        # ages render even before the first generate (phases that only
        # register / move never call generate, but still need timers).
        init_t0()

    # ------------------------------------------------------------------
    # register
    # ------------------------------------------------------------------

    @staticmethod
    def register(model_id: str, vllm_config: dict | str) -> None:
        """Cold-start a new model, save its image, and register it.

        *vllm_config* can be a dict or a plain model name string
        (e.g. ``"Qwen/Qwen3-32B"``), which is shorthand for
        ``{"model": "Qwen/Qwen3-32B"}``.

        The dump is destructive — the child process is killed after the
        image is written.  The model ends up in **saved** state (image on
        disk, no live process).
        """
        if isinstance(vllm_config, str):
            vllm_config = {"model": vllm_config}
        vllm_config.setdefault("gpu_memory_utilization", 0.7)
        if model_id in Orchestrator._registry:
            print(f"[orchestrator] register  model_id={model_id}  "
                  f"already registered – skipping")
            return
        _console(f"{model_id}: register received")
        print(f"[orchestrator] register  model_id={model_id}  "
              f"model={vllm_config.get('model', '?')}")

        fut = Orchestrator._pool.submit(
            Orchestrator._register_sync, model_id, vllm_config,
        )
        Orchestrator._futures[model_id] = fut

    @staticmethod
    def _register_sync(model_id: str, vllm_config: dict) -> None:
        t0 = time.perf_counter()
        image_dir = Orchestrator._image_dir_for(model_id)
        inst = Instance(vllm_config)

        # Cold-start always uses a full GPU; the runtime level (used at
        # `checkpoint -> sleep` later) is computed from the model's
        # configured `gpu_memory_utilization`.
        register_slot = Slots.allocate(level=1)
        gpu = register_slot.gpu_id
        t_acquired = time.perf_counter()

        run_level = _pick_level(
            vllm_config.get("gpu_memory_utilization", 0.7))
        Orchestrator._registry[model_id] = {
            "state": "init",
            "instance": inst,
            "gpu": gpu,
            "slot": None,
            "level": run_level,
            "vllm_config": vllm_config,
            "image_dir": image_dir,
            "pinned_cpu_bytes": 0,
            "total_gpu_bytes": 0,
            "state_since": time.perf_counter(),
            "_lock": threading.RLock(),
        }
        Orchestrator._print_states()

        with Orchestrator._locks_ordered(model_id):
            inst.init(gpu).attach().repin().stage().unpin().sleep().checkpoint_cuda().wait()
            inst.save_image(image_dir).wait()
            print(f"[orchestrator] {model_id}: image saved to {image_dir}")

            inst._send("exit")
            inst._reset()

            # The cold-start slot is local to this function and never
            # flowed into entry["slot"]; release it inline.
            Slots.deallocate(register_slot)
            t_done = time.perf_counter()
            t_wait = t_acquired - t0
            t_exec = t_done - t_acquired
            print(f"[orchestrator] {model_id}: registered on GPU {gpu}  "
                  f"({t_done - t0:.1f}s)")
            _console(f"{model_id}: registered "
                     f"(wait={t_wait:.1f}s, cold-start={t_exec:.1f}s, "
                     f"total={t_wait + t_exec:.1f}s)")

            entry = Orchestrator._registry[model_id]
            entry["instance"] = None
            entry["gpu"] = None
            entry["pinned_cpu_bytes"] = inst.pinned_cpu_bytes
            entry["total_gpu_bytes"] = inst.total_gpu_bytes
            Orchestrator._set_state(model_id, "saved")

    # ------------------------------------------------------------------
    # move  (walk the state ladder)
    # ------------------------------------------------------------------

    @staticmethod
    def move_all(target: str, target_gpu: int | None = None) -> None:
        """Fan out :meth:`move` to every registered model.

        Mirrors the no-model_id flavours of :meth:`wait` / :meth:`remove`.
        Each per-model move is submitted to the thread pool independently,
        so they execute concurrently; call :meth:`wait` to join.
        """
        for mid in list(Orchestrator._registry):
            Orchestrator.move(mid, target, target_gpu=target_gpu)

    @staticmethod
    def move(model_id: str, target: str, target_gpu: int | None = None) -> None:
        """Move *model_id* to *target* state by walking the ladder.

        Valid targets: ``"saved"``, ``"checkpoint"``, ``"sleep"``, ``"up"``.
        Submits the transition to the thread pool (non-blocking).

        *target_gpu* (optional) is only valid when *target* is ``"sleep"``
        and pins the model onto that specific GPU.  This is a **slotless
        sleep** flavour: no slot is allocated and any slot the model
        currently holds is released.  The model parks on ``target_gpu``
        without competing for slot resources; on the next ``sleep -> up``
        it goes through the standard tier-A/B/C acquisition logic (which
        may stay on ``target_gpu`` or migrate elsewhere depending on what
        is free).
        """
        if target not in _STATES:
            raise ValueError(f"invalid target state '{target}'; "
                             f"must be one of {_STATES}")
        if target_gpu is not None and target != "sleep":
            raise ValueError("target_gpu may only be specified when target is 'sleep'")
        entry = Orchestrator._registry.get(model_id)
        if entry is None:
            print(f"[orchestrator] WARNING: move({model_id!r}, {target!r}) "
                  f"skipped – model not registered")
            _console(f"WARNING: {model_id} not registered, skipping move")
            return
        _console(f"{model_id}: move -> {target}")
        print(f"[orchestrator] move  model_id={model_id}  target={target}"
              + (f"  target_gpu={target_gpu}" if target_gpu is not None else ""))

        prev = Orchestrator._futures.get(model_id)
        prev_gen = Orchestrator._last_generate_future.pop(model_id, None)
        fut = Orchestrator._pool.submit(
            Orchestrator._move_sync, model_id, target, prev, target_gpu,
            prev_gen_future=prev_gen,
        )
        Orchestrator._futures[model_id] = fut

    @staticmethod
    def _move_sync(model_id: str, target: str, prev_future=None,
                   target_gpu: int | None = None,
                   announce_state: str | None = None,
                   prev_gen_future: Future | None = None) -> None:
        if prev_future is not None:
            prev_future.result()
        if prev_gen_future is not None:
            prev_gen_future.result()

        entry = Orchestrator._registry[model_id]
        current = entry["state"]
        if current == "running":
            if announce_state == "running":
                return
            raise RuntimeError(
                f"model '{model_id}' is currently running a generate; "
                f"wait for it to finish before calling move()")
        if current == target and target_gpu is None and announce_state is None:
            _console(f"{model_id}: already in '{target}' state")
            return
        if current == target and announce_state is not None:
            # Sub-state announce (typically up -> running): preserve the
            # invariant that ``running`` always holds a slot.  A slotless
            # ``up`` must acquire one first.
            if (target == "up" and announce_state == "running"
                    and entry.get("slot") is None):
                level = entry["level"]
                home_gpu = entry["gpu"]
                # Tier A: home GPU is free right now -> claim the slot
                # in place; weights stay in HBM.
                slot = Slots.try_allocate(level=level, gpu=home_gpu)
                if slot is not None:
                    with Orchestrator._locks_ordered(model_id):
                        entry["slot"] = slot
                    _console(f"{model_id}: claimed slot on GPU {home_gpu}")
                    print(f"[orchestrator] {model_id}: claimed slot "
                          f"on GPU {home_gpu}")
                else:
                    # Need migration / FIFO: retreat to sleep (frees
                    # HBM) and climb back up via the standard
                    # ``sleep -> up`` tier-A/B/C path, which handles
                    # migration and FIFO blocking + Phase-2 eviction.
                    _console(f"{model_id}: no slot on GPU {home_gpu}, "
                             f"retreating to sleep")
                    print(f"[orchestrator] {model_id}: retreating to "
                          f"sleep to acquire slot")
                    Orchestrator._step_down(model_id, "up", "sleep")
                    Orchestrator._step_up(model_id, "sleep", "up")
            with Orchestrator._locks_ordered(model_id):
                if entry["state"] == target:
                    Orchestrator._set_state(model_id, announce_state)
                    return
            current = entry["state"]

        cur_idx = _STATES.index(current)
        tgt_idx = _STATES.index(target)

        t0 = time.perf_counter()

        if (target == "sleep" and target_gpu is not None
                and current in ("up", "sleep")
                and entry.get("gpu") != target_gpu):
            # Migrate to a specific GPU: walk down to checkpoint, then
            # back up to (slotless) sleep with target_gpu pinned.
            chk_idx = _STATES.index("checkpoint")
            for step in range(cur_idx, chk_idx, -1):
                Orchestrator._step_down(
                    model_id, _STATES[step], _STATES[step - 1])
            Orchestrator._step_up(model_id, "checkpoint", "sleep",
                                  target_gpu=target_gpu)
        elif (target == "sleep" and target_gpu is not None
                and current == "sleep" and entry.get("gpu") == target_gpu):
            # Already on target_gpu in sleep; fall through so the tail
            # below releases any slot still held.
            pass
        elif cur_idx < tgt_idx:
            for step in range(cur_idx, tgt_idx):
                cur = _STATES[step]
                nxt = _STATES[step + 1]
                kw = {}
                if cur == "checkpoint" and nxt == "sleep":
                    kw["target_gpu"] = target_gpu
                if nxt == "up" and announce_state is not None:
                    kw["announce_state"] = announce_state
                Orchestrator._step_up(model_id, cur, nxt, **kw)
        else:
            for step in range(cur_idx, tgt_idx, -1):
                Orchestrator._step_down(model_id, _STATES[step], _STATES[step - 1])

        # Slotless-sleep flavour tail: when the user designates target_gpu
        # for a sleep target, ensure the model holds no slot.  Covers the
        # cases where the ladder walk left a slot in place (e.g. up -> sleep
        # on the same GPU, or already-sleeping-and-slotted on target_gpu).
        if target == "sleep" and target_gpu is not None:
            with Orchestrator._locks_ordered(model_id):
                if entry.get("slot") is not None:
                    Slots.deallocate(entry["slot"])
                    entry["slot"] = None
                    _console(f"{model_id}: released slot, "
                             f"slotless on GPU {target_gpu}")
                    print(f"[orchestrator] {model_id}: released slot "
                          f"on GPU {target_gpu} (slotless)")

        elapsed = time.perf_counter() - t0

        print(f"[orchestrator] {model_id}: {current} -> {target}  ({elapsed:.1f}s)")
        _console(f"{model_id}: {current} -> {target} ({elapsed:.1f}s)")

    @staticmethod
    def _step_up(model_id: str, from_state: str, to_state: str,
                 *, target_gpu: int | None = None,
                 announce_state: str | None = None) -> None:
        """Execute one upward step on the ladder.

        *announce_state* overrides the published state when the step
        completes.  Used by generate to atomically go sleep -> running
        (skipping the observable 'up' window).
        """
        entry = Orchestrator._registry[model_id]

        if from_state == "saved" and to_state == "checkpoint":
            with Orchestrator._locks_ordered(model_id):
                inst = Instance(entry["vllm_config"])
                # load() reads meta.json and hydrates total_gpu_bytes /
                # pinned_cpu_bytes on the instance; plan_load_weights
                # uses those to build the chunk plan in the child once.
                inst.load_image(entry["image_dir"]).plan_load_weights().wait()
                # Mirror onto the registry entry to keep registry-as-truth
                # even though Instance.load_image already set self.* from meta.
                inst.pinned_cpu_bytes = entry.get("pinned_cpu_bytes", 0)
                inst.total_gpu_bytes = entry.get("total_gpu_bytes", 0)
                entry["instance"] = inst
                Orchestrator._set_state(model_id, "checkpoint")
            return

        if from_state == "checkpoint" and to_state == "sleep":
            if target_gpu is None:
                # Publish a transient "wait" state only if Slots.allocate
                # actually has to block; otherwise the model races
                # checkpoint -> sleep without ever observably entering
                # "wait", which keeps zero-duration ``wait``/``ckpt``
                # entries out of every request's state_log.
                published_wait = False

                def _on_block() -> None:
                    nonlocal published_wait
                    published_wait = True
                    Orchestrator._set_state(model_id, "wait")

                t_wait = time.perf_counter()
                slot = Slots.allocate(level=entry["level"],
                                      on_block=_on_block)
                gpu = slot.gpu_id
                Orchestrator._timing.gpu_wait_s = time.perf_counter() - t_wait
                if published_wait:
                    Orchestrator._set_state(model_id, "checkpoint")
            else:
                # Slotless-sleep flavour: user pinned a target GPU, so
                # park the model there without consuming a slot.
                slot = None
                gpu = target_gpu
                Orchestrator._timing.gpu_wait_s = 0.0
            with Orchestrator._locks_ordered(model_id):
                entry["slot"] = slot
                entry["gpu"] = gpu
                tag = "" if slot is not None else " (slotless)"
                _console(f"{model_id}: placed on GPU {gpu}{tag}")
                print(f"[orchestrator] {model_id}: placed on GPU {gpu}{tag}")
                entry["instance"].restore_cuda(gpu).repin().wait()
                Orchestrator._set_state(model_id, "sleep")
            return

        if from_state == "sleep" and to_state == "up":
            inst = entry["instance"]
            level = entry["level"]
            home_gpu = entry["gpu"]

            # Phase 1: ensure we hold a slot.
            if entry["slot"] is None:
                # Tier A: home GPU is free right now.
                slot = Slots.try_allocate(level=level, gpu=home_gpu)
                if slot is None:
                    # Tier B: any GPU free right now (possibly elsewhere).
                    slot = Slots.try_allocate(level=level)
                if slot is not None:
                    # Commit ownership before any (potentially slow)
                    # migration so the dashboard sees the slot held
                    # throughout instead of a slotless ``sleep``
                    # interlude on the wrong GPU.
                    with Orchestrator._locks_ordered(model_id):
                        entry["slot"] = slot
                    if slot.gpu_id != home_gpu:
                        t_mig = time.perf_counter()
                        _console(f"{model_id}: migrating from GPU {home_gpu} "
                                 f"to GPU {slot.gpu_id}")
                        print(f"[orchestrator] {model_id}: migrating "
                              f"GPU {home_gpu} -> GPU {slot.gpu_id}")
                        with Orchestrator._locks_ordered(model_id):
                            # Mirror the brief checkpoint pass-through in
                            # the published state so dashboards see the
                            # sleep -> checkpoint -> sleep transition.
                            inst.unpin().checkpoint_cuda().wait()
                            entry["gpu"] = None
                            Orchestrator._set_state(model_id, "checkpoint")
                            inst.restore_cuda(slot.gpu_id).repin().wait()
                            entry["gpu"] = slot.gpu_id
                            Orchestrator._set_state(model_id, "sleep")
                        Orchestrator._timing.migrate_s = (
                            time.perf_counter() - t_mig)
                else:
                    # Tier C: nothing free -> retreat and FIFO.
                    _console(f"{model_id}: waiting for slot...")
                    print(f"[orchestrator] {model_id}: waiting for slot ...")
                    t_wait = time.perf_counter()
                    Orchestrator._step_down(model_id, "sleep", "checkpoint")
                    Orchestrator._step_up(model_id, "checkpoint", "sleep")
                    Orchestrator._timing.gpu_wait_s = (
                        time.perf_counter() - t_wait)
                    slot = entry["slot"]
                entry["gpu"] = slot.gpu_id
                home_gpu = slot.gpu_id

            # Phase 2: free enough HBM on home_gpu for this model's
            # wake-up by evicting the oldest slotless `up` incumbents,
            # but no more than necessary.
            #
            # HBM accounting (each level-L share == 1 / 2**(L-1) of the
            # GPU):
            #
            #   slotted_others = sum of slot shares for every OTHER
            #                    slotted model on home_gpu, regardless
            #                    of state.  We use the slot share, not
            #                    the live state, because slot allocation
            #                    is what serialises wake-ups: a slotted
            #                    `sleep` model whose Phase 3 is already
            #                    in flight on the worker (queued behind
            #                    ours) will be HBM-resident by the time
            #                    our Phase 3 finishes, even though its
            #                    registry state still reads "sleep".
            #   new_share      = the slot we just took for this model;
            #                    it will be resident at end of Phase 3.
            #   slack          = 1.0 - slotted_others - new_share
            #                  = HBM the buddy allocator hasn't handed
            #                    out, available for slotless squatters.
            #
            # The buddy allocator already guarantees the sum of all
            # slot shares on home_gpu is <= 1.0, so slack is >= 0.
            # Slotless `up` squatters consume HBM but no slot, so they
            # must fit inside `slack`.  Evict oldest-first until the
            # remaining slotless share fits.
            #
            # Example (the case originally raised): an L2 wakes up on
            # a GPU with two slotless L3 squatters (each 0.25), one
            # slotless L2 squatter (0.5), and no other slotted models.
            # new_share=0.5, slotted_others=0, slack=0.5.  Slotless
            # total = 1.0; eviction stops as soon as the remaining
            # slotless share is <= 0.5.  If the two L3s are oldest,
            # both get evicted (frees 0.5) and the L2 squatter stays.
            # Use ``slot.gpu_id`` as the source of truth for slotted
            # residency on ``home_gpu`` -- it matches ``Slots._live``
            # exactly.  ``entry["gpu"]`` is *not* reliable here: during a
            # peer's migration its slot flips to the new GPU at line 624
            # well before ``entry["gpu"]`` is updated (None during the
            # checkpoint pass-through, then the new GPU after restore).
            # A scan keyed on ``entry["gpu"] == home_gpu`` would therefore
            # double-count peers whose slot has already moved away,
            # driving ``slack`` negative and either tripping the assert
            # below or starving Phase 3 of room.  Slotless residents
            # don't migrate their slot (they have none), so falling back
            # to ``entry["gpu"]`` for them is correct.
            new_share = 1.0 / (1 << (level - 1))
            slotted_others = 0.0
            slotless_cands: list[tuple[float, str, float]] = []
            for mid, e in Orchestrator._registry.items():
                if mid == model_id:
                    continue
                s = e.get("slot")
                if s is not None:
                    if s.gpu_id != home_gpu:
                        continue
                    slotted_others += 1.0 / (1 << (s.level - 1))
                elif (e.get("gpu") == home_gpu
                        and e.get("state") == "up"):
                    e_share = 1.0 / (1 << (e["level"] - 1))
                    slotless_cands.append(
                        (e.get("state_since", 0.0), mid, e_share))
            slack = 1.0 - slotted_others - new_share
            assert slack >= -1e-9, (
                f"HBM over-subscribed on GPU {home_gpu}: "
                f"slotted_others={slotted_others}, new_share={new_share}")
            slotless_cands.sort()
            remaining = sum(share for _, _, share in slotless_cands)
            for _, incumbent, share in slotless_cands:
                if remaining <= slack + 1e-9:
                    break
                # Re-validate the candidate under its own lock before
                # touching its instance.  The Phase 2 scan above is
                # lock-free, so an incumbent we picked may have
                # self-evacuated in the meantime (its own thread retreated
                # from `up` to acquire a slot, leaving the child process
                # CRIU-checkpointed).  Queuing a `sleep` on a checkpointed
                # child raises "child pipe broken".
                #
                # Holding the incumbent's RLock across `_step_down` is
                # safe because `_step_down` re-enters the same lock.
                # Either branch decrements `remaining`: if we evicted, we
                # freed `share`; if the incumbent self-evacuated, it
                # already freed `share` on its own.
                inc_entry = Orchestrator._registry[incumbent]
                with Orchestrator._locks_ordered(incumbent):
                    if (inc_entry.get("slot") is None
                            and inc_entry.get("state") == "up"
                            and inc_entry.get("gpu") == home_gpu):
                        _console(f"{model_id}: evicting {incumbent} "
                                 f"from GPU {home_gpu}")
                        print(f"[orchestrator] {model_id}: evicting "
                              f"{incumbent} from GPU {home_gpu}")
                        Orchestrator._step_down(incumbent, "up", "sleep")
                remaining -= share

            # Phase 3: weights to HBM, announce up/running.
            with Orchestrator._locks_ordered(model_id):
                inst.wake_up_weights().load_weights().wake_up_kv_cache().wait()
                Orchestrator._set_state(model_id, announce_state or "up")
            return

        raise AssertionError(
            f"_step_up unexpected transition {from_state} -> {to_state}")

    @staticmethod
    def _step_down(model_id: str, from_state: str, to_state: str) -> None:
        """Execute one downward step on the ladder."""
        entry = Orchestrator._registry[model_id]

        with Orchestrator._locks_ordered(model_id):
            if from_state == "up" and to_state == "sleep":
                entry["instance"].sleep().wait()
                Orchestrator._set_state(model_id, "sleep")
                return

            if from_state == "sleep" and to_state == "checkpoint":
                entry["instance"].unpin().checkpoint_cuda().wait()
                if entry.get("slot") is not None:
                    Slots.deallocate(entry["slot"])
                    entry["slot"] = None
                entry["gpu"] = None
                Orchestrator._set_state(model_id, "checkpoint")
                return

            if from_state == "checkpoint" and to_state == "saved":
                entry["instance"].teardown().wait().remove()
                entry["instance"] = None
                Orchestrator._set_state(model_id, "saved")
                return

            raise AssertionError(
                f"_step_down unexpected transition {from_state} -> {to_state}")

    # ------------------------------------------------------------------
    # generate
    # ------------------------------------------------------------------

    @staticmethod
    def submit_generate(model_id: str, prompts: list[str] | str,
                        sampling_params: dict | int | None = None
                        ) -> tuple[int | None, Future | None]:
        """Submit a non-blocking generate; returns ``(req_id, future)``.

        Companion to :meth:`generate` for callers that need a stable
        request id (e.g. the HTTP control plane) without racing on
        ``_request_counter``.  Returns ``(None, None)`` when the model is
        not registered, mirroring :meth:`generate`'s warn-and-skip
        semantics.
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        if isinstance(sampling_params, int):
            sampling_params = {"max_tokens": sampling_params, "ignore_eos": True}
        if sampling_params is None:
            sampling_params = {}
        entry = Orchestrator._registry.get(model_id)
        if entry is None:
            _console(f"WARNING: model '{model_id}' is not registered, skipping generate")
            return None, None
        _console(f"{model_id}: generate received")
        print(f"[orchestrator] generate  model_id={model_id}")

        with Orchestrator._request_lock:
            ent = Orchestrator._registry[model_id]
            start_state = ent.get("state")
            req_id = Orchestrator._request_counter
            Orchestrator._request_counter += 1
            t_submit = time.perf_counter()
            req_record = {
                "req_id": req_id,
                "model_id": model_id,
                "state": "waiting",
                "start_state": start_state,
                "t_submit": t_submit,
                "t_gen_start": None,
                "t_done": None,
                "prompt_tokens": None,
                "completion_tokens": None,
                # Timeline of (t_perf, model_state) pairs observed
                # while this request is in flight.  Seeded with the
                # model's state at submit; _set_state appends more.
                "state_log": [(t_submit, start_state)] if start_state else [],
            }
            Orchestrator._request_log.append(req_record)

        Orchestrator._generate_queue.append(
            (model_id, prompts, sampling_params, req_record)
        )

        prev_move = Orchestrator._futures.get(model_id)
        fut = Orchestrator._pool.submit(
            Orchestrator._generate_sync, model_id, prompts, sampling_params,
            prev_move, req_record,
        )
        Orchestrator._generate_futures.append(fut)
        Orchestrator._last_generate_future[model_id] = fut
        return req_id, fut

    @staticmethod
    def generate(model_id: str, prompts: list[str] | str,
                 sampling_params: dict | int | None = None) -> Future:
        """Submit a non-blocking generate.  Returns a Future[list].

        *sampling_params* can be a dict, or an int shorthand for
        ``{"max_tokens": N}``.

        Automatically moves the model to **up** if needed, runs inference,
        and leaves the model in **running** state until a waiter thread
        detects all in-flight requests are done, then transitions to **up**.
        """
        _, fut = Orchestrator.submit_generate(model_id, prompts, sampling_params)
        return fut

    @staticmethod
    def generate_all(prompts: list[str] | str,
                     sampling_params: dict | int | None = None
                     ) -> list[int]:
        """Fan out :meth:`generate` (same prompts) to every registered model.

        Returns the list of server-assigned ``req_id``s, one per model in
        registry insertion order, mirroring how :meth:`move_all` fans out.
        Each per-model submit goes through the standard generate
        pipeline, so the futures are independent and run concurrently.
        """
        req_ids: list[int] = []
        for mid in list(Orchestrator._registry):
            req_id, _fut = Orchestrator.submit_generate(
                mid, prompts, sampling_params)
            if req_id is not None:
                req_ids.append(req_id)
        return req_ids

    @staticmethod
    def _generate_sync(model_id: str, prompts: list[str],
                       sampling_params: dict,
                       prev_move_future=None,
                       req_record: dict | None = None) -> list:

        entry = Orchestrator._registry[model_id]
        gen_lock = Orchestrator._get_generate_lock(model_id)

        # -- Phase 1: ensure model is running --
        # Wait for any prior move/register outside the lock.
        if prev_move_future is not None:
            cur_state = entry.get("state", "?")
            if cur_state != "running":
                print(f"[orchestrator] _generate_sync  model_id={model_id}  "
                      f"state={cur_state}  waiting for move future")
                prev_move_future.result()

        if req_record is not None and req_record["state"] == "done":
            return []

        # Serialize the state-check + move under gen_lock so only one
        # thread does the move.  _move_sync("up", announce_state="running")
        # is fast when the model is already up (just sets the state).
        # When the model is NOT up, the move takes time, but other
        # threads block here and skip the move once the first completes.
        with gen_lock:
            if entry["state"] != "running":
                Orchestrator._timing.gpu_wait_s = 0.0
                Orchestrator._timing.migrate_s = 0.0
                t_move = time.perf_counter()
                Orchestrator._move_sync(model_id, "up",
                                        announce_state="running")
                total_move = time.perf_counter() - t_move
                move_gpu_wait = Orchestrator._timing.gpu_wait_s
                move_migrate = Orchestrator._timing.migrate_s
                move_up = total_move - move_gpu_wait - move_migrate
            else:
                move_gpu_wait = 0.0
                move_migrate = 0.0
                move_up = 0.0

        if req_record is not None and req_record["state"] == "done":
            return []

        inst = entry["instance"]

        # -- Phase 2: drain queue and submit (under gen_lock -- short hold) --
        with gen_lock:
            batch = []
            while True:
                item = None
                for queued in list(Orchestrator._generate_queue):
                    if queued[0] == model_id:
                        try:
                            Orchestrator._generate_queue.remove(queued)
                        except ValueError:
                            continue
                        item = queued
                        break
                if item is None:
                    break
                batch.append(item)

            if model_id not in Orchestrator._inflight:
                Orchestrator._inflight[model_id] = []

            my_events = []
            for i, (_, q_prompts, q_sp, q_rec) in enumerate(batch):
                if i == 0:
                    q_rec["gpu_wait_s"] = move_gpu_wait
                    q_rec["migrate_s"] = move_migrate
                    q_rec["up_s"] = move_up
                else:
                    q_rec["gpu_wait_s"] = 0.0
                    q_rec["migrate_s"] = 0.0
                    q_rec["up_s"] = 0.0

                done_event = threading.Event()
                try:
                    q_rec["state"] = "generating"
                    q_rec["t_gen_start"] = time.perf_counter()
                    inst.generate(q_prompts, q_sp)
                    rid = inst.last_req_id
                    Orchestrator._inflight[model_id].append(
                        (rid, q_rec, done_event))
                except Exception as exc:
                    import traceback; traceback.print_exc()
                    print(f"[orchestrator] {model_id}: generate submit "
                          f"failed: {exc}")
                    q_rec["state"] = "error"
                    q_rec["t_done"] = time.perf_counter()
                    rid = None
                    done_event.set()

                my_events.append((q_rec, rid, done_event))

            if not Orchestrator._waiter_active.get(model_id):
                Orchestrator._waiter_active[model_id] = True
                t = threading.Thread(
                    target=Orchestrator._start_generate_waiter,
                    args=(model_id,),
                    daemon=True,
                )
                t.start()

        # -- Phase 3: wait for our requests (lock released) --
        if my_events:
            for q_rec, rid, done_event in my_events:
                if rid is not None:
                    done_event.wait()
        else:
            while req_record is not None and req_record["state"] not in ("done", "error"):
                time.sleep(0.05)

        result = None
        for q_rec, rid, _ in my_events:
            if q_rec.get("state") == "done" and rid:
                gen_result = inst.generate_results.pop(rid, None)
                if gen_result:
                    result = gen_result.get("outputs")

        return result or []

    @staticmethod
    def _start_generate_waiter(model_id: str):
        """Background thread: reads inst._result_queue, resolves inflight
        Events.  Exits when no inflight requests remain and
        inst._pending_count is 0."""
        entry = Orchestrator._registry[model_id]
        inst = entry["instance"]
        inst._external_waiter = True
        gen_lock = Orchestrator._get_generate_lock(model_id)

        while True:
            try:
                result = inst._result_queue.get(timeout=0.5)
            except Exception:
                with gen_lock:
                    inflight = Orchestrator._inflight.get(model_id, [])
                    if not inflight and inst._pending_count == 0:
                        Orchestrator._waiter_active[model_id] = False
                        inst._external_waiter = False
                        # Mutate slot + state under the model's _lock so
                        # concurrent Phase 2 scans on other models
                        # cannot observe a torn (slot=None, state=running)
                        # snapshot that drops this model from HBM
                        # accounting.  Lock order matches the rest of
                        # the orchestrator: gen_lock -> _lock.
                        with Orchestrator._locks_ordered(model_id):
                            if entry.get("slot") is not None:
                                Slots.deallocate(entry["slot"])
                                entry["slot"] = None
                            Orchestrator._set_state(model_id, "up")
                        return
                continue

            cmd, elapsed, error, info = result

            inst._pending_count -= 1
            if inst._pending_cmds:
                inst._pending_cmds.pop(0)

            status = "OK" if error is None else "FAILED"
            display_info = ({k: v for k, v in info.items() if k != "outputs"}
                            if cmd == "generate" else info)
            inst._print(f"[gpu{inst.gpu}] [{time.strftime('%H:%M:%S')}] "
                        f"{cmd} {status} ({elapsed:.3f}s) {display_info}")

            if error is None:
                inst._apply_result(cmd, info)

            if cmd != "generate":
                continue

            completed_rid = info.get("req_id")
            with gen_lock:
                inflight = Orchestrator._inflight.get(model_id, [])
                matched = None
                if completed_rid is not None:
                    for i, (rid, q_rec, done_event) in enumerate(inflight):
                        if rid == completed_rid:
                            matched = (i, rid, q_rec, done_event)
                            break
                if matched is None and inflight:
                    matched = (0, inflight[0][0], inflight[0][1],
                               inflight[0][2])

                if matched:
                    i, rid, q_rec, done_event = matched
                    inflight.pop(i)
                    t_done = time.perf_counter()
                    q_rec["state"] = "done"
                    q_rec["t_done"] = t_done

                    gen_result = inst.generate_results.get(rid)
                    if gen_result is not None:
                        q_rec["prompt_tokens"] = gen_result["prompt_tokens"]
                        q_rec["completion_tokens"] = gen_result["completion_tokens"]
                        outputs = gen_result["outputs"]
                    else:
                        q_rec["prompt_tokens"] = inst.last_prompt_tokens
                        q_rec["completion_tokens"] = inst.last_completion_tokens
                        outputs = inst.last_generate_result

                    print(f"[orchestrator] {model_id}: generate done  "
                          f"({elapsed:.1f}s)")
                    snippet = ""
                    try:
                        if outputs:
                            snippet = outputs[0][0].replace("\n", " ")[:100]
                    except (TypeError, IndexError):
                        snippet = str(outputs)[:100]
                    _console(f"{model_id}: generated ({elapsed:.1f}s) "
                             f"-> \"{snippet}\"")

                    done_event.set()

                if not inflight and inst._pending_count == 0:
                    Orchestrator._waiter_active[model_id] = False
                    inst._external_waiter = False
                    # See comment in the timeout branch above: take the
                    # model's _lock so the slot release + state flip is
                    # atomic from the perspective of other models'
                    # Phase 2 scans.
                    with Orchestrator._locks_ordered(model_id):
                        if entry.get("slot") is not None:
                            Slots.deallocate(entry["slot"])
                            entry["slot"] = None
                        Orchestrator._set_state(model_id, "up")
                    return

    # ------------------------------------------------------------------
    # remove
    # ------------------------------------------------------------------

    @staticmethod
    def remove_all() -> None:
        """Fan out :meth:`remove` to every registered model."""
        for mid in list(Orchestrator._registry):
            Orchestrator.remove(mid)

    @staticmethod
    def remove(model_id: str) -> None:
        """Delete a model's image and remove it from the registry.

        Auto-transitions to **saved** if needed.  Use :meth:`remove_all`
        to delete every registered model.
        """
        entry = Orchestrator._registry.get(model_id)
        if entry is None:
            msg = f"{model_id}: not registered, skipping remove"
            _console(msg)
            print(f"[orchestrator] WARNING: {msg}")
            return
        _console(f"{model_id}: remove received")
        prev = Orchestrator._futures.get(model_id)
        prev_gen = Orchestrator._last_generate_future.pop(model_id, None)
        fut = Orchestrator._pool.submit(
            Orchestrator._remove_sync, model_id, prev, prev_gen,
        )
        Orchestrator._futures[model_id] = fut

    @staticmethod
    def _remove_sync(model_id: str, prev_future=None,
                     prev_gen_future=None) -> None:
        if prev_future is not None:
            prev_future.result()
        if prev_gen_future is not None:
            prev_gen_future.result()
        entry = Orchestrator._registry.get(model_id)
        if entry is None:
            return
        if entry["state"] != "saved":
            Orchestrator._move_sync(model_id, "saved")
        image_dir = entry.get("image_dir")
        if image_dir and os.path.isdir(image_dir):
            shutil.rmtree(image_dir)
            print(f"[orchestrator] {model_id}: deleted image {image_dir}")
        Orchestrator._registry.pop(model_id, None)
        Orchestrator._futures.pop(model_id, None)
        _console(f"{model_id}: removed")
        Orchestrator._print_states()

    # ------------------------------------------------------------------
    # wait
    # ------------------------------------------------------------------

    @staticmethod
    def wait_all() -> None:
        """Block until every pending move/generate future completes."""
        print(f"[orchestrator] wait  model_id=all")
        t0 = time.perf_counter()
        for fut in list(Orchestrator._futures.values()):
            fut.result()
        gen_futs = list(Orchestrator._generate_futures)
        Orchestrator._generate_futures = []
        for fut in gen_futs:
            try:
                fut.result()
            except Exception:
                pass
        elapsed = time.perf_counter() - t0
        print(f"[orchestrator] wait done  ({elapsed:.1f}s)")

    @staticmethod
    def wait(model_id: str) -> None:
        """Block until pending futures for *model_id* complete.

        Use :meth:`wait_all` to wait on every pending future.
        """
        print(f"[orchestrator] wait  model_id={model_id}")
        t0 = time.perf_counter()
        fut = Orchestrator._futures.get(model_id)
        if fut is not None:
            fut.result()
        gen_fut = Orchestrator._last_generate_future.get(model_id)
        if gen_fut is not None:
            gen_fut.result()
        elapsed = time.perf_counter() - t0
        print(f"[orchestrator] wait done  ({elapsed:.1f}s)")

    # ------------------------------------------------------------------
    # status
    # ------------------------------------------------------------------

    @staticmethod
    def status() -> None:
        """Print GPUs and registered models with their states."""
        if not Orchestrator._gpu_ids:
            _console("Orchestrator not initialized. Call Orchestrator.init() first.")
            return

        _console(f"\nOrchestrator  image_cache={Orchestrator._image_cache}")

        try:
            pynvml.nvmlInit()
            _console(f"\nGPUs ({len(Orchestrator._gpu_ids)}):")
            for idx in range(pynvml.nvmlDeviceGetCount()):
                h = pynvml.nvmlDeviceGetHandleByIndex(idx)
                name = pynvml.nvmlDeviceGetName(h)
                if isinstance(name, bytes):
                    name = name.decode()
                m = pynvml.nvmlDeviceGetMemoryInfo(h)
                used = m.used / (1 << 30)
                total = m.total / (1 << 30)
                free = total - used
                _console(f"  GPU {idx}: {name}  "
                         f"{used:.1f} / {total:.1f} GiB used  "
                         f"({free:.1f} GiB free)")
        except Exception:
            _console(f"\nGPUs: {Orchestrator._gpu_ids}")

        if not Orchestrator._registry:
            _console("\nNo models registered.\n")
            return

        pid_gpu_mib: dict[int, int] = {}
        try:
            pynvml.nvmlInit()
            _NVML_NOT_AVAILABLE = 0xFFFFFFFFFFFFFFFF
            for idx in range(pynvml.nvmlDeviceGetCount()):
                h = pynvml.nvmlDeviceGetHandleByIndex(idx)
                try:
                    procs = pynvml.nvmlDeviceGetComputeRunningProcesses(h)
                except Exception:
                    continue
                for p in procs:
                    used = getattr(p, "usedGpuMemory", None)
                    if used is None or used == _NVML_NOT_AVAILABLE:
                        continue
                    pid_gpu_mib[p.pid] = pid_gpu_mib.get(p.pid, 0) + int(used // (1 << 20))
        except Exception:
            pass

        sorted_models = sorted(Orchestrator._registry.items(),
                               key=lambda item: item[1].get("pinned_cpu_bytes", 0),
                               reverse=True)
        max_id = max(len(mid) for mid in Orchestrator._registry)
        _console(f"\nModels ({len(Orchestrator._registry)}):")
        for model_id, entry in sorted_models:
            state = entry["state"]
            inst = entry["instance"]
            gpu = entry.get("gpu")
            gpu_str = f"  gpu={gpu}" if gpu is not None else ""
            pinned = entry.get("pinned_cpu_bytes", 0)
            pinned_str = f"  pinned_cpu={pinned / 2**30:.1f} GiB" if pinned and state != "saved" else ""
            gpu_mem_str = ""
            if inst is not None and inst.pid and inst.pid in pid_gpu_mib:
                gpu_mem_str = f"  memory={pid_gpu_mib[inst.pid] / 1024:.1f} GiB"
            saved_str = ""
            if state == "saved":
                saved_str = f"  image={entry.get('image_dir', '?')}"
            _console(f"  {model_id:<{max_id}}  [{state}]{gpu_str}{pinned_str}{gpu_mem_str}{saved_str}")
        _console("")
