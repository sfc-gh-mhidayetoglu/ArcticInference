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
import os
import shutil
import subprocess
import threading
import time
from contextlib import contextmanager
from concurrent.futures import Future, ThreadPoolExecutor
from http.server import HTTPServer
from typing import Any

from gpu_slot import SlotPool, GpuSlot
from instance import Instance
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
    """Return GPU device indices via nvidia-smi without initializing CUDA."""
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
        text=True,
    )
    return [int(line.strip()) for line in out.strip().splitlines()]


_STATES = ["saved", "checkpoint", "sleep", "up"]


class _StateRegressed(Exception):
    """Raised when a concurrent eviction moved a model backward mid-climb."""


class Orchestrator:
    """Static registry that maps human-readable model IDs to Instances.

    States (ordered ladder):
        saved      -- image on disk, no process, no GPU
        checkpoint -- image on disk + live CRIU process in memory, no GPU
        sleep      -- CUDA context restored on a GPU (small footprint),
                      GPU is NOT locked; multiple sleep models can coexist
        up         -- image on disk + live process + GPU held + weights on GPU
        running    -- transient sub-state of 'up' during generate()
    """

    _registry: dict[str, Any] = {}
    _futures: dict[str, Future] = {}
    _gpu_ids: list[int] = []
    _gpu_pool: SlotPool | None = None
    _gpus: dict[int, GpuSlot] = {}
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
        Orchestrator._registry[model_id]["state"] = state
        Orchestrator._registry[model_id]["state_since"] = time.perf_counter()
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
        Orchestrator._gpu_pool = SlotPool(Orchestrator._gpu_ids)
        Orchestrator._gpus = Orchestrator._gpu_pool.slots
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
            Orchestrator._registry[model_id] = {
                "state": "saved",
                "instance": None,
                "gpu": None,
                "vllm_config": vllm_config,
                "image_dir": image_dir,
                "pinned_bytes": meta.get("pinned_bytes", 0),
                "_lock": threading.RLock(),
            }
            pinned = meta.get("pinned_bytes", 0)
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

        # t0 is set lazily on the first generate, not here.

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

        gpu, wait_s = Orchestrator._gpu_pool.acquire_exclusive(model_id)
        t_acquired = time.perf_counter()

        Orchestrator._registry[model_id] = {
            "state": "init",
            "instance": inst,
            "gpu": gpu,
            "vllm_config": vllm_config,
            "image_dir": image_dir,
            "pinned_bytes": 0,
            "state_since": time.perf_counter(),
            "_lock": threading.RLock(),
        }
        Orchestrator._print_states()

        with Orchestrator._locks_ordered(model_id):
            inst.init(gpu).attach().repin().stage().unpin().sleep().checkpoint().wait()
            inst.save(image_dir).wait()
            print(f"[orchestrator] {model_id}: image saved to {image_dir}")

            inst._send("exit")
            inst._reset()

            Orchestrator._gpu_pool.release_exclusive(gpu, lambda: None)
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
            entry["pinned_bytes"] = inst.pinned_bytes
            Orchestrator._set_state(model_id, "saved")

    # ------------------------------------------------------------------
    # move  (walk the state ladder)
    # ------------------------------------------------------------------

    @staticmethod
    def move(model_id: str, target: str, target_gpu: int | None = None) -> None:
        """Move *model_id* to *target* state by walking the ladder.

        Valid targets: ``"saved"``, ``"checkpoint"``, ``"sleep"``, ``"up"``.
        Submits the transition to the thread pool (non-blocking).

        *target_gpu* (optional) is only valid when *target* is ``"sleep"``
        and forces the model to restore onto that specific GPU.
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
            with Orchestrator._locks_ordered(model_id):
                if entry["state"] == target:
                    Orchestrator._set_state(model_id, announce_state)
                    return
            current = entry["state"]

        cur_idx = _STATES.index(current)
        tgt_idx = _STATES.index(target)

        t0 = time.perf_counter()

        if target == "sleep" and target_gpu is not None and current == "up":
            current_gpu = entry.get("gpu")
            if target_gpu == current_gpu:
                Orchestrator._step_down(model_id, "up", "sleep")
            else:
                chk_idx = _STATES.index("checkpoint")
                for step in range(cur_idx, chk_idx, -1):
                    Orchestrator._step_down(model_id, _STATES[step], _STATES[step - 1])
                Orchestrator._step_up(model_id, "checkpoint", "sleep",
                                      target_gpu=target_gpu)
        elif target == "sleep" and target_gpu is not None and current == "sleep":
            current_gpu = entry.get("gpu")
            if target_gpu == current_gpu:
                _console(f"{model_id}: already sleeping on GPU {target_gpu}")
                return
            Orchestrator._step_down(model_id, "sleep", "checkpoint")
            Orchestrator._step_up(model_id, "checkpoint", "sleep",
                                  target_gpu=target_gpu)
        elif cur_idx < tgt_idx:
            while True:
                cur = entry["state"]
                if cur == "running":
                    break
                ci = _STATES.index(cur) if cur in _STATES else -1
                if ci >= tgt_idx:
                    break
                if ci < 0:
                    time.sleep(0.05)
                    continue
                nxt = _STATES[ci + 1]
                kw = {}
                if cur == "checkpoint" and nxt == "sleep":
                    kw["target_gpu"] = target_gpu
                if nxt == "up" and announce_state is not None:
                    kw["announce_state"] = announce_state
                try:
                    Orchestrator._step_up(model_id, cur, nxt, **kw)
                except _StateRegressed:
                    _console(f"{model_id}: evicted mid-climb, re-planning")
                    print(f"[orchestrator] {model_id}: evicted mid-climb "
                          f"(was {cur}->{nxt}), re-planning from "
                          f"{entry['state']}")
                    continue
        else:
            for step in range(cur_idx, tgt_idx, -1):
                Orchestrator._step_down(model_id, _STATES[step], _STATES[step - 1])
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

        The ``sleep -> up`` branch manages its own lock scope so the
        migration path can release model_id's lock before acquiring
        a new slot (which may evict arbitrary models).
        """
        entry = Orchestrator._registry[model_id]

        with Orchestrator._locks_ordered(model_id):
            if from_state == "saved" and to_state == "checkpoint":
                inst = Instance(entry["vllm_config"])
                inst.load(entry["image_dir"]).wait()
                inst.pinned_bytes = entry.get("pinned_bytes", 0)
                entry["instance"] = inst
                Orchestrator._set_state(model_id, "checkpoint")
                return

            if from_state == "checkpoint" and to_state == "sleep":
                inst = entry["instance"]
                if target_gpu is not None:
                    slot = Orchestrator._gpus[target_gpu]
                else:
                    slots = list(Orchestrator._gpus.values())
                    slot = GpuSlot.pick_for_sleep_placement(slots)
                gpu = slot.gpu_id
                slot.add_sleeper(model_id)
                _console(f"{model_id}: placed on GPU {gpu}")
                print(f"[orchestrator] {model_id}: placed on GPU {gpu}")
                inst.restore(gpu).repin().wait()
                entry["gpu"] = gpu
                Orchestrator._set_state(model_id, "sleep")
                return

        if from_state != "sleep" or to_state != "up":
            raise AssertionError(
                f"_step_up unexpected transition {from_state} -> {to_state}")

        # --- sleep -> up  (lock managed per-branch) ---
        while True:
            with Orchestrator._locks_ordered(model_id):
                actual = entry["state"]
                home_gpu = entry.get("gpu")
                if actual != "sleep" or home_gpu is None:
                    raise _StateRegressed(
                        f"{model_id}: expected sleep with gpu, "
                        f"got state={actual} gpu={home_gpu}")
                inst = entry["instance"]
                slot = Orchestrator._gpus[home_gpu]

                already_locked = (slot.locked_by == model_id)

                if already_locked or slot.try_lock(model_id):
                    slot.remove_sleeper(model_id)
                    _console(f"{model_id}: acquired home GPU {home_gpu}")
                    print(f"[orchestrator] {model_id}: acquired home GPU {home_gpu}")
                    inst.wake_up_weights().h2d().scatter().wake_up_kv_cache().wait()
                    Orchestrator._set_state(model_id, announce_state or "up")
                    return

                victim_id = slot.locked_by
                victim_state = Orchestrator._registry[victim_id]["state"]
                print(f"victim is={victim_id}  victim state is={victim_state}")

                if victim_state == "up":
                    evict_ok = False
                    with Orchestrator._locks_ordered(victim_id):
                        if slot.locked_by == victim_id and Orchestrator._registry[victim_id]["state"] == "up":
                            victim_inst = Orchestrator._registry[victim_id]["instance"]
                            victim_inst.sleep().wait()
                            slot.transfer_lock(victim_id, model_id)
                            Orchestrator._set_state(victim_id, "sleep")
                            evict_ok = True
                    if evict_ok:
                        if os.environ.get("SP_DEMO_MODE") == "1":
                            try:
                                Orchestrator._move_sync(victim_id, "saved")
                            except Exception as exc:
                                print(f"[orchestrator] WARNING: demo-mode "
                                      f"evict {victim_id} -> saved failed: {exc}")
                                _console(f"WARNING: evict {victim_id} -> saved failed")
                        inst.wake_up_weights().h2d().scatter().wake_up_kv_cache().wait()
                        Orchestrator._set_state(model_id, announce_state or "up")
                        return
                    _console(f"{model_id}: victim {victim_id} changed, retrying")
                    continue

                _console(f"{model_id}: home GPU {home_gpu} busy, migrating")
                print(f"[orchestrator] {model_id}: home GPU {home_gpu} busy, migrating")
                t_mig = time.perf_counter()
                slot.remove_sleeper(model_id)
                inst.unpin().checkpoint().wait()
                entry["gpu"] = None
                Orchestrator._timing.migrate_s = time.perf_counter() - t_mig
                Orchestrator._set_state(model_id, "wait")

            # model_id lock released -- safe to acquire a slot which may
            # evict arbitrary models (acquiring their locks in any order).
            pool = Orchestrator._gpu_pool
            _console(f"{model_id}: waiting for slot...")
            print(f"[orchestrator] {model_id}: waiting for slot ...")
            t_wait = time.perf_counter()
            my_turn = pool.enqueue_waiter()
            try:
                while True:
                    all_slots = list(pool.slots.values())

                    free = GpuSlot.coldest_free_slot(all_slots)
                    if free is not None and free.try_lock(model_id):
                        Orchestrator._timing.gpu_wait_s = time.perf_counter() - t_wait
                        _console(f"{model_id}: locked free GPU {free.gpu_id}")
                        print(f"[orchestrator] {model_id}: locked free GPU {free.gpu_id}")
                        gpu = free.gpu_id
                        break

                    preemptable = [
                        s for s in all_slots
                        if s.locked_by is not None
                        and s.locked_by != model_id
                        and Orchestrator._registry.get(s.locked_by, {}).get("state") == "up"
                    ]
                    if preemptable:
                        victim_slot = min(preemptable, key=lambda s: s.last_event_ts)
                        victim_id = victim_slot.locked_by
                        evict_ok = False
                        with Orchestrator._locks_ordered(victim_id):
                            if (victim_slot.locked_by == victim_id
                                    and Orchestrator._registry.get(victim_id, {}).get("state") == "up"):
                                victim_inst = Orchestrator._registry[victim_id]["instance"]
                                victim_inst.sleep().wait()
                                victim_slot.transfer_lock(victim_id, model_id)
                                Orchestrator._set_state(victim_id, "sleep")
                                evict_ok = True
                        if evict_ok:
                            if os.environ.get("SP_DEMO_MODE") == "1":
                                try:
                                    Orchestrator._move_sync(victim_id, "saved")
                                except Exception as exc:
                                    print(f"[orchestrator] WARNING: demo-mode "
                                          f"evict {victim_id} -> saved failed: {exc}")
                                    _console(f"WARNING: evict {victim_id} -> saved failed")
                            Orchestrator._timing.gpu_wait_s = time.perf_counter() - t_wait
                            _console(f"{model_id}: evicted {victim_id}, locked GPU {victim_slot.gpu_id}")
                            print(f"[orchestrator] {model_id}: evicted {victim_id}, locked GPU {victim_slot.gpu_id}")
                            gpu = victim_slot.gpu_id
                            break

                    my_turn.wait(timeout=0.5)
                    my_turn.clear()
            finally:
                pool.dequeue_waiter(my_turn)

            with Orchestrator._locks_ordered(model_id):
                new_slot = Orchestrator._gpus[gpu]
                new_slot.add_sleeper(model_id)
                _console(f"{model_id}: placed on GPU {gpu}")
                print(f"[orchestrator] {model_id}: placed on GPU {gpu}")
                entry["instance"].restore(gpu).repin().wait()
                entry["gpu"] = gpu
                Orchestrator._set_state(model_id, "sleep")

    @staticmethod
    def _step_down(model_id: str, from_state: str, to_state: str) -> None:
        """Execute one downward step on the ladder."""
        entry = Orchestrator._registry[model_id]

        with Orchestrator._locks_ordered(model_id):
            if from_state == "up" and to_state == "sleep":
                inst = entry["instance"]
                inst.sleep().wait()
                gpu = entry["gpu"]
                slot = Orchestrator._gpus[gpu]
                slot.unlock()
                slot.add_sleeper(model_id)
                Orchestrator._gpu_pool.notify_acquire_waiters()
                Orchestrator._set_state(model_id, "sleep")

            elif from_state == "sleep" and to_state == "checkpoint":
                if entry["state"] != "sleep":
                    _console(f"{model_id}: eviction skipped (state={entry['state']})")
                    print(f"[orchestrator] {model_id}: eviction skipped "
                          f"(state={entry['state']})")
                    return
                inst = entry["instance"]
                gpu = entry["gpu"]
                slot = Orchestrator._gpus[gpu]
                slot.remove_sleeper(model_id)
                inst.unpin().checkpoint().wait()
                entry["gpu"] = None
                Orchestrator._set_state(model_id, "checkpoint")

            elif from_state == "checkpoint" and to_state == "saved":
                inst = entry["instance"]
                inst.teardown().wait().remove()
                entry["instance"] = None
                Orchestrator._set_state(model_id, "saved")

            else:
                raise AssertionError(
                    f"_step_down unexpected transition {from_state} -> {to_state}")

    # ------------------------------------------------------------------
    # generate
    # ------------------------------------------------------------------

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
        if isinstance(prompts, str):
            prompts = [prompts]
        if isinstance(sampling_params, int):
            sampling_params = {"max_tokens": sampling_params, "ignore_eos": True}
        if sampling_params is None:
            sampling_params = {}
        entry = Orchestrator._registry.get(model_id)
        if entry is None:
            _console(f"WARNING: model '{model_id}' is not registered, skipping generate")
            return
        _console(f"{model_id}: generate received")
        print(f"[orchestrator] generate  model_id={model_id}")

        with Orchestrator._request_lock:
            if Orchestrator._request_counter == 0:
                init_t0()
            ent = Orchestrator._registry[model_id]
            start_state = ent.get("state")
            req_id = Orchestrator._request_counter
            Orchestrator._request_counter += 1
            req_record = {
                "req_id": req_id,
                "model_id": model_id,
                "state": "waiting",
                "start_state": start_state,
                "t_submit": time.perf_counter(),
                "t_gen_start": None,
                "t_done": None,
                "prompt_tokens": None,
                "completion_tokens": None,
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
        return fut

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
                        Orchestrator._set_state(model_id, "up")
                        Orchestrator._gpu_pool.notify_acquire_waiters()
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
                    Orchestrator._set_state(model_id, "up")
                    Orchestrator._gpu_pool.notify_acquire_waiters()
                    return

    # ------------------------------------------------------------------
    # remove
    # ------------------------------------------------------------------

    @staticmethod
    def remove(model_id: str | None = None) -> None:
        """Delete a model's image and remove it from the registry.

        Auto-transitions to **saved** if needed.  Pass *None* to remove all.
        """
        if model_id is None:
            for mid in list(Orchestrator._registry):
                Orchestrator.remove(mid)
            return
        entry = Orchestrator._registry.get(model_id)
        if entry is None:
            raise KeyError(f"model '{model_id}' is not registered")
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
    def wait(model_id: str | None = None) -> None:
        """Block until futures complete.  None = wait on all."""
        label = model_id or "all"
        print(f"[orchestrator] wait  model_id={label}")
        t0 = time.perf_counter()
        if model_id is not None:
            fut = Orchestrator._futures.get(model_id)
            if fut is not None:
                fut.result()
            gen_fut = Orchestrator._last_generate_future.get(model_id)
            if gen_fut is not None:
                gen_fut.result()
        else:
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
            out = subprocess.check_output(
                ["nvidia-smi",
                 "--query-gpu=index,name,memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                text=True,
            )
            _console(f"\nGPUs ({len(Orchestrator._gpu_ids)}):")
            for line in out.strip().splitlines():
                idx, name, used_mib, total_mib = (x.strip() for x in line.split(","))
                used = int(used_mib) / 1024
                total = int(total_mib) / 1024
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
            out = subprocess.check_output(
                ["nvidia-smi",
                 "--query-compute-apps=pid,used_gpu_memory",
                 "--format=csv,noheader,nounits"],
                text=True,
            )
            for line in out.strip().splitlines():
                parts = line.split(",")
                if len(parts) == 2:
                    pid_gpu_mib[int(parts[0].strip())] = int(parts[1].strip())
        except Exception:
            pass

        sorted_models = sorted(Orchestrator._registry.items(),
                               key=lambda item: item[1].get("pinned_bytes", 0),
                               reverse=True)
        max_id = max(len(mid) for mid in Orchestrator._registry)
        _console(f"\nModels ({len(Orchestrator._registry)}):")
        for model_id, entry in sorted_models:
            state = entry["state"]
            inst = entry["instance"]
            gpu = entry.get("gpu")
            gpu_str = f"  gpu={gpu}" if gpu is not None else ""
            pinned = entry.get("pinned_bytes", 0)
            pinned_str = f"  pinned={pinned / 2**30:.1f} GiB" if pinned and state != "saved" else ""
            gpu_mem_str = ""
            if inst is not None and inst.pid and inst.pid in pid_gpu_mib:
                gpu_mem_str = f"  memory={pid_gpu_mib[inst.pid] / 1024:.1f} GiB"
            saved_str = ""
            if state == "saved":
                saved_str = f"  image={entry.get('image_dir', '?')}"
            _console(f"  {model_id:<{max_id}}  [{state}]{gpu_str}{pinned_str}{gpu_mem_str}{saved_str}")
        _console("")
