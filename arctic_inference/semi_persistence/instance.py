"""Standalone Instance for a vLLM engine.

Each Instance is a GPU-agnostic handle.  The GPU is specified at
init(gpu) time, which also spawns the worker process.  All primitives
are non-blocking and return self for chaining.  Cross-instance
dependencies are expressed via after().

Usage:
    instance = Instance({"model": "/data-fast/Qwen/Qwen3-32B"})
    instance.init(gpu=0).attach().sleep().checkpoint().wait()
"""
from __future__ import annotations

import os
import time
import weakref

import torch.multiprocessing as mp

from worker import worker_loop, _read_architecture, _weight_footprint

_fork_ctx = mp.get_context("fork")


def _gpu_mem_info(gpu_id: int) -> tuple[int, int]:
    """Return (free_bytes, total_bytes) without initializing CUDA."""
    import subprocess
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free,memory.total",
         "--format=csv,noheader,nounits", f"--id={gpu_id}"],
        text=True,
    )
    free_mib, total_mib = (int(x.strip()) for x in out.split(","))
    return free_mib * 1024 * 1024, total_mib * 1024 * 1024


def _gpu_count() -> int:
    """Return number of GPUs without initializing CUDA."""
    import subprocess
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        text=True,
    )
    return len(out.strip().splitlines())

_next_instance_id = 0
_counter_registry: dict[int, mp.Value] = {}


def _alloc_instance_id():
    global _next_instance_id
    _next_instance_id += 1
    return _next_instance_id - 1


class Instance:

    _all: weakref.WeakValueDictionary[int, "Instance"] = weakref.WeakValueDictionary()

    def __init__(self, vllm_config: dict, *, use_odirect: bool = True):
        self.gpu = None
        self.vllm_config = vllm_config
        self.instance_id = _alloc_instance_id()
        Instance._all[self.instance_id] = self

        model_path = vllm_config["model"]
        self.arch = _read_architecture(model_path)
        self.weight_bytes = _weight_footprint(model_path)

        self.pid = None
        self.state = "created"
        self.pinned_bytes = 0
        self._pending_count = 0
        self._pending_cmds = []
        self._total_sent = 0
        self._use_odirect = use_odirect

        self._cmd_queue = _fork_ctx.Queue()
        self._result_queue = _fork_ctx.Queue()
        self._completed_counter = _fork_ctx.Value('i', 0)
        _counter_registry[self.instance_id] = self._completed_counter
        self._worker = None

    def __repr__(self):
        return (f"Instance(id={self.instance_id}, gpu={self.gpu}, "
                f"arch={self.arch[0] if self.arch else None}, "
                f"pid={self.pid}, state={self.state}, "
                f"pinned={self.pinned_bytes / 2**30:.2f} GiB)")

    # -- Internal helpers -------------------------------------------------------

    def _reset(self):
        """Reset instance to created state after remove completes."""
        if self._worker is not None:
            self._worker.join(timeout=30)
            self._worker = None
        self.state = "created"
        self.gpu = None
        self.pid = None
        self.pinned_bytes = 0

    def _send(self, cmd, **kwargs):
        self._cmd_queue.put((cmd, kwargs))
        self._pending_count += 1
        self._pending_cmds.append(cmd)
        self._total_sent += 1
        return self

    # -- Primitives (non-blocking, return self) --------------------------------

    def init(self, gpu: int):
        self.gpu = gpu
        self._log("init")
        self._worker = _fork_ctx.Process(
            target=worker_loop,
            args=(gpu, self._cmd_queue, self._result_queue,
                  self._completed_counter, self._use_odirect),
        )
        self._worker.start()
        return self._send("init", vllm_config=self.vllm_config)

    def attach(self):
        self._log("attach")
        return self._send("attach", model_path=self.vllm_config["model"])

    def detach(self):
        self._log("detach")
        return self._send("detach")

    def sleep(self):
        self._log("sleep")
        return self._send("sleep")

    def checkpoint(self):
        self._log("checkpoint")
        return self._send("checkpoint")

    def restore(self, gpu: int):
        self._log(f"restore(gpu={gpu})")
        return self._send("restore", gpu=gpu)

    def stage(self, data_path=None):
        if data_path is not None:
            data_arch = _read_architecture(data_path)
            if data_arch != self.arch:
                raise ValueError(
                    f"architecture mismatch: instance arch {self.arch} "
                    f"!= data arch {data_arch} at {data_path}")
        self._log("stage")
        return self._send("stage", data_path=data_path)

    def wake_up(self, tags):
        self._log(f"wake_up({tags})")
        if tags == ["weights"]:
            return self._send("wake_up_weights")
        elif tags == ["kv_cache"]:
            return self._send("wake_up_kv_cache")
        else:
            raise ValueError(f"unsupported wake_up tags: {tags}")

    def h2d(self):
        self._log("h2d")
        return self._send("h2d")

    def scatter(self):
        self._log("scatter")
        return self._send("scatter")

    def teardown(self):
        self._log("teardown")
        return self._send("teardown")

    def remove(self):
        """Remove from the instance registry."""
        self._log("remove")
        Instance._all.pop(self.instance_id, None)

    # -- Cross-instance dependency ---------------------------------------------

    def after(self, other: Instance):
        """Wait for all of other's currently pending commands before proceeding.

        Non-blocking from the main process.  The worker blocks until
        other._completed_counter >= target.

        We send only the instance ID (not the Value itself) through the
        queue.  The worker resolves it via _counter_registry, which it
        inherited at fork time.
        """
        target = other._total_sent
        self._log(f"after(inst{other.instance_id}, target={target})")
        self._cmd_queue.put(("wait_for", {
            "instance_id": other.instance_id,
            "target": target,
        }))
        return self

    # -- Synchronization -------------------------------------------------------

    def wait(self):
        """Block until all pending commands complete for this instance.

        Raises RuntimeError on the first command that failed.
        """
        self._log(f"wait ({self._pending_count} pending)")

        while self._pending_count > 0:
            result = self._result_queue.get()
            cmd, elapsed, error, info = result

            self._pending_count -= 1
            if self._pending_cmds:
                self._pending_cmds.pop(0)

            status = "OK" if error is None else "FAILED"
            self._print(f"[gpu{self.gpu}] [{time.strftime('%H:%M:%S')}] "
                         f"{cmd} {status} ({elapsed:.3f}s) {info}")

            if error is None:
                if cmd == "init":
                    self.pid = info.get("pid")
                    self.state = "alive"
                elif cmd == "attach":
                    self.pinned_bytes = info.get("pinned_bytes", self.pinned_bytes)
                elif cmd == "detach":
                    self.pinned_bytes = 0
                elif cmd == "checkpoint":
                    self.gpu = None
                    self.state = "checkpointed"
                elif cmd == "restore":
                    self.gpu = info.get("gpu", self.gpu)
                    self.state = "alive"
                elif cmd == "teardown":
                    self._reset()

            if error is not None:
                raise RuntimeError(f"GPU {self.gpu} command '{cmd}' failed: {error}")

        return self

    # -- Status ----------------------------------------------------------------

    def _sync_state(self):
        """Drain completed results from the worker without blocking.

        Updates local state to reflect what the worker has actually finished,
        using _completed_counter to know how many results are available.
        """
        completed = self._completed_counter.value
        available = completed - (self._total_sent - self._pending_count)
        for _ in range(available):
            try:
                result = self._result_queue.get_nowait()
            except Exception:
                break
            cmd, elapsed, error, info = result
            self._pending_count -= 1
            if self._pending_cmds:
                self._pending_cmds.pop(0)
            if error is None:
                if cmd == "init":
                    self.pid = info.get("pid")
                    self.state = "alive"
                elif cmd == "attach":
                    self.pinned_bytes = info.get("pinned_bytes", self.pinned_bytes)
                elif cmd == "detach":
                    self.pinned_bytes = 0
                elif cmd == "checkpoint":
                    self.gpu = None
                    self.state = "checkpointed"
                elif cmd == "restore":
                    self.gpu = info.get("gpu", self.gpu)
                    self.state = "alive"
                elif cmd == "teardown":
                    self._reset()

    @classmethod
    def print_status(cls):
        from collections import defaultdict
        by_gpu = defaultdict(list)
        unassigned = []
        for inst in cls._all.values():
            inst._sync_state()
            if inst.gpu is None:
                unassigned.append(inst)
            else:
                by_gpu[inst.gpu].append(inst)

        num_gpus = _gpu_count()
        gpu_mem = {}
        for g in range(num_gpus):
            gpu_mem[g] = _gpu_mem_info(g)

        print(f"\n{'=' * 80}", flush=True)
        print(f"  Instance Status  [{time.strftime('%H:%M:%S')}]", flush=True)
        print(f"{'=' * 80}", flush=True)
        for gpu in range(num_gpus):
            free, total = gpu_mem[gpu]
            used = total - free
            print(f"  GPU {gpu}:  {used / 2**30:.2f} GiB / {total / 2**30:.2f} GiB used  "
                  f"({free / 2**30:.2f} GiB free)", flush=True)
            for inst in by_gpu[gpu]:
                marker = "*" if inst.state == "alive" else " "
                model = inst.vllm_config["model"].split("/")[-1]
                model_w = f"{model} ({inst.weight_bytes / 2**30:.2f} GiB)"
                pinned = f"pinned={inst.pinned_bytes / 2**30:.2f} GiB"
                pending = inst._pending_cmds or []
                print(f"    [{marker}] inst{inst.instance_id:<3} {inst.state:<15} "
                      f"{model_w:<45} {pinned:<18} "
                      f"pending={pending}", flush=True)
        if unassigned:
            print(f"  Unassigned:", flush=True)
            for inst in unassigned:
                model = inst.vllm_config["model"].split("/")[-1]
                model_w = f"{model} ({inst.weight_bytes / 2**30:.2f} GiB)"
                pinned = f"pinned={inst.pinned_bytes / 2**30:.2f} GiB"
                pending = inst._pending_cmds or []
                print(f"    [ ] inst{inst.instance_id:<3} {inst.state:<15} "
                      f"{model_w:<45} {pinned:<18} "
                      f"pending={pending}", flush=True)
        print(f"{'=' * 80}\n", flush=True)

    # -- Logging ---------------------------------------------------------------

    def _print(self, msg):
        print(msg, flush=True)

    def _log(self, cmd):
        self._print(f"[inst{self.instance_id}] [{time.strftime('%H:%M:%S')}] (pid={os.getpid()}) "
                     f"{cmd} -> gpu{self.gpu} (pending={self._pending_cmds})")
