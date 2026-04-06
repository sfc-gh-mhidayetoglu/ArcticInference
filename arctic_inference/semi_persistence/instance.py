"""Standalone Instance for a vLLM engine.

Each Instance is a GPU-agnostic handle.  The GPU is specified at
init(gpu) time, which also spawns the worker process.  All primitives
are non-blocking and return self for chaining.

Instances can be saved to disk via CRIU after CUDA checkpoint, then
restored (possibly on a different GPU):

    inst.unpin().sleep().checkpoint().save(filename="/data-fast/ckpt/m").wait()
    inst.restore(gpu=2).wake_up_weights().repin().h2d().scatter().wake_up_kv_cache().wait()

On a later run, load() restores from the on-disk image:

    inst = Instance(vllm_config)
    inst.load("/data-fast/ckpt/m").wait()
    inst.restore(gpu=0).wake_up_weights().repin().h2d().scatter().wake_up_kv_cache().wait()
"""
from __future__ import annotations

import json
import os
import time
import weakref

import torch.multiprocessing as mp

from worker import worker_loop, _weight_footprint

_spawn_ctx = mp.get_context("spawn")


def _print_process_tree(pid, indent=0):
    """Print the process tree rooted at *pid* using /proc."""
    prefix = " " * indent
    try:
        with open(f"/proc/{pid}/status") as f:
            name = comm = ""
            threads = 0
            state = ""
            for line in f:
                if line.startswith("Name:"):
                    name = line.split(":", 1)[1].strip()
                elif line.startswith("State:"):
                    state = line.split(":", 1)[1].strip()
                elif line.startswith("Threads:"):
                    threads = int(line.split(":", 1)[1].strip())
        with open(f"/proc/{pid}/cmdline") as f:
            raw = f.read()
        cmdline = raw.replace("\x00", " ").strip()
        if len(cmdline) > 60:
            cmdline = cmdline[:57] + "..."
        thr_info = f" [{threads} threads]" if threads > 1 else ""
        print(f"{prefix}{pid}  {state}  {name}{thr_info}  {cmdline}", flush=True)
    except (FileNotFoundError, ProcessLookupError):
        print(f"{prefix}{pid}  (not running)", flush=True)
        return

    try:
        with open(f"/proc/{pid}/task/{pid}/children") as f:
            children = [int(c) for c in f.read().split() if c]
    except (FileNotFoundError, ProcessLookupError):
        children = []
    for child in children:
        _print_process_tree(child, indent=indent + 4)


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


def _alloc_instance_id():
    global _next_instance_id
    _next_instance_id += 1
    return _next_instance_id - 1


class Instance:

    _all: weakref.WeakValueDictionary[int, "Instance"] = weakref.WeakValueDictionary()

    def __init__(self, vllm_config: dict):
        self.gpu = None
        self.vllm_config = vllm_config
        self.instance_id = _alloc_instance_id()
        Instance._all[self.instance_id] = self

        model_path = vllm_config["model"]
        self.weight_bytes = _weight_footprint(model_path)

        self.pid = None
        self.state = "created"
        self.pinned_bytes = 0
        self._image_dir = None
        self._pending_count = 0
        self._pending_cmds = []
        self._total_sent = 0

        self._cmd_queue = None
        self._result_queue = None
        self._completed_counter = None
        self._worker = None
        self.last_generate_result = None

    def _ensure_queues(self):
        """Create mp queues/counter on demand, right before spawning a worker."""
        if self._cmd_queue is None:
            self._cmd_queue = _spawn_ctx.Queue()
            self._result_queue = _spawn_ctx.Queue()
            self._completed_counter = _spawn_ctx.Value('i', 0)
            self._pending_count = 0
            self._pending_cmds = []
            self._total_sent = 0

    def __repr__(self):
        parts = [f"id={self.instance_id}", f"gpu={self.gpu}",
                 f"pid={self.pid}", f"state={self.state}",
                 f"pinned={self.pinned_bytes / 2**30:.2f} GiB"]
        if self._image_dir:
            parts.append(f"image={self._image_dir}")
        return f"Instance({', '.join(parts)})"

    # -- Internal helpers -------------------------------------------------------

    def _close_queues(self):
        """Deterministically close mp queues so semaphores are released now."""
        for q in (self._cmd_queue, self._result_queue):
            if q is not None:
                try:
                    q.close()
                    q.join_thread()
                except Exception:
                    pass
        self._cmd_queue = None
        self._result_queue = None
        self._completed_counter = None

    def _reset(self):
        """Reset instance to created state after teardown completes."""
        if self._worker is not None:
            self._worker.join(timeout=10)
            if self._worker.is_alive():
                self._print(f"[inst{self.instance_id}] worker still alive after join, force-killing")
                try:
                    import signal
                    os.kill(self._worker.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self._worker.join(timeout=5)
            self._worker = None
        self._close_queues()
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
        self._ensure_queues()
        self._worker = _spawn_ctx.Process(
            target=worker_loop,
            args=(gpu, self._cmd_queue, self._result_queue,
                  self._completed_counter),
        )
        self._worker.start()
        return self._send("init", vllm_config=self.vllm_config)

    def attach(self):
        self._log("attach")
        return self._send("attach")

    def detach(self):
        self._log("detach")
        return self._send("detach")

    def unpin(self):
        self._log("unpin")
        return self._send("unpin")

    def repin(self):
        self._log("repin")
        return self._send("repin")

    def sleep(self):
        self._log("sleep")
        return self._send("sleep")

    def checkpoint(self):
        self._log("checkpoint")
        return self._send("checkpoint")

    def save(self, filename: str):
        """CRIU-dump the child process tree to disk (non-destructive).

        Must be called after checkpoint() (GPU resources released).
        Uses --leave-running so the child stays alive after the dump.
        After save(), the instance remains in 'checkpointed' state and
        can proceed directly to restore(gpu).  The on-disk image can
        also be used later via load() on a fresh instance.
        """
        self._log(f"save({filename})")
        self._image_dir = filename
        return self._send("save", filename=filename,
                          meta_extra={"vllm_config": self.vllm_config,
                                      "weight_bytes": self.weight_bytes,
                                      "pinned_bytes": self.pinned_bytes})

    def load(self, filename: str | None = None):
        """Restore a live process from a CRIU image on disk.

        If filename is None, uses the image from the last save().
        Validates that the image's vllm_config matches this instance's
        config (raises RuntimeError on mismatch).  Spawns a new worker
        and CRIU-restores the child.  After load completes the instance
        is in 'checkpointed' state, ready for restore(gpu).
        """
        if filename is None:
            filename = self._image_dir
        if filename is None:
            raise RuntimeError("load() requires a filename or prior save()")
        meta_path = os.path.join(filename, "meta.json")
        if os.path.isfile(meta_path):
            with open(meta_path) as f:
                saved_config = json.load(f).get("vllm_config")
            if saved_config is not None and saved_config != self.vllm_config:
                raise RuntimeError(
                    f"vllm_config mismatch: instance has {self.vllm_config} "
                    f"but image at {filename} was saved with {saved_config}")
        self._log(f"load({filename})")
        self._image_dir = filename
        self._close_queues()
        self._ensure_queues()
        self._worker = _spawn_ctx.Process(
            target=worker_loop,
            args=(0, self._cmd_queue, self._result_queue,
                  self._completed_counter),
        )
        self._worker.start()
        return self._send("load", filename=filename)

    def restore(self, gpu: int):
        self._log(f"restore(gpu={gpu})")
        return self._send("restore", gpu=gpu)

    def stage(self):
        self._log("stage")
        return self._send("stage")

    def wake_up_weights(self):
        self._log("wake_up_weights")
        return self._send("wake_up_weights")

    def wake_up_kv_cache(self):
        self._log("wake_up_kv_cache")
        return self._send("wake_up_kv_cache")

    def h2d(self):
        self._log("h2d")
        return self._send("h2d")

    def scatter(self):
        self._log("scatter")
        return self._send("scatter")

    def generate(self, prompts, sampling_params):
        self._log(f"generate({len(prompts)} prompts)")
        return self._send("generate", prompts=prompts,
                           sampling_params=sampling_params)

    def teardown(self):
        self._log("teardown")
        return self._send("teardown")

    def remove(self):
        """Remove from the instance registry."""
        self._log("remove")
        Instance._all.pop(self.instance_id, None)

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
                elif cmd == "save":
                    self._image_dir = info.get("image_dir", self._image_dir)
                    self.state = "checkpointed"
                elif cmd == "load":
                    self.pid = info.get("pid")
                    self.gpu = None
                    self.state = "checkpointed"
                elif cmd == "restore":
                    self.gpu = info.get("gpu", self.gpu)
                    self.state = "alive"
                elif cmd == "generate":
                    self.last_generate_result = info.get("outputs")
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
                elif cmd == "save":
                    self._image_dir = info.get("image_dir", self._image_dir)
                    self.state = "checkpointed"
                elif cmd == "load":
                    self.pid = info.get("pid")
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
                cls._print_instance(inst, marker="*" if inst.state == "alive" else " ")
        if unassigned:
            print(f"  Unassigned:", flush=True)
            for inst in unassigned:
                cls._print_instance(inst, marker=" ")
        print(f"{'=' * 80}\n", flush=True)

    @staticmethod
    def _print_instance(inst, marker=" "):
        model = inst.vllm_config["model"].split("/")[-1]
        model_w = f"{model} ({inst.weight_bytes / 2**30:.2f} GiB)"
        pinned = f"pinned={inst.pinned_bytes / 2**30:.2f} GiB"
        pending = inst._pending_cmds or []
        print(f"    [{marker}] inst{inst.instance_id:<3} {inst.state:<15} "
              f"{model_w:<45} {pinned:<18} "
              f"pending={pending}", flush=True)
        if inst.pid is not None:
            _print_process_tree(inst.pid, indent=10)

    # -- Logging ---------------------------------------------------------------

    def _print(self, msg):
        print(msg, flush=True)

    def _log(self, cmd):
        self._print(f"[inst{self.instance_id}] [{time.strftime('%H:%M:%S')}] (pid={os.getpid()}) "
                     f"{cmd} -> gpu{self.gpu} (pending={self._pending_cmds})")
