"""Standalone Instance for a vLLM engine.

Each Instance is a GPU-agnostic handle.  The GPU is specified at
init(gpu) time, which also spawns the worker process.  All primitives
are non-blocking and return self for chaining.

Instances can be saved to disk via CRIU after CUDA checkpoint, then
restored (possibly on a different GPU):

    inst.unpin().sleep().checkpoint_cuda().save_image(filename="/data-fast/ckpt/m").wait()
    inst.restore_cuda(gpu=2).wake_up_weights().repin() \
        .plan_load_weights().load_weights().wake_up_kv_cache().wait()

On a later run, load_image() restores from the on-disk image:

    inst = Instance(vllm_config)
    inst.load_image("/data-fast/ckpt/m").plan_load_weights().wait()
    inst.restore_cuda(gpu=0).wake_up_weights().repin() \
        .load_weights().wake_up_kv_cache().wait()
"""
from __future__ import annotations

import json
import os
import threading
import time
import weakref

import pynvml
import torch.multiprocessing as mp

from worker import worker_loop

_spawn_ctx = mp.get_context("spawn")

_next_instance_id = 0
_id_lock = threading.Lock()


def _alloc_instance_id():
    global _next_instance_id
    with _id_lock:
        _next_instance_id += 1
        return _next_instance_id - 1


class Instance:

    _all: weakref.WeakValueDictionary[int, "Instance"] = weakref.WeakValueDictionary()

    def __init__(self, vllm_config: dict):
        self.gpu = None
        self.vllm_config = vllm_config
        self.instance_id = _alloc_instance_id()
        Instance._all[self.instance_id] = self

        self.pid = None
        self.state = "created"
        self.pinned_cpu_bytes = 0
        self.total_gpu_bytes = 0
        self._image_dir = None
        self._pending_count = 0
        self._pending_cmds = []
        self._total_sent = 0

        self._cmd_queue = None
        self._result_queue = None
        self._completed_counter = None
        self._worker = None
        self._next_req_id = 0
        self.last_generate_result = None
        self.last_prompt_tokens = None
        self.last_completion_tokens = None
        self.generate_results = {}  # req_id -> {outputs, prompt_tokens, completion_tokens}
        self._external_waiter = False  # set True when orchestrator waiter owns the queue

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
                 f"pid={self.pid}",
                 f"pinned_cpu={self.pinned_cpu_bytes / 2**30:.2f} GiB"]
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
        self.pinned_cpu_bytes = 0

    def _send(self, cmd, **kwargs):
        self._cmd_queue.put((cmd, kwargs))
        self._pending_count += 1
        self._pending_cmds.append(cmd)
        self._total_sent += 1
        return self

    # -- Primitives (non-blocking, return self) --------------------------------

    def init(self, gpu: int):
        self.gpu = gpu
        # Snapshot the GPU's physical capacity once per instance lifetime.
        # Safe under the orchestrator contract that init always takes a
        # full L1 slot (no shared tenants), so .total matches what vLLM
        # uses internally for its gpu_memory_utilization budget.
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(gpu)
        self.total_gpu_bytes = int(pynvml.nvmlDeviceGetMemoryInfo(h).total)
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

    def checkpoint_cuda(self):
        self._log("checkpoint_cuda")
        return self._send("checkpoint_cuda")

    def save_image(self, filename: str):
        """CRIU-dump the child process tree to disk (destructive).

        Must be called after checkpoint_cuda() (GPU resources released).
        The child process is killed after a successful dump.  The
        on-disk image is later restored via load_image().
        """
        self._log(f"save_image({filename})")
        self._image_dir = filename
        return self._send(
            "save_image", filename=filename,
            meta_extra={"vllm_config":      self.vllm_config,
                        "total_gpu_bytes":  self.total_gpu_bytes,
                        "pinned_cpu_bytes": self.pinned_cpu_bytes})

    def load_image(self, filename: str | None = None):
        """Restore a live process from a CRIU image on disk.

        If filename is None, uses the image from the last save_image().
        Validates that the image's vllm_config matches this instance's
        config (raises RuntimeError on mismatch).  Spawns a new worker
        and CRIU-restores the child.  After load_image completes the
        instance is in 'checkpointed' state, ready for restore_cuda(gpu).
        """
        if filename is None:
            filename = self._image_dir
        if filename is None:
            raise RuntimeError("load_image() requires a filename or prior save_image()")
        meta_path = os.path.join(filename, "meta.json")
        if os.path.isfile(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            saved_config = meta.get("vllm_config")
            if saved_config is not None and saved_config != self.vllm_config:
                raise RuntimeError(
                    f"vllm_config mismatch: instance has {self.vllm_config} "
                    f"but image at {filename} was saved with {saved_config}")
            # Hydrate budget inputs from meta.json; the child holds the
            # real pinned buffer that survived CRIU.  Old images without
            # ``total_gpu_bytes`` degrade to single-chunk behavior in
            # plan_load_weights().  Fall back to the legacy
            # ``pinned_bytes`` key for one release.
            self.total_gpu_bytes = int(meta.get("total_gpu_bytes", 0))
            self.pinned_cpu_bytes = int(meta.get(
                "pinned_cpu_bytes", meta.get("pinned_bytes", 0)))
        self._log(f"load_image({filename})")
        self._image_dir = filename
        self._close_queues()
        self._ensure_queues()
        self._worker = _spawn_ctx.Process(
            target=worker_loop,
            args=(0, self._cmd_queue, self._result_queue,
                  self._completed_counter),
        )
        self._worker.start()
        return self._send("load_image", filename=filename)

    def restore_cuda(self, gpu: int):
        self._log(f"restore_cuda(gpu={gpu})")
        return self._send("restore_cuda", gpu=gpu)

    def stage(self):
        self._log("stage")
        return self._send("stage")

    def wake_up_weights(self):
        self._log("wake_up_weights")
        return self._send("wake_up_weights")

    def wake_up_kv_cache(self):
        self._log("wake_up_kv_cache")
        return self._send("wake_up_kv_cache")

    def plan_load_weights(self):
        """Precompute the chunk plan that the next load_weights() will consume.

        Self-computes the staging budget from instance state populated by
        ``init`` (cold start) or by ``load`` reading meta.json (restore):

            allotment = self.total_gpu_bytes * gpu_memory_utilization
            budget    = min(self.pinned_cpu_bytes,
                            allotment - self.pinned_cpu_bytes)

        The formula is self-validating: if ``budget`` ends up smaller
        than the largest single parameter, the child's plan walk raises
        with a precise ``param X exceeds chunk_size`` message.

        If ``self.total_gpu_bytes`` or ``self.pinned_cpu_bytes`` is
        missing (legacy meta.json, or not yet attached), passes
        ``max_buffer_bytes=None`` to the child, yielding the single-chunk
        fallback (today's behavior).
        """
        if self.total_gpu_bytes <= 0 or self.pinned_cpu_bytes <= 0:
            mb = None
        else:
            util = self.vllm_config.get("gpu_memory_utilization", 0.7)
            allotment = int(self.total_gpu_bytes * util)
            mb = min(self.pinned_cpu_bytes,
                     allotment - self.pinned_cpu_bytes)
        self._log(f"plan_load_weights(max_buffer_bytes={mb})")
        return self._send("plan_load_weights", max_buffer_bytes=mb)

    def load_weights(self):
        """Copy staged weights from pinned CPU into model parameters.

        Pure execution against the chunk plan cached by a prior
        ``plan_load_weights()``.  For each chunk, the worker copies a
        slice of the pinned buffer to a single reused GPU staging buffer
        (PCIe H2D) and then scatters into ``model.named_parameters()``
        in place.  If no plan is cached (paths that skip
        ``plan_load_weights``), falls back to a single-chunk path
        identical to the prior unbounded behavior.

        Requires a prior ``attach() -> ... -> stage()`` to have populated
        the pinned buffer, and ``wake_up_weights()`` to have allocated
        the destination parameter tensors.
        """
        self._log("load_weights")
        return self._send("load_weights")

    def generate(self, prompts, sampling_params):
        self._log(f"generate({len(prompts)} prompts)")
        req_id = f"inst{self.instance_id}-{self._next_req_id}"
        self._next_req_id += 1
        self.last_req_id = req_id
        return self._send("generate", req_id=req_id, prompts=prompts,
                           sampling_params=sampling_params)

    def teardown(self):
        self._log("teardown")
        return self._send("teardown")

    def remove(self):
        """Remove from the instance registry."""
        self._log("remove")
        Instance._all.pop(self.instance_id, None)

    # -- Synchronization -------------------------------------------------------

    def _apply_result(self, cmd: str, info: dict) -> None:
        """Update local state after a successful command completion."""
        if cmd == "init":
            self.pid = info.get("pid")
            self.state = "alive"
        elif cmd == "attach":
            self.pinned_cpu_bytes = info.get(
                "pinned_cpu_bytes", self.pinned_cpu_bytes)
        elif cmd == "detach":
            self.pinned_cpu_bytes = 0
        elif cmd == "checkpoint_cuda":
            self.gpu = None
            self.state = "checkpointed"
        elif cmd == "save_image":
            self._image_dir = info.get("image_dir", self._image_dir)
            self.state = "checkpointed"
        elif cmd == "load_image":
            self.pid = info.get("pid")
            self.gpu = None
            self.state = "checkpointed"
        elif cmd == "restore_cuda":
            self.gpu = info.get("gpu", self.gpu)
            self.state = "alive"
        elif cmd == "generate":
            self.last_generate_result = info.get("outputs")
            self.last_prompt_tokens = info.get("prompt_tokens")
            self.last_completion_tokens = info.get("completion_tokens")
            req_id = info.get("req_id")
            if req_id is not None:
                self.generate_results[req_id] = {
                    "outputs": info.get("outputs"),
                    "prompt_tokens": info.get("prompt_tokens"),
                    "completion_tokens": info.get("completion_tokens"),
                }
        elif cmd == "teardown":
            self._reset()

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
            display_info = {k: v for k, v in info.items() if k != "outputs"} if cmd == "generate" else info
            self._print(f"[gpu{self.gpu}] [{time.strftime('%H:%M:%S')}] "
                         f"{cmd} {status} ({elapsed:.3f}s) {display_info}")

            if error is None:
                self._apply_result(cmd, info)

            if error is not None:
                raise RuntimeError(f"GPU {self.gpu} command '{cmd}' failed: {error}")

        return self

    def _sync_state(self):
        """Drain completed results from the worker without blocking.

        Updates local state to reflect what the worker has actually finished,
        using _completed_counter to know how many results are available.
        Skipped when an external waiter (orchestrator) owns the queue.
        """
        if self._external_waiter:
            return
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
                self._apply_result(cmd, info)

    # -- Logging ---------------------------------------------------------------

    def _print(self, msg):
        print(msg, flush=True)

    def _log(self, cmd):
        self._print(f"[inst{self.instance_id}] [{time.strftime('%H:%M:%S')}] (pid={os.getpid()}) "
                     f"{cmd} -> gpu{self.gpu} (pending={self._pending_cmds})")
