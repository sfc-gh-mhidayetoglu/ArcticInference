"""Lightweight orchestrator for managing multiple vLLM instances by model ID."""
from __future__ import annotations

import os
import subprocess
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from huggingface_hub import snapshot_download

from instance import Instance

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


@dataclass
class GPUView:
    gpu_id: int
    name: str
    free_bytes: int
    total_bytes: int


def _discover_gpus() -> list[GPUView]:
    """Discover GPUs via nvidia-smi without initializing CUDA."""
    out = subprocess.check_output(
        ["nvidia-smi",
         "--query-gpu=index,name,memory.free,memory.total",
         "--format=csv,noheader,nounits"],
        text=True,
    )
    gpus = []
    for line in out.strip().splitlines():
        idx, name, free_mib, total_mib = (x.strip() for x in line.split(","))
        gpus.append(GPUView(
            gpu_id=int(idx),
            name=name,
            free_bytes=int(free_mib) * 1024 * 1024,
            total_bytes=int(total_mib) * 1024 * 1024,
        ))
    return gpus


class Orchestrator:
    """Static registry that maps human-readable model IDs to Instances."""

    _registry: dict[str, Any] = {}
    _futures: dict[str, Future] = {}
    _gpus: list[GPUView] = []
    _assigned_gpus: set[int] = set()
    _local_cache: str | None = None
    _pool: ThreadPoolExecutor | None = None
    _gpu_available = threading.Condition()
    _attach_lock = threading.Lock()

    @staticmethod
    def _acquire_gpu(label: str) -> int:
        """Block until a GPU is free, assign it, and return its ID."""
        announced = False
        with Orchestrator._gpu_available:
            while True:
                available = [g for g in Orchestrator._gpus
                             if g.gpu_id not in Orchestrator._assigned_gpus]
                if available:
                    gpu = min(g.gpu_id for g in available)
                    Orchestrator._assigned_gpus.add(gpu)
                    _console(f"{label}: acquired GPU {gpu}")
                    print(f"[orchestrator] {label}: acquired GPU {gpu}")
                    return gpu
                if not announced:
                    _console(f"{label}: waiting for GPU...")
                    announced = True
                print(f"[orchestrator] {label}: waiting for GPU ...")
                Orchestrator._gpu_available.wait()

    @staticmethod
    def _release_gpu(gpu: int) -> None:
        """Release a GPU and wake up any waiters."""
        with Orchestrator._gpu_available:
            Orchestrator._assigned_gpus.discard(gpu)
            Orchestrator._gpu_available.notify_all()

    @staticmethod
    def init(local_cache: str = "/data-fast/model-cache",
             gpu_carveout_gb: float = 0) -> None:
        """Discover GPUs, create cache dir, and start the thread pool."""
        Orchestrator._gpus = _discover_gpus()
        Orchestrator._local_cache = local_cache
        os.makedirs(local_cache, exist_ok=True)
        Orchestrator._pool = ThreadPoolExecutor()
        print(f"[orchestrator] init  local_cache={local_cache}  "
              f"gpus={[g.gpu_id for g in Orchestrator._gpus]}")

    # ------------------------------------------------------------------
    # register
    # ------------------------------------------------------------------

    @staticmethod
    def register(model_id: str, vllm_config: dict) -> None:
        """Submit model download + GPU registration to the pool."""
        _console(f"{model_id}: register received")
        print(f"[orchestrator] register  model_id={model_id}  "
              f"model={vllm_config['model']}")

        fut = Orchestrator._pool.submit(
            Orchestrator._register_sync, model_id, vllm_config,
        )
        Orchestrator._futures[model_id] = fut

    @staticmethod
    def _register_sync(model_id: str, vllm_config: dict) -> None:
        """Download model then do blocking GPU registration (runs in a pool thread)."""
        hf_model = vllm_config["model"]
        local_dir = os.path.join(Orchestrator._local_cache, hf_model)

        t0 = time.perf_counter()
        print(f"[orchestrator] {model_id}: downloading {hf_model} ...")
        snapshot_download(hf_model, local_dir=local_dir)
        t_dl = time.perf_counter() - t0
        print(f"[orchestrator] {model_id}: download done  ({t_dl:.1f}s)")

        vllm_config = dict(vllm_config, model=local_dir)
        gpu = Orchestrator._acquire_gpu(model_id)
        t_acquired = time.perf_counter()

        inst = Instance(vllm_config).init(gpu)
        with Orchestrator._attach_lock:
            inst.attach()
        inst.stage().sleep().checkpoint().wait()

        t_done = time.perf_counter()
        t_wait = t_acquired - t0
        t_exec = t_done - t_acquired
        print(f"[orchestrator] {model_id}: cold-start done on GPU {gpu}  "
              f"({t_done - t0:.1f}s)")

        Orchestrator._release_gpu(gpu)

        Orchestrator._registry[model_id] = {
            "instance": inst,
            "vllm_config": vllm_config,
        }
        print(f"[orchestrator] {model_id}: registered  ({t_done - t0:.1f}s total)")
        _console(f"{model_id}: registered (wait={t_wait:.1f}s, register={t_exec:.1f}s, total={t_wait + t_exec:.1f}s)")

    # ------------------------------------------------------------------
    # generate
    # ------------------------------------------------------------------

    @staticmethod
    def generate(model_id: str, prompts: list[str],
                 sampling_params: dict) -> Future:
        """Submit a non-blocking generate.  Returns a Future[list]."""
        _console(f"{model_id}: generate received")
        print(f"[orchestrator] generate  model_id={model_id}")
        prev = Orchestrator._futures.get(model_id)
        fut = Orchestrator._pool.submit(
            Orchestrator._generate_sync, model_id, prompts, sampling_params,
            prev,
        )
        Orchestrator._futures[model_id] = fut
        return fut

    @staticmethod
    def _generate_sync(model_id: str, prompts: list[str],
                       sampling_params: dict,
                       prev_future=None) -> list:
        """Blocking generate (runs in a pool thread).

        Acquires a GPU, restores the checkpointed process onto it,
        loads weights from CPU→GPU, runs inference, then re-checkpoints
        to free the GPU.
        """
        if prev_future is not None:
            prev_future.result()

        t0 = time.perf_counter()
        entry = Orchestrator._registry[model_id]
        inst = entry["instance"]

        gpu = Orchestrator._acquire_gpu(model_id)
        t_acquired = time.perf_counter()

        inst.restore(gpu) \
            .wake_up(["weights"]) \
            .h2d() \
            .scatter() \
            .wake_up(["kv_cache"]) \
            .wait()
        t_ready = time.perf_counter()

        inst.generate(prompts, sampling_params).wait()
        result = inst.last_generate_result
        t_generated = time.perf_counter()

        inst.sleep().checkpoint().wait()
        Orchestrator._release_gpu(gpu)
        t_done = time.perf_counter()

        t_wait = t_acquired - t0
        t_restore = t_ready - t_acquired
        t_gen = t_generated - t_ready
        t_ckpt = t_done - t_generated
        print(f"[orchestrator] {model_id}: generate done  ({t_done - t0:.1f}s)")

        snippet = ""
        if result:
            try:
                snippet = result[0][0].replace("\n", " ")[:100]
            except (TypeError, IndexError):
                snippet = str(result)[:100]
        t_total = t_wait + t_restore + t_gen + t_ckpt
        _console(f"{model_id}: generated "
                 f"(wait={t_wait:.1f}s, restore={t_restore:.1f}s, "
                 f"generate={t_gen:.1f}s, checkpoint={t_ckpt:.1f}s, total={t_total:.1f}s) "
                 f"-> \"{snippet}\"")

        return result

    # ------------------------------------------------------------------
    # wait / remove / status
    # ------------------------------------------------------------------

    @staticmethod
    def wait(model_id: str | None = None) -> None:
        """Block until futures complete.  None = wait on all."""
        label = model_id or "all"
        print(f"[orchestrator] wait  model_id={label}")
        t0 = time.perf_counter()
        if model_id is not None:
            Orchestrator._futures[model_id].result()
        else:
            for fut in list(Orchestrator._futures.values()):
                fut.result()
        elapsed = time.perf_counter() - t0
        print(f"[orchestrator] wait done  ({elapsed:.1f}s)")

    @staticmethod
    def remove(model_id: str | None = None) -> None:
        """Remove a model, or all models if *model_id* is None."""
        if model_id is None:
            for mid in list(Orchestrator._registry):
                Orchestrator.remove(mid)
            return
        _console(f"{model_id}: remove received")
        prev = Orchestrator._futures.get(model_id)
        fut = Orchestrator._pool.submit(
            Orchestrator._remove_sync, model_id, prev,
        )
        Orchestrator._futures[model_id] = fut

    @staticmethod
    def _remove_sync(model_id: str, prev_future=None) -> None:
        if prev_future is not None:
            prev_future.result()
        entry = Orchestrator._registry.pop(model_id, None)
        if entry is not None:
            inst = entry["instance"]
            inst.teardown().wait().remove()
            _console(f"{model_id}: removed")
        Orchestrator._futures.pop(model_id, None)

    @staticmethod
    def print_status() -> None:
        """Print GPUs and registered models."""
        if not Orchestrator._gpus:
            _console("Orchestrator not initialized. Call Orchestrator.init() first.")
            return

        _console(f"\nOrchestrator  local_cache={Orchestrator._local_cache}")

        _console(f"\nGPUs ({len(Orchestrator._gpus)}):")
        for g in Orchestrator._gpus:
            used = g.total_bytes - g.free_bytes
            _console(f"  GPU {g.gpu_id}: {g.name}  "
                     f"{used / 2**30:.1f} / {g.total_bytes / 2**30:.1f} GiB used  "
                     f"({g.free_bytes / 2**30:.1f} GiB free)")

        if not Orchestrator._registry:
            _console("\nNo models registered.\n")
            return
        max_id = max(len(mid) for mid in Orchestrator._registry)
        _console(f"\nModels ({len(Orchestrator._registry)}):")
        for model_id, entry in Orchestrator._registry.items():
            inst = entry["instance"]
            hf_model = entry["vllm_config"].get("model", "?")
            _console(f"  {model_id:<{max_id}}  {hf_model}")
            _console(f"  {' ' * max_id}  {inst}")
        _console("")
