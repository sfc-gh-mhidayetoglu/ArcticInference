"""Lightweight orchestrator for managing multiple vLLM instances by model ID."""
from __future__ import annotations

import os
import subprocess
import queue
import time
from concurrent.futures import Future, ThreadPoolExecutor
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


def _discover_gpu_ids() -> list[int]:
    """Return GPU device indices via nvidia-smi without initializing CUDA."""
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
        text=True,
    )
    return [int(line.strip()) for line in out.strip().splitlines()]


class Orchestrator:
    """Static registry that maps human-readable model IDs to Instances."""

    _registry: dict[str, Any] = {}
    _futures: dict[str, Future] = {}
    _gpu_ids: list[int] = []
    _model_cache: str | None = None
    _image_cache: str | None = None
    _pool: ThreadPoolExecutor | None = None
    _gpu_queue: queue.Queue[int] = queue.Queue()

    @staticmethod
    def _acquire_gpu(label: str) -> int:
        """Block (FIFO) until a GPU is free and return its ID."""
        _console(f"{label}: waiting for GPU...")
        print(f"[orchestrator] {label}: waiting for GPU ...")
        gpu = Orchestrator._gpu_queue.get()
        _console(f"{label}: acquired GPU {gpu}")
        print(f"[orchestrator] {label}: acquired GPU {gpu}")
        return gpu

    @staticmethod
    def _release_gpu(gpu: int) -> None:
        """Return a GPU to the pool."""
        Orchestrator._gpu_queue.put(gpu)

    @staticmethod
    def init(model_cache: str = "/data-fast/model-cache",
             image_cache: str = "/data-fast/image-cache",
             gpus: list[int] | None = None) -> None:
        """Discover GPUs, create cache dirs, and start the thread pool."""
        Orchestrator._gpu_ids = gpus if gpus is not None else _discover_gpu_ids()
        Orchestrator._model_cache = model_cache
        Orchestrator._image_cache = image_cache
        os.makedirs(model_cache, exist_ok=True)
        os.makedirs(image_cache, exist_ok=True)
        Orchestrator._pool = ThreadPoolExecutor()
        Orchestrator._gpu_queue = queue.Queue()
        for gpu_id in Orchestrator._gpu_ids:
            Orchestrator._gpu_queue.put(gpu_id)
        print(f"[orchestrator] init  model_cache={model_cache}  "
              f"image_cache={image_cache}  gpus={Orchestrator._gpu_ids}")

    # ------------------------------------------------------------------
    # register
    # ------------------------------------------------------------------

    @staticmethod
    def _image_dir_for(model_id: str) -> str:
        """Return the image-cache directory for a given model_id."""
        safe_name = model_id.replace("/", "--")
        return os.path.join(Orchestrator._image_cache, safe_name)

    @staticmethod
    def register(model_id: str, vllm_config: dict) -> None:
        """Download model weights, then submit GPU registration to the pool."""
        _console(f"{model_id}: register received")
        hf_model = vllm_config["model"]
        print(f"[orchestrator] register  model_id={model_id}  model={hf_model}")

        local_dir = os.path.join(Orchestrator._model_cache, hf_model)
        t0 = time.perf_counter()
        print(f"[orchestrator] {model_id}: downloading {hf_model} ...")
        snapshot_download(hf_model, local_dir=local_dir)
        t_dl = time.perf_counter() - t0
        print(f"[orchestrator] {model_id}: download done  ({t_dl:.1f}s)")

        vllm_config = dict(vllm_config, model=local_dir)
        fut = Orchestrator._pool.submit(
            Orchestrator._register_sync, model_id, vllm_config,
        )
        Orchestrator._futures[model_id] = fut

    @staticmethod
    def _register_sync(model_id: str, vllm_config: dict) -> None:
        """Blocking GPU registration (runs in a pool thread).

        On image-cache hit: load from disk (no GPU needed).
        On miss: acquire GPU, cold-start, save image, release GPU.
        """
        t0 = time.perf_counter()
        image_dir = Orchestrator._image_dir_for(model_id)
        cache_hit = os.path.isfile(os.path.join(image_dir, "meta.json"))
        inst = Instance(vllm_config)

        if cache_hit:
            print(f"[orchestrator] {model_id}: image cache HIT — loading from {image_dir}")
            inst.load(image_dir).wait()
            t_done = time.perf_counter()
            print(f"[orchestrator] {model_id}: loaded from image  ({t_done - t0:.1f}s)")
            _console(f"{model_id}: registered via image load ({t_done - t0:.1f}s)")
        else:
            print(f"[orchestrator] {model_id}: image cache MISS — cold-starting")
            gpu = Orchestrator._acquire_gpu(model_id)
            t_acquired = time.perf_counter()

            inst.init(gpu).attach().repin().stage().unpin().sleep().checkpoint().wait()
            inst.save(image_dir).wait()
            print(f"[orchestrator] {model_id}: image saved to {image_dir}")

            Orchestrator._release_gpu(gpu)
            t_done = time.perf_counter()
            t_wait = t_acquired - t0
            t_exec = t_done - t_acquired
            print(f"[orchestrator] {model_id}: cold-start done on GPU {gpu}  "
                  f"({t_done - t0:.1f}s)")
            _console(f"{model_id}: registered via cold-start "
                     f"(wait={t_wait:.1f}s, register={t_exec:.1f}s, total={t_wait + t_exec:.1f}s)")

        Orchestrator._registry[model_id] = {
            "instance": inst,
            "vllm_config": vllm_config,
        }

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
            .repin() \
            .wake_up_weights() \
            .h2d() \
            .scatter() \
            .wake_up_kv_cache() \
            .wait()
        t_ready = time.perf_counter()

        inst.generate(prompts, sampling_params).wait()
        result = inst.last_generate_result
        t_generated = time.perf_counter()

        inst.unpin().sleep().checkpoint().wait()
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
        if not Orchestrator._gpu_ids:
            _console("Orchestrator not initialized. Call Orchestrator.init() first.")
            return

        _console(f"\nOrchestrator  model_cache={Orchestrator._model_cache}  image_cache={Orchestrator._image_cache}")

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
        max_id = max(len(mid) for mid in Orchestrator._registry)
        _console(f"\nModels ({len(Orchestrator._registry)}):")
        for model_id, entry in Orchestrator._registry.items():
            inst = entry["instance"]
            hf_model = entry["vllm_config"].get("model", "?")
            _console(f"  {model_id:<{max_id}}  {hf_model}")
            _console(f"  {' ' * max_id}  {inst}")
        _console("")
