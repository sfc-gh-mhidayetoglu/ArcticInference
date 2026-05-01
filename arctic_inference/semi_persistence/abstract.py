"""Abstract interface for a standalone vLLM instance.

Each Instance is a GPU-agnostic handle for a vLLM engine.  The GPU is
specified at init(gpu) time.  All primitives are non-blocking (except
wait) and return Self for chaining.
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class OrchestratorBase(ABC):
    """Abstract interface for an orchestrator that manages named model instances.

    State ladder:  saved <-> checkpoint <-> sleep <-> up -> running (transient)
    """

    @staticmethod
    @abstractmethod
    def init(image_cache: str, gpus: list[int] | None = None) -> None:
        """Discover GPUs, scan image cache, and populate registry."""

    @staticmethod
    @abstractmethod
    def register(model_id: str, vllm_config: dict | str) -> None:
        """Cold-start a new model.  Ends in checkpoint state."""

    @staticmethod
    @abstractmethod
    def move(model_id: str, target: str) -> None:
        """Walk the state ladder to *target* (saved/checkpoint/sleep/up)."""

    @staticmethod
    @abstractmethod
    def generate(model_id: str, prompts: list[str] | str,
                 sampling_params: dict | None = None) -> object:
        """Run inference.  Auto-transitions to up if needed."""

    @staticmethod
    @abstractmethod
    def remove(model_id: str | None = None) -> None:
        """Auto-transition to saved, delete image, de-register."""

    @staticmethod
    @abstractmethod
    def wait(model_id: str | None = None) -> None:
        """Block until pending operations complete."""

    @staticmethod
    @abstractmethod
    def status() -> None:
        """Print GPU view and registered models with states."""


class InstanceBase(ABC):
    """Abstract interface for a vLLM instance.

    Usage:
        instance = Instance(vllm_config)
        instance.init(gpu=0).attach().sleep().checkpoint_cuda().wait()
    """

    @abstractmethod
    def init(self, gpu: int) -> InstanceBase:
        """Cold start a model with random weights on the given GPU."""

    @abstractmethod
    def wait(self) -> InstanceBase:
        """Block until all pending commands complete for this instance.

        Raises RuntimeError on the first command that failed.
        """

    @abstractmethod
    def sleep(self) -> InstanceBase:
        """Free GPU memory for weights and KV cache (vLLM sleep level=2)."""

    @abstractmethod
    def checkpoint_cuda(self) -> InstanceBase:
        """Save CUDA state to CPU. Instance becomes stateless (gpu=None)."""

    @abstractmethod
    def save_image(self, filename: str) -> InstanceBase:
        """CRIU-dump the child process tree to disk (non-destructive).

        Uses --leave-running so the child stays alive after the dump.
        Writes meta.json with vllm_config and CRIU metadata.
        """

    @abstractmethod
    def load_image(self, filename: str | None = None) -> InstanceBase:
        """Restore a live process from a CRIU image on disk.

        Validates that the image's vllm_config matches this instance.
        Spawns a new worker and CRIU-restores the child.
        """

    @abstractmethod
    def restore_cuda(self, gpu: int) -> InstanceBase:
        """Restore checkpointed CUDA state onto the specified GPU."""

    @abstractmethod
    def attach(self) -> InstanceBase:
        """Allocate pinned CPU memory for staging weights."""

    @abstractmethod
    def detach(self) -> InstanceBase:
        """Free pinned CPU memory."""

    @abstractmethod
    def stage(self, model_path: str | None = None) -> InstanceBase:
        """Load model weights from disk into pinned CPU buffer.

        Requires a prior attach().
        """

    @abstractmethod
    def wake_up_weights(self) -> InstanceBase:
        """Re-allocate weight tensors on GPU."""

    @abstractmethod
    def wake_up_kv_cache(self) -> InstanceBase:
        """Re-allocate KV cache on GPU."""

    @abstractmethod
    def plan_load_weights(self) -> InstanceBase:
        """Precompute the chunk plan that the next ``load_weights()`` consumes.

        Self-computes the staging budget from instance state populated
        by ``init`` (cold start) or by ``load`` reading ``meta.json``
        (restore):

            allotment = total_gpu_bytes * gpu_memory_utilization
            budget    = min(pinned_cpu_bytes, allotment - pinned_cpu_bytes)

        Splits the parameter index into chunks of ``<= budget`` bytes
        and caches the result on the worker.  Subsequent
        ``load_weights()`` calls execute the cached plan.
        """

    @abstractmethod
    def load_weights(self) -> InstanceBase:
        """Copy staged weights from pinned CPU into model parameters.

        Pure execution against the chunk plan cached by a prior
        ``plan_load_weights()``.  For each chunk the worker copies a
        slice of the pinned buffer into a single reused GPU staging
        buffer (PCIe H2D) and then scatters into
        ``model.named_parameters()`` in place.  When no plan is cached,
        falls back to a single-chunk path identical to the pre-chunk
        behavior.  The staging buffer is freed before returning.
        """

    @abstractmethod
    def teardown(self) -> InstanceBase:
        """Tear down this instance and its worker. Resets to created state."""

    @abstractmethod
    def remove(self) -> InstanceBase:
        """Teardown and remove from the instance registry."""
