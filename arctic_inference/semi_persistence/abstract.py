"""Abstract interface for a standalone vLLM instance.

Each Instance is a GPU-agnostic handle for a vLLM engine.  The GPU is
specified at init(gpu) time.  All primitives are non-blocking (except
wait) and return Self for chaining.  Cross-instance dependencies are
expressed via after().
"""
from __future__ import annotations

from abc import ABC, abstractmethod


class InstanceBase(ABC):
    """Abstract interface for a vLLM instance.

    Usage:
        instance = Instance(vllm_config)
        instance.init(gpu=0).attach().sleep().checkpoint().wait()
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
    def after(self, other: InstanceBase) -> InstanceBase:
        """Wait for all of other's pending commands before proceeding.

        Non-blocking from the main process: enqueues a wait_for on the
        worker, which blocks until the dependency is satisfied.
        """

    @abstractmethod
    def sleep(self) -> InstanceBase:
        """Free GPU memory for weights and KV cache (vLLM sleep level=2)."""

    @abstractmethod
    def checkpoint(self) -> InstanceBase:
        """Save CUDA state to CPU. Instance becomes stateless (gpu=None)."""

    @abstractmethod
    def restore(self, gpu: int) -> InstanceBase:
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
    def wake_up(self, tags: list[str]) -> InstanceBase:
        """Re-allocate tensors on GPU.  tags: ["weights"] or ["kv_cache"]."""

    @abstractmethod
    def h2d(self) -> InstanceBase:
        """Async host-to-device transfer, then synchronize."""

    @abstractmethod
    def scatter(self) -> InstanceBase:
        """Place weights from GPU staging buffer into model params.

        Frees the GPU staging buffer after scatter completes.
        """

    @abstractmethod
    def teardown(self) -> InstanceBase:
        """Tear down this instance and its worker. Resets to created state."""

    @abstractmethod
    def remove(self) -> InstanceBase:
        """Teardown and remove from the instance registry."""
