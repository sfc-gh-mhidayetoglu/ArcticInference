"""Per-GPU slot tracking and fleet-wide GPU scheduling for the orchestrator."""
from __future__ import annotations

import threading
import time
from collections.abc import Callable


class GpuSlot:
    """Per-GPU state tracker.

    Tracks which model holds the GPU exclusively (up/running/init) and
    which models are sleeping on it.  All mutations are serialized by
    ``self.lock`` so concurrent threads see a consistent view.
    """

    def __init__(self, gpu_id: int):
        self.gpu_id = gpu_id
        self.lock = threading.Lock()
        self._free = threading.Event()
        self._free.set()
        self.locked_by: str | None = None
        self.sleepers: list[str] = []
        #: When each model became a sleeper on this GPU (``perf_counter``); used to order
        #: sleepers with **longest sleeping at the bottom** of the list (newest at index 0).
        self._sleeper_since: dict[str, float] = {}
        self.last_event_ts: float = 0.0

    @property
    def is_free(self) -> bool:
        return self._free.is_set()

    def lock_gpu(self, model_id: str, timeout: float | None = None) -> bool:
        """Block until the GPU is free, then lock it for *model_id*.

        Returns True if acquired, False on timeout.
        """
        deadline = (time.perf_counter() + timeout) if timeout else None
        while True:
            remaining = (deadline - time.perf_counter()) if deadline else None
            if remaining is not None and remaining <= 0:
                return False
            if not self._free.wait(timeout=remaining):
                return False
            with self.lock:
                if self._free.is_set():
                    self._free.clear()
                    self.locked_by = model_id
                    self.last_event_ts = time.perf_counter()
                    return True

    def try_lock(self, model_id: str) -> bool:
        """Non-blocking lock attempt. Returns True if acquired."""
        with self.lock:
            if not self._free.is_set():
                return False
            self._free.clear()
            self.locked_by = model_id
            self.last_event_ts = time.perf_counter()
            return True

    def unlock(self) -> None:
        with self.lock:
            self.locked_by = None
            self.last_event_ts = time.perf_counter()
            self._free.set()

    def remove_sleeper(self, model_id: str) -> None:
        with self.lock:
            try:
                self.sleepers.remove(model_id)
            except ValueError:
                pass
            self._sleeper_since.pop(model_id, None)
            self.last_event_ts = time.perf_counter()

    def add_sleeper(self, model_id: str) -> None:
        """Register *model_id* as a sleeper on this GPU."""
        with self.lock:
            self._sleeper_since[model_id] = time.perf_counter()
            self.sleepers.append(model_id)
            self._resort_sleepers()
            self.last_event_ts = time.perf_counter()

    def transfer_lock(self, from_model: str, to_model: str) -> None:
        """Atomically transfer exclusive lock from *from_model* to *to_model*.

        *from_model* becomes a sleeper; *to_model* is removed from sleepers
        and becomes the exclusive holder. The GPU is never marked free.
        """
        with self.lock:
            assert self.locked_by == from_model
            self._sleeper_since[from_model] = time.perf_counter()
            self.sleepers.append(from_model)
            self._resort_sleepers()
            try:
                self.sleepers.remove(to_model)
            except ValueError:
                pass
            self._sleeper_since.pop(to_model, None)
            self.locked_by = to_model
            self.last_event_ts = time.perf_counter()

    def _resort_sleepers(self) -> None:
        """Newest sleeper first (top), longest sleeping last (bottom)."""
        self.sleepers.sort(
            key=lambda m: self._sleeper_since.get(m, 0.0),
            reverse=True,
        )

    def resident_count(self) -> int:
        with self.lock:
            return len(self.sleepers) + (1 if self.locked_by else 0)

    @classmethod
    def pick_for_sleep_placement(cls, slots: list[GpuSlot]) -> GpuSlot:
        """Choose a GPU to host a new sleeper (prefer free, then least loaded, then coldest)."""
        return min(
            slots,
            key=lambda s: (
                0 if s.is_free else 1,
                s.resident_count(),
                s.last_event_ts,
            ),
        )

    # ------------------------------------------------------------------
    # Fleet-wide exclusive lock (one pool of slots + shared condition)
    # ------------------------------------------------------------------

    @classmethod
    def coldest_free_slot(cls, slots: list[GpuSlot]) -> GpuSlot | None:
        """Among slots with ``is_free``, return the one with smallest ``last_event_ts``."""
        free = [s for s in slots if s.is_free]
        if not free:
            return None
        return min(free, key=lambda s: s.last_event_ts)

    @classmethod
    def acquire_exclusive(
        cls,
        pool: dict[int, GpuSlot],
        pool_cv: threading.Condition,
        label: str,
    ) -> tuple[int, float]:
        """Block until *label* holds an exclusive lock on some GPU in *pool*.

        Prefers the free GPU that has been idle longest (smallest ``last_event_ts``).

        Returns ``(gpu_id, wait_s)`` — *wait_s* is wall time spent in this wait loop.
        """
        t0 = time.perf_counter()
        while True:
            candidate = None
            with pool_cv:
                candidate = cls.coldest_free_slot(list(pool.values()))
                if candidate is None:
                    pool_cv.wait(timeout=0.1)
            if candidate is not None and candidate.try_lock(label):
                return candidate.gpu_id, time.perf_counter() - t0


class SlotPool:
    """All :class:`GpuSlot` devices plus shared scheduling for exclusive GPU locks.

    Waiting threads form a FIFO queue.  When a slot becomes available only
    the *longest-waiting* thread is woken, avoiding thundering-herd races
    where multiple migrating models fight over the same GPU.
    """

    def __init__(self, gpu_ids: list[int]):
        self._acquire_cv = threading.Condition()
        self.slots: dict[int, GpuSlot] = {
            g: GpuSlot(g) for g in gpu_ids
        }
        self._waiter_queue: list[threading.Event] = []
        self._queue_lock = threading.Lock()

    def enqueue_waiter(self) -> threading.Event:
        """Register a new waiter at the back of the FIFO and return its Event."""
        evt = threading.Event()
        with self._queue_lock:
            self._waiter_queue.append(evt)
        return evt

    def dequeue_waiter(self, evt: threading.Event) -> None:
        """Remove a waiter from the queue (after it acquired a slot or gave up)."""
        with self._queue_lock:
            try:
                self._waiter_queue.remove(evt)
            except ValueError:
                pass

    def _wake_first_waiter(self) -> None:
        """Set the event for the oldest waiter so only it proceeds."""
        with self._queue_lock:
            for evt in self._waiter_queue:
                if not evt.is_set():
                    evt.set()
                    break

    def acquire_exclusive(self, label: str) -> tuple[int, float]:
        """Block until *label* holds an exclusive lock; see :meth:`GpuSlot.acquire_exclusive`."""
        return GpuSlot.acquire_exclusive(self.slots, self._acquire_cv, label)

    def release_exclusive(self, gpu_id: int, after_unlock: Callable[[], None]) -> None:
        """Unlock *gpu_id*, run *after_unlock* (e.g. flush metrics), then wake the longest waiter."""
        self.slots[gpu_id].unlock()
        after_unlock()
        self._wake_first_waiter()

    def notify_acquire_waiters(self) -> None:
        """Wake the longest-waiting thread (no lock state change)."""
        self._wake_first_waiter()

