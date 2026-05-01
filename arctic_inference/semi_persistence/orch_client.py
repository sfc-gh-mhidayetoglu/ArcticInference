"""HTTP client stub mirroring the :class:`Orchestrator` static API.

A workload script imports :class:`RemoteOrchestrator` and uses it as a
drop-in replacement for the in-process ``Orchestrator``::

    from orch_client import RemoteOrchestrator as orch

    orch.connect("http://localhost:8157")
    orch.register("model 1", {"model": "Qwen/Qwen3-32B"})
    orch.wait_all()
    orch.move("model 1", "sleep", target_gpu=0)
    orch.generate("model 1", "What is gravity?", 100)
    orch.wait_all()

The server is :mod:`orch_server`; the wire protocol is the JSON HTTP API
exposed by :class:`state_server.StateHandler`.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request


class RemoteOrchestrator:
    """Class-method API that talks to a remote orchestrator over HTTP.

    Mirrors the surface of :class:`orchestrator.Orchestrator` so existing
    scripts keep working with a one-line import swap and a ``connect``
    call in place of ``init``.  Uses only the standard library
    (``urllib.request`` + ``json``) -- no third-party dependencies.
    """

    _base: str = ""
    _timeout_s: float = 600.0

    # ------------------------------------------------------------------
    # connection
    # ------------------------------------------------------------------

    @classmethod
    def connect(cls, base_url: str, *, timeout_s: float = 600.0) -> None:
        """Bind to a server URL and sanity-ping ``GET /state``.

        *timeout_s* applies to every subsequent HTTP call, including
        long-blocking ones like :meth:`wait`.  Bump it if your server
        runs huge cold-starts under contention.
        """
        cls._base = base_url.rstrip("/")
        cls._timeout_s = timeout_s
        cls._get("/state")

    # ------------------------------------------------------------------
    # API surface (mirrors Orchestrator's static methods)
    # ------------------------------------------------------------------

    @classmethod
    def register(cls, model_id: str, vllm_config: dict | str) -> None:
        if isinstance(vllm_config, str):
            vllm_config = {"model": vllm_config}
        cls._post("/register",
                  {"model_id": model_id, "vllm_config": vllm_config})

    @classmethod
    def move(cls, model_id: str, target: str,
             target_gpu: int | None = None) -> None:
        cls._post("/move", {
            "model_id": model_id,
            "target": target,
            "target_gpu": target_gpu,
        })

    @classmethod
    def move_all(cls, target: str, target_gpu: int | None = None) -> None:
        """Fan out :meth:`move` server-side to every registered model."""
        cls._post("/move_all", {
            "target": target,
            "target_gpu": target_gpu,
        })

    @classmethod
    def generate(cls, model_id: str, prompts: list[str] | str,
                 sampling_params: dict | int | None = None) -> int | None:
        """Submit a non-blocking generate; returns the server-assigned ``req_id``.

        Accepts the same shorthand as :meth:`Orchestrator.generate`:
        ``prompts`` may be a single string, ``sampling_params`` may be an
        int (interpreted as ``{"max_tokens": N, "ignore_eos": True}``).
        Returns ``None`` if the server reports the model is not
        registered (mirrors the in-process warn-and-skip behaviour).
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        if isinstance(sampling_params, int):
            sampling_params = {"max_tokens": sampling_params,
                               "ignore_eos": True}
        resp = cls._post("/generate", {
            "model_id": model_id,
            "prompts": prompts,
            "sampling_params": sampling_params,
        })
        return resp.get("req_id")

    @classmethod
    def generate_all(cls, prompts: list[str] | str,
                     sampling_params: dict | int | None = None
                     ) -> list[int]:
        """Fan out :meth:`generate` (same prompts) to every registered model.

        Returns one ``req_id`` per model the server accepted.
        """
        if isinstance(prompts, str):
            prompts = [prompts]
        if isinstance(sampling_params, int):
            sampling_params = {"max_tokens": sampling_params,
                               "ignore_eos": True}
        resp = cls._post("/generate_all", {
            "prompts": prompts,
            "sampling_params": sampling_params,
        })
        return resp.get("req_ids", [])

    @classmethod
    def wait(cls, model_id: str) -> None:
        """Block until the server's pending futures for *model_id* complete."""
        cls._post("/wait", {"model_id": model_id})

    @classmethod
    def wait_all(cls) -> None:
        """Block until every pending future on the server completes."""
        cls._post("/wait_all", {})

    @classmethod
    def remove(cls, model_id: str) -> None:
        cls._post("/remove", {"model_id": model_id})

    @classmethod
    def remove_all(cls) -> None:
        """Fan out :meth:`remove` server-side to every registered model."""
        cls._post("/remove_all", {})

    @classmethod
    def status(cls) -> dict:
        """Return the full state snapshot (same payload as ``GET /state``)."""
        _, body = cls._get("/state")
        return body

    # ------------------------------------------------------------------
    # transport helpers
    # ------------------------------------------------------------------

    @classmethod
    def _ensure_connected(cls) -> None:
        if not cls._base:
            raise RuntimeError(
                "RemoteOrchestrator.connect(...) must be called first")

    @classmethod
    def _post(cls, path: str, payload: dict | None = None) -> dict:
        cls._ensure_connected()
        data = json.dumps(payload or {}).encode()
        req = urllib.request.Request(
            cls._base + path, data=data,
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=cls._timeout_s) as r:
                return json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            body = json.loads(e.read() or b"{}")
            msg = body.get("error", str(e))
            raise RuntimeError(f"orch_server {path} -> HTTP {e.code}: {msg}")

    @classmethod
    def _get(cls, path: str) -> tuple[int, dict]:
        cls._ensure_connected()
        try:
            with urllib.request.urlopen(
                    cls._base + path, timeout=cls._timeout_s) as r:
                return r.status, json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")
