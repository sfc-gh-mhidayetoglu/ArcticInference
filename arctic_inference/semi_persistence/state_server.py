"""Embedded HTTP server that exposes orchestrator state as JSON (GET /state)."""
from __future__ import annotations

import json
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

_t0: float = 0.0


def init_t0() -> None:
    """Record the orchestrator start time for relative request timestamps."""
    global _t0
    _t0 = time.perf_counter()


def snapshot_state() -> dict:
    """Return a JSON-serialisable snapshot of the orchestrator state."""
    from orchestrator import Orchestrator

    now = time.perf_counter()
    t0_valid = _t0 and _t0 > 0.0
    elapsed_s = round(now - _t0, 3) if t0_valid else None
    models = {}
    for mid, entry in list(Orchestrator._registry.items()):
        inst = entry.get("instance")
        pid = None
        if inst is not None and getattr(inst, "pid", None):
            pid = inst.pid
        raw_since = entry.get("state_since", 0)
        state_since_rel = (
            round(raw_since - _t0, 3)
            if t0_valid and raw_since
            else None
        )
        models[mid] = {
            "state": entry.get("state"),
            "gpu": entry.get("gpu"),
            "pid": pid,
            "pinned_bytes": entry.get("pinned_bytes", 0),
            "vllm_config": entry.get("vllm_config", {}),
            "state_since": entry.get("state_since", 0),
            "state_since_rel_s": state_since_rel,
        }
    free_gpus = sorted(
        s.gpu_id for s in Orchestrator._gpus.values() if s.is_free
    )

    requests = []
    with Orchestrator._request_lock:
        for rec in Orchestrator._request_log:
            state = rec["state"]
            t_submit = rec["t_submit"]
            t_gen_start = rec["t_gen_start"]
            t_done = rec["t_done"]

            if state == "waiting":
                wait_s = now - t_submit
                gen_s = None
            elif state == "generating":
                if t_gen_start is not None:
                    wait_s = t_gen_start - t_submit
                    gen_s = now - t_gen_start
                else:
                    wait_s = now - t_submit
                    gen_s = None
            else:
                wait_s = (t_gen_start - t_submit) if t_gen_start else None
                gen_s = (t_done - t_gen_start) if (t_done and t_gen_start) else None

            if _t0 and _t0 > 0.0:
                submit_rel_s = round(t_submit - _t0, 3)
                gen_start_rel_s = (
                    round(t_gen_start - _t0, 3)
                    if t_gen_start is not None
                    else None
                )
                done_rel_s = (
                    round(t_done - _t0, 3) if t_done is not None else None
                )
            else:
                submit_rel_s = None
                gen_start_rel_s = None
                done_rel_s = None

            requests.append({
                "req_id": rec["req_id"],
                "model_id": rec["model_id"],
                "state": state,
                "start_state": rec.get("start_state"),
                "wait_s": round(wait_s, 2) if wait_s is not None else None,
                "gen_s": round(gen_s, 2) if gen_s is not None else None,
                "prompt_tokens": rec.get("prompt_tokens"),
                "completion_tokens": rec.get("completion_tokens"),
                "gpu_wait_s": round(rec["gpu_wait_s"], 2) if rec.get("gpu_wait_s") is not None else None,
                "migrate_s": round(rec["migrate_s"], 2) if rec.get("migrate_s") is not None else None,
                "up_s": round(rec["up_s"], 2) if rec.get("up_s") is not None else None,
                "submit_rel_s": submit_rel_s,
                "gen_start_rel_s": gen_start_rel_s,
                "done_rel_s": done_rel_s,
            })

    return {
        "gpu_ids": list(Orchestrator._gpu_ids),
        "gpu_pool": free_gpus,
        "image_cache": Orchestrator._image_cache,
        "models": models,
        "requests": requests,
        "elapsed_s": elapsed_s,
    }


class StateHandler(BaseHTTPRequestHandler):
    """Serves GET /state as a JSON snapshot of the orchestrator registry."""

    def do_GET(self):
        if self.path != "/state":
            self.send_error(404)
            return
        payload = snapshot_state()
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


def start_state_server(port: int) -> HTTPServer:
    server = HTTPServer(("0.0.0.0", port), StateHandler)
    server.socket.set_inheritable(False)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server
