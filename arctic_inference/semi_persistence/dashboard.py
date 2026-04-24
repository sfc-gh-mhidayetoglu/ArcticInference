#!/usr/bin/env python3
"""Terminal dashboard for the Orchestrator.

Polls GET /state from the orchestrator's embedded HTTP server and renders
a three-tier curses display:

    Top tier    -- one column per GPU, showing models in up/running state
    Middle tier -- models in checkpoint (or wait) state
    Bottom tier -- models in saved state

Usage:
    python dashboard.py [--host HOST] [--port PORT] [--interval SECS]
"""
from __future__ import annotations

import argparse
import curses
import json
import subprocess
import threading
import time
import urllib.request
import urllib.error


# ---------------------------------------------------------------------------
# Colour pairs (initialised in _main)
# ---------------------------------------------------------------------------
C_TITLE = 1
C_GPU_HEADER = 2
C_RUNNING = 3
C_UP = 4
C_WAIT = 5
C_CHECKPOINT = 6
C_SAVED = 7
C_STATUS_OK = 8
C_STATUS_ERR = 9
C_BORDER = 10
C_SECTION = 11


def _init_colours():
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(C_TITLE, curses.COLOR_WHITE, -1)
    curses.init_pair(C_GPU_HEADER, curses.COLOR_CYAN, -1)
    curses.init_pair(C_RUNNING, curses.COLOR_GREEN, -1)
    curses.init_pair(C_UP, curses.COLOR_CYAN, -1)
    curses.init_pair(C_WAIT, curses.COLOR_YELLOW, -1)
    curses.init_pair(C_CHECKPOINT, curses.COLOR_YELLOW, -1)
    curses.init_pair(C_SAVED, curses.COLOR_WHITE, -1)
    curses.init_pair(C_STATUS_OK, curses.COLOR_GREEN, -1)
    curses.init_pair(C_STATUS_ERR, curses.COLOR_RED, -1)
    curses.init_pair(C_BORDER, curses.COLOR_WHITE, -1)
    curses.init_pair(C_SECTION, curses.COLOR_WHITE, -1)


# ---------------------------------------------------------------------------
# Local nvidia-smi queries
# ---------------------------------------------------------------------------

def _query_cpu_memory() -> tuple[float, float]:
    """Return (used_gib, total_gib) from /proc/meminfo."""
    try:
        with open("/proc/meminfo") as f:
            info = {}
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    info[parts[0].rstrip(":")] = int(parts[1])
        total = info.get("MemTotal", 0)
        avail = info.get("MemAvailable", 0)
        used = total - avail
        return used / (1 << 20), total / (1 << 20)
    except Exception:
        return 0.0, 0.0


def _query_gpu_memory() -> dict[int, dict]:
    """Return {gpu_index: {"used_mib": int, "total_mib": int}}."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=index,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        )
        result = {}
        for line in out.strip().splitlines():
            idx, used, total = (x.strip() for x in line.split(","))
            result[int(idx)] = {"used_mib": int(used), "total_mib": int(total)}
        return result
    except Exception:
        return {}


def _query_process_gpu_memory() -> dict[int, int]:
    """Return {pid: used_mib} for every compute process."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-compute-apps=pid,used_gpu_memory",
             "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        )
        result = {}
        for line in out.strip().splitlines():
            parts = line.split(",")
            if len(parts) == 2:
                result[int(parts[0].strip())] = int(parts[1].strip())
        return result
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# State poller (background thread)
# ---------------------------------------------------------------------------

class Poller:
    def __init__(self, url: str, interval: float, record_path: str | None = None):
        self._url = url
        self._interval = interval
        self._state: dict | None = None
        self._connected = False
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._record_file = open(record_path, "w") if record_path else None
        self._record_t0: float | None = None

    def start(self):
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()

    def stop(self):
        self._stop.set()
        if self._record_file is not None:
            self._record_file.close()

    def snapshot(self) -> tuple[dict | None, bool]:
        with self._lock:
            return self._state, self._connected

    def _loop(self):
        while not self._stop.is_set():
            try:
                req = urllib.request.Request(self._url, method="GET")
                with urllib.request.urlopen(req, timeout=2) as resp:
                    data = json.loads(resp.read())
                gpu_mem = _query_gpu_memory()
                pid_mem = _query_process_gpu_memory()
                cpu_used, cpu_total = _query_cpu_memory()
                data["gpu_mem"] = gpu_mem
                data["cpu_mem"] = {"used_gib": cpu_used, "total_gib": cpu_total}
                for _mid, info in data.get("models", {}).items():
                    pid = info.get("pid")
                    info["gpu_mem_mib"] = pid_mem.get(pid, 0) if pid else 0
                if self._record_file is not None and data.get("requests"):
                    if self._record_t0 is None:
                        self._record_t0 = time.monotonic()
                    t = time.monotonic() - self._record_t0
                    self._record_file.write(
                        json.dumps({"t": round(t, 3), "state": data}) + "\n"
                    )
                    self._record_file.flush()
                with self._lock:
                    self._state = data
                    self._connected = True
            except Exception:
                with self._lock:
                    was_connected = self._connected
                    self._connected = False
                if was_connected and self._record_file is not None:
                    print("Dashboard disconnected – stopping recording.")
                    self._record_file.close()
                    self._record_file = None
            self._stop.wait(self._interval)


class ReplayPoller:
    """Replays a recorded JSONL file, mimicking the Poller interface."""

    def __init__(self, path: str):
        self._entries: list[tuple[float, dict]] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                self._entries.append((rec["t"], rec["state"]))
        self._idx = 0
        self._wall_t0: float | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()

    def start(self):
        self._wall_t0 = time.monotonic()
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()

    def stop(self):
        self._stop.set()

    def snapshot(self) -> tuple[dict | None, bool]:
        with self._lock:
            if not self._entries:
                return None, False
            return self._entries[self._idx][1], True

    def _loop(self):
        while not self._stop.is_set():
            if self._wall_t0 is not None and self._entries:
                elapsed = time.monotonic() - self._wall_t0
                with self._lock:
                    while (self._idx < len(self._entries) - 1
                           and self._entries[self._idx + 1][0] <= elapsed):
                        self._idx += 1
            self._stop.wait(0.05)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

_SPINNER = "|/-\\"


def _safe_addstr(win, y: int, x: int, text: str, attr=0, max_x: int | None = None):
    """Write text clipped to window bounds, avoiding the curses bottom-right corner bug."""
    h, w = win.getmaxyx()
    if max_x is not None:
        w = min(w, max_x)
    if y < 0 or y >= h or x >= w:
        return
    avail = w - x
    if avail <= 0:
        return
    text = text[:avail]
    try:
        win.addstr(y, x, text, attr)
    except curses.error:
        pass


def _hline(win, y: int, x: int, width: int, char="─", attr=0):
    for i in range(width):
        _safe_addstr(win, y, x + i, char, attr)


def _vline(win, y: int, x: int, height: int, char="│", attr=0):
    for i in range(height):
        _safe_addstr(win, y + i, x, char, attr)


def _pinned_str(pinned_bytes: int) -> str:
    if pinned_bytes <= 0:
        return ""
    gib = pinned_bytes / (1 << 30)
    return f" ({gib:.1f}G)"


def _gpu_mem_str(info: dict) -> str:
    """Format GPU memory for a model: prefer actual gpu_mem, fall back to pinned."""
    gpu_mib = info.get("gpu_mem_mib", 0)
    if gpu_mib > 0:
        return f" ({gpu_mib / 1024:.1f}G)"
    return _pinned_str(info.get("pinned_bytes", 0))


def _fmt_duration_compact(seconds: float) -> str:
    """Format a duration as a compact string like ``12s``, ``3m24s``, ``1h05m``."""
    if seconds < 0:
        seconds = 0.0
    total = int(round(seconds))
    if total < 60:
        return f"{total}s"
    m = total // 60
    s = total % 60
    if m < 60:
        return f"{m}m{s:02d}s"
    h = m // 60
    m = m % 60
    return f"{h}h{m:02d}m"


def _state_age_str(info: dict, elapsed_s: float | None) -> str:
    """Return `` (3m24s)`` showing how long the model has been in its current state."""
    since = info.get("state_since_rel_s")
    if since is None or elapsed_s is None:
        return ""
    age = elapsed_s - since
    if age < 1:
        return ""
    return f" {_fmt_duration_compact(age)}"


def _fmt_rel_t0_hms(seconds: float) -> str:
    """Elapsed seconds since orchestrator ``t=0`` as ``h:mm:ss`` (minutes and seconds zero-padded)."""
    if seconds < 0:
        seconds = 0.0
    total = int(round(seconds))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h}:{m:02d}:{s:02d}"


def _fmt_request_time_log(req: dict) -> str:
    """Submission time since orchestrator ``t=0`` as ``h:mm:ss``."""
    rs = req.get("submit_rel_s")
    if rs is None:
        return ""
    return _fmt_rel_t0_hms(float(rs))


_START_STATE_SHORT = {
    "saved": "saved",
    "checkpoint": "ckpt",
    "sleep": "sleep",
    "up": "up",
    "running": "running",
    "wait": "wait",
}


def _fmt_model_with_start_state(model_id: str, start_state: str | None) -> str:
    """``model_id (ckpt)``-style label using model ladder state at generate submit."""
    if not start_state:
        return model_id
    short = _START_STATE_SHORT.get(start_state, start_state)
    return f"{model_id} ({short})"



# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def _render(win, state: dict | None, connected: bool, tick: int,
            req_scroll: int = 0, follow_newest: bool = True,
            poll_interval: float = 0.5) -> tuple[int, int, int]:
    """Render the dashboard.

    Returns ``(total_requests, avail_rows, effective_req_scroll)``.
    Requests are shown **oldest first** (top to bottom).  When *follow_newest*
    is true, the window is scrolled so the **newest** requests appear at the
    bottom of the panel (tail-follow).
    """
    win.erase()
    h, w = win.getmaxyx()
    if h < 6 or w < 30:
        _safe_addstr(win, 0, 0, "Terminal too small", curses.A_BOLD)
        win.noutrefresh()
        return 0, 0, 0

    border = curses.color_pair(C_BORDER) | curses.A_DIM

    # -- Title bar --
    title = " Orchestrator Dashboard "
    _safe_addstr(win, 0, 0, "╔" + "═" * (w - 2) + "╗", border)
    _safe_addstr(win, 0, 2, title, curses.color_pair(C_TITLE) | curses.A_BOLD)

    elapsed_s = state.get("elapsed_s") if state else None
    if elapsed_s is not None:
        t0_tag = f" t={_fmt_rel_t0_hms(elapsed_s)} "
        _safe_addstr(win, 0, 2 + len(title), t0_tag,
                     curses.color_pair(C_TITLE) | curses.A_DIM)

    interval_tag = f" poll {poll_interval}s "
    if connected:
        tag = " connected "
        combined = interval_tag + tag
        _safe_addstr(win, 0, max(0, w - len(combined) - 4), interval_tag,
                     curses.color_pair(C_TITLE) | curses.A_DIM)
        _safe_addstr(win, 0, max(0, w - len(tag) - 4), tag,
                     curses.color_pair(C_STATUS_OK) | curses.A_BOLD)
        _safe_addstr(win, 0, max(0, w - 3), "● ", curses.color_pair(C_STATUS_OK))
    else:
        spin = _SPINNER[tick % len(_SPINNER)]
        tag = f" connecting {spin} "
        combined = interval_tag + tag
        _safe_addstr(win, 0, max(0, w - len(combined) - 2), interval_tag,
                     curses.color_pair(C_TITLE) | curses.A_DIM)
        _safe_addstr(win, 0, max(0, w - len(tag) - 2), tag,
                     curses.color_pair(C_STATUS_ERR) | curses.A_BOLD)

    if state is None:
        _safe_addstr(win, 2, 2, "Waiting for orchestrator...",
                     curses.color_pair(C_STATUS_ERR))
        win.noutrefresh()
        return 0, 0, 0

    gpu_ids: list[int] = state.get("gpu_ids", [])
    gpu_pool: list[int] = state.get("gpu_pool", [])
    gpu_mem: dict = state.get("gpu_mem", {})
    models: dict = state.get("models", {})
    free_set = set(gpu_pool)

    # Classify models by tier
    gpu_active: dict[int, list[tuple[str, dict]]] = {g: [] for g in gpu_ids}
    gpu_sleep: dict[int, list[tuple[str, dict]]] = {g: [] for g in gpu_ids}
    checkpoint_models: list[tuple[str, dict]] = []
    saved_models: list[tuple[str, dict]] = []

    for mid, info in models.items():
        s = info.get("state")
        gpu = info.get("gpu")
        if s in ("up", "running", "init") and gpu is not None and gpu in gpu_active:
            gpu_active[gpu].append((mid, info))
        elif s == "sleep" and gpu is not None and gpu in gpu_sleep:
            gpu_sleep[gpu].append((mid, info))
        elif s in ("checkpoint", "wait"):
            checkpoint_models.append((mid, info))
        elif s == "saved":
            saved_models.append((mid, info))
        else:
            checkpoint_models.append((mid, info))

    n_gpus = max(len(gpu_ids), 1)
    inner_w = w - 2
    col_w = max(inner_w // n_gpus, 12)

    # -- GPU tier --
    row = 1
    _safe_addstr(win, row, 0, "╠" + "═" * (inner_w) + "╣", border)
    row += 1

    # GPU headers (memory on same row)
    hdr_row = row
    for i, gpu_id in enumerate(gpu_ids):
        cx = 1 + i * col_w
        max_col = 1 + (i + 1) * col_w
        free_tag = "" if gpu_id in free_set else "*"
        header = f" GPU {gpu_id}{free_tag}"
        _safe_addstr(win, hdr_row, cx, header,
                     curses.color_pair(C_GPU_HEADER) | curses.A_BOLD,
                     max_x=max_col)
        gi = gpu_mem.get(gpu_id, gpu_mem.get(str(gpu_id), {}))
        used = gi.get("used_mib", 0)
        total = gi.get("total_mib", 0)
        rest_parts: list[str] = []
        if total:
            used_gib = used / 1024
            rest_parts.append(f"{used_gib:.1f}/{total / 1024:.1f}G")
        if not rest_parts:
            rest_parts.append("—")
        if rest_parts:
            rest = "  " + "  ".join(rest_parts)
            _safe_addstr(
                win, hdr_row, cx + len(header), rest,
                curses.color_pair(C_GPU_HEADER) | curses.A_DIM,
                max_x=max_col,
            )
    row = hdr_row + 1

    # Compute tier heights per GPU column
    max_active = max((len(v) for v in gpu_active.values()), default=0)
    max_sleep = max((len(v) for v in gpu_sleep.values()), default=0)
    active_height = max(max_active, 1)
    sleep_height = max(max_sleep, 0)
    # 1 row for separator between active and sleep (if any sleep models exist)
    has_sleep = max_sleep > 0
    gpu_tier_height = active_height + (1 + sleep_height if has_sleep else 0)

    gpu_tier_start = row

    # -- Active (up/running) models --
    for i, gpu_id in enumerate(gpu_ids):
        cx = 1 + i * col_w
        max_col = 1 + (i + 1) * col_w - 1
        mlist = sorted(gpu_active[gpu_id],
                       key=lambda x: x[1].get("gpu_mem_mib", 0), reverse=True)
        for j, (mid, info) in enumerate(mlist):
            s = info.get("state", "")
            if s == "running":
                indicator = "▶ "
                colour = curses.color_pair(C_RUNNING) | curses.A_BOLD
            elif s == "init":
                indicator = _SPINNER[tick % len(_SPINNER)] + " "
                colour = curses.color_pair(C_WAIT)
            else:
                indicator = "● "
                colour = curses.color_pair(C_UP)
            label = indicator + mid + _gpu_mem_str(info) + _state_age_str(info, elapsed_s)
            _safe_addstr(win, gpu_tier_start + j, cx + 1, label, colour,
                         max_x=max_col)
        if not mlist:
            _safe_addstr(win, gpu_tier_start, cx + 1, "—",
                         curses.A_DIM, max_x=max_col)

    # -- Thin separator + sleep models --
    if has_sleep:
        sep_y = gpu_tier_start + active_height
        for i, gpu_id in enumerate(gpu_ids):
            cx = 1 + i * col_w
            _hline(win, sep_y, cx, col_w - 1, char="·", attr=curses.A_DIM)
        sleep_start = sep_y + 1
        for i, gpu_id in enumerate(gpu_ids):
            cx = 1 + i * col_w
            max_col = 1 + (i + 1) * col_w - 1
            sleep_sorted = sorted(gpu_sleep[gpu_id],
                                   key=lambda x: x[1].get("gpu_mem_mib", 0),
                                   reverse=True)
            for j, (mid, info) in enumerate(sleep_sorted):
                label = "○ " + mid + _gpu_mem_str(info) + _state_age_str(info, elapsed_s)
                _safe_addstr(win, sleep_start + j, cx + 1, label,
                             curses.color_pair(C_UP),
                             max_x=max_col)

    # Vertical separators between GPU columns
    for i in range(1, n_gpus):
        sep_x = 1 + i * col_w - 1
        if sep_x < w - 1:
            for dy in range(gpu_tier_height):
                _safe_addstr(win, gpu_tier_start + dy, sep_x, "│", border)

    row = gpu_tier_start + gpu_tier_height

    # -- Checkpoint tier --
    _safe_addstr(win, row, 0, "╠" + "═" * inner_w + "╣", border)
    row += 1
    cpu = state.get("cpu_mem", {})
    cpu_used = cpu.get("used_gib", 0)
    cpu_total = cpu.get("total_gib", 0)
    section_label = " CPU "
    _safe_addstr(win, row, 1, section_label,
                 curses.color_pair(C_SECTION) | curses.A_BOLD)
    if cpu_total:
        mem_label = f" {cpu_used:.1f}/{cpu_total:.1f}G"
        _safe_addstr(win, row, 1 + len(section_label), mem_label,
                     curses.color_pair(C_SECTION) | curses.A_DIM)
    row += 1

    gpu_resident: list[tuple[str, dict]] = []
    for mid, info in models.items():
        s = info.get("state")
        if s in ("sleep", "up", "running"):
            gpu_resident.append((mid, info))

    cpu_models = checkpoint_models + gpu_resident
    cpu_models.sort(key=lambda x: x[1].get("pinned_bytes", 0), reverse=True)

    if cpu_models:
        for mid, info in cpu_models:
            s = info.get("state", "checkpoint")
            if s == "wait":
                spin = _SPINNER[tick % len(_SPINNER)]
                label = f"{spin} {mid}"
                colour = curses.color_pair(C_WAIT)
            elif s in ("checkpoint",):
                label = mid
                colour = curses.color_pair(C_CHECKPOINT)
            else:
                label = mid
                colour = curses.color_pair(C_CHECKPOINT) | curses.A_DIM
            label += _pinned_str(info.get("pinned_bytes", 0))
            label += _state_age_str(info, elapsed_s)
            pid = info.get("pid")
            if pid:
                label += f"  pid={pid}"
            _safe_addstr(win, row, 2, label, colour)
            row += 1
    else:
        _safe_addstr(win, row, 2, "—", curses.A_DIM)
        row += 1

    # -- Saved tier --
    _safe_addstr(win, row, 0, "╠" + "═" * inner_w + "╣", border)
    row += 1
    saved_label = " Image Cache "
    _safe_addstr(win, row, 1, saved_label,
                 curses.color_pair(C_SECTION) | curses.A_BOLD)
    image_cache = state.get("image_cache", "")
    if image_cache:
        _safe_addstr(win, row, 1 + len(saved_label), f" {image_cache}",
                     curses.color_pair(C_SECTION) | curses.A_DIM)
    row += 1

    all_models = list(models.items())
    all_models.sort(key=lambda x: x[1].get("pinned_bytes", 0), reverse=True)
    if all_models:
        for mid, info in all_models:
            vc = info.get("vllm_config", {})
            model_path = vc.get("model", "")
            label = f"{mid}{_pinned_str(info.get('pinned_bytes', 0))}  {model_path}"
            if info.get("state") == "saved":
                colour = curses.color_pair(C_SAVED)
            else:
                colour = curses.color_pair(C_SAVED) | curses.A_DIM
            _safe_addstr(win, row, 2, label, colour)
            row += 1
    else:
        _safe_addstr(win, row, 2, "—", curses.A_DIM)
        row += 1

    # -- Requests tier --
    requests: list[dict] = state.get("requests", [])
    total_requests = len(requests)

    _safe_addstr(win, row, 0, "╠" + "═" * inner_w + "╣", border)
    row += 1
    active_requests = sum(
        1 for r in requests if r.get("state") in ("waiting", "generating")
    )
    req_label = " Requests "
    _safe_addstr(win, row, 1, req_label,
                 curses.color_pair(C_SECTION) | curses.A_BOLD)
    req_header_extra = 0
    if active_requests:
        active_label = f" ({active_requests} active)"
        _safe_addstr(win, row, 1 + len(req_label), active_label,
                     curses.color_pair(C_SECTION) | curses.A_DIM)
        req_header_extra = len(active_label)
    tail_tag = "  tail" if follow_newest else "  scroll"
    _safe_addstr(win, row, 1 + len(req_label) + req_header_extra, tail_tag,
                 curses.color_pair(C_SECTION) | curses.A_DIM)
    row += 1

    # 2 rows reserved: bottom border + at least 1 border row
    avail_rows = max(h - row - 1, 0)
    eff_scroll = 0

    if requests:
        if avail_rows <= 0:
            eff_scroll = 0
        elif follow_newest:
            eff_scroll = max(0, total_requests - avail_rows)
        else:
            eff_scroll = min(req_scroll, max(0, total_requests - avail_rows))
        visible = requests[eff_scroll:eff_scroll + avail_rows]
        for req in visible:
            if row >= h - 1:
                break
            rid = req.get("req_id", 0)
            mid = req.get("model_id", "?")
            rstate = req.get("state", "?")
            wait_s = req.get("wait_s")
            gen_s = req.get("gen_s")
            gpu_wait_s = req.get("gpu_wait_s") or 0
            migrate_s = req.get("migrate_s") or 0
            up_s = req.get("up_s") or 0
            ptok = req.get("prompt_tokens")
            ctok = req.get("completion_tokens")

            wait_details = []
            if migrate_s > 0:
                wait_details.append(f"{migrate_s:.1f}s down")
            if gpu_wait_s >= 0.1:
                wait_details.append(f"{gpu_wait_s:.1f}s gpu")
            if up_s >= 0.1 and (migrate_s > 0 or gpu_wait_s >= 0.1):
                wait_details.append(f"{up_s:.1f}s up")
            gpu_suffix = f" ({', '.join(wait_details)})" if wait_details else ""

            tlog = _fmt_request_time_log(req)
            model_col = _fmt_model_with_start_state(mid, req.get("start_state"))
            parts = [f"#{rid + 1}"]
            if tlog:
                parts.append(tlog)
            parts.append(f"{model_col:<28s}")

            if rstate == "waiting":
                spin = _SPINNER[tick % len(_SPINNER)]
                parts.append(f"{spin} wait {wait_s:.1f}s" if wait_s is not None else f"{spin} wait")
                colour = curses.color_pair(C_WAIT)
            elif rstate == "generating":
                wait_str = f"wait {wait_s:.1f}s" if wait_s is not None else "wait"
                spin = _SPINNER[tick % len(_SPINNER)]
                gen_str = f"{spin} gen {gen_s:.1f}s" if gen_s is not None else f"{spin} gen"
                parts.append(wait_str)
                parts.append(gen_str)
                if gpu_suffix:
                    parts.append(gpu_suffix)
                colour = curses.color_pair(C_RUNNING)
            else:
                wait_str = f"wait {wait_s:.1f}s" if wait_s is not None else ""
                gen_str = f"gen {gen_s:.1f}s" if gen_s is not None else ""
                parts.append(wait_str)
                parts.append(gen_str)
                if ptok is not None or ctok is not None:
                    tok_parts = []
                    if ptok is not None:
                        tok_parts.append(f"{ptok} in")
                    if ctok is not None:
                        tok_parts.append(f"{ctok} out")
                    parts.append(", ".join(tok_parts))
                if gpu_suffix:
                    parts.append(gpu_suffix)
                colour = curses.color_pair(C_SAVED) | curses.A_DIM

            label = " ".join(p for p in parts if p)
            _safe_addstr(win, row, 2, label, colour)
            row += 1
    else:
        _safe_addstr(win, row, 2, "—", curses.A_DIM)
        row += 1

    # -- Bottom border --
    if row < h:
        _safe_addstr(win, row, 0, "╚" + "═" * inner_w + "╝", border)

    # -- Left/right borders --
    for y in range(1, min(row, h)):
        _safe_addstr(win, y, 0, "║", border)
        _safe_addstr(win, y, w - 1, "║", border)

    win.noutrefresh()
    return total_requests, avail_rows, eff_scroll


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def _main(stdscr, args):
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(100)
    _init_colours()
    curses.mousemask(curses.ALL_MOUSE_EVENTS | curses.REPORT_MOUSE_POSITION)

    if args.replay:
        poller = ReplayPoller(args.replay)
    else:
        url = f"http://{args.host}:{args.port}/state"
        poller = Poller(url, args.interval, record_path=args.record)
    poller.start()

    tick = 0
    req_scroll = 0
    prev_total_requests = 0
    follow_newest = True
    SCROLL_STEP = 3

    try:
        while True:
            key = stdscr.getch()
            if key == ord("q") or key == ord("Q") or key == 27:
                break
            if key in (ord("t"), ord("T")):
                follow_newest = True
            if key == curses.KEY_RESIZE:
                curses.update_lines_cols()
            if key == curses.KEY_MOUSE:
                try:
                    _, _, _, _, bstate = curses.getmouse()
                    if bstate & curses.BUTTON4_PRESSED:
                        follow_newest = False
                        req_scroll = max(0, req_scroll - SCROLL_STEP)
                    elif bstate & curses.BUTTON5_PRESSED:
                        follow_newest = False
                        req_scroll += SCROLL_STEP
                except curses.error:
                    pass

            state, connected = poller.snapshot()
            total, _avail, eff = _render(
                stdscr, state, connected, tick, req_scroll,
                follow_newest=follow_newest,
                poll_interval=args.interval,
            )

            if total > prev_total_requests:
                follow_newest = True
            prev_total_requests = total
            req_scroll = eff

            curses.doupdate()
            tick += 1
            time.sleep(0.1)
    finally:
        poller.stop()


def main():
    parser = argparse.ArgumentParser(description="Orchestrator dashboard")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8157)
    parser.add_argument("--interval", type=float, default=0.5,
                        help="polling interval in seconds")
    parser.add_argument("--record", metavar="FILE", default=None,
                        help="record each polled state snapshot to a JSONL file")
    parser.add_argument("--replay", metavar="FILE", default=None,
                        help="replay from a recorded JSONL file instead of polling")
    args = parser.parse_args()
    curses.wrapper(lambda stdscr: _main(stdscr, args))


if __name__ == "__main__":
    main()
