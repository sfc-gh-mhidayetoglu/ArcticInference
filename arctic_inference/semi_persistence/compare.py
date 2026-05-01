#!/usr/bin/env python3
"""Side-by-side comparison of two orchestrator recordings.

Reads two JSONL files produced by ``dashboard.py --record`` and displays:

    Left column   -- Dashboard A
    Middle column -- Dashboard B
    Right column  -- Shared charts (TTFT scatter + GPU utilization + efficiency)

Usage:
    python compare.py file_a.jsonl file_b.jsonl [--speed N] [--interval N] [--window N]
"""
from __future__ import annotations

import argparse
import curses
import json
import time

from monitor import (
    _compute_utilization,
    _backfill_util,
)

# ---------------------------------------------------------------------------
# Colour pairs  (1-11 match dashboard.py, 12+ are compare-only)
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

C_CHART_BORDER = 12
C_CHART_TITLE = 13
C_CHART_AXIS = 14
C_CHART_LABEL = 15

C_CASE1 = 16
C_CASE1_DIM = 17
C_CASE2 = 18
C_CASE2_DIM = 19


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

    curses.init_pair(C_CHART_BORDER, curses.COLOR_WHITE, -1)
    curses.init_pair(C_CHART_TITLE, curses.COLOR_WHITE, -1)
    curses.init_pair(C_CHART_AXIS, curses.COLOR_WHITE, -1)
    curses.init_pair(C_CHART_LABEL, curses.COLOR_WHITE, -1)

    curses.init_pair(C_CASE1, curses.COLOR_BLUE, -1)
    curses.init_pair(C_CASE1_DIM, curses.COLOR_BLUE, -1)
    curses.init_pair(C_CASE2, curses.COLOR_CYAN, -1)
    curses.init_pair(C_CASE2_DIM, curses.COLOR_CYAN, -1)


# ---------------------------------------------------------------------------
# Dashboard rendering helpers (forked from dashboard.py)
# ---------------------------------------------------------------------------

_SPINNER = "|/-\\"


def _safe_addstr(win, y: int, x: int, text: str, attr=0, max_x: int | None = None):
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


def _pinned_str(pinned_cpu_bytes: int) -> str:
    if pinned_cpu_bytes <= 0:
        return ""
    gib = pinned_cpu_bytes / (1 << 30)
    return f" ({gib:.1f}G)"


def _gpu_mem_str(info: dict) -> str:
    gpu_mib = info.get("gpu_mem_mib", 0)
    if gpu_mib > 0:
        return f" ({gpu_mib / 1024:.1f}G)"
    return _pinned_str(info.get("pinned_cpu_bytes", 0))


def _fmt_rel_t0_hms(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    total = int(round(seconds))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h}:{m:02d}:{s:02d}"


def _fmt_request_time_log(req: dict) -> str:
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
    if not start_state:
        return model_id
    short = _START_STATE_SHORT.get(start_state, start_state)
    return f"{model_id} ({short})"


# ---------------------------------------------------------------------------
# Dashboard _render  (forked from dashboard.py -- compare-specific tweaks)
# ---------------------------------------------------------------------------

def _measure_req_rows(win, state: dict | None) -> int:
    """Return how many request-list rows this panel would naturally have.

    This performs the same layout arithmetic as ``_render`` without drawing
    anything, so two panels can be sized to the same request height.
    """
    if state is None:
        return 0
    h, w = win.getmaxyx()
    if h < 6 or w < 30:
        return 0

    gpu_ids: list[int] = state.get("gpu_ids", [])
    models: dict = state.get("models", {})

    gpu_active: dict[int, list] = {g: [] for g in gpu_ids}
    gpu_sleep: dict[int, list] = {g: [] for g in gpu_ids}
    checkpoint_models: list = []

    for mid, info in models.items():
        s = info.get("state")
        gpu = info.get("gpu")
        if s in ("up", "running", "init") and gpu is not None and gpu in gpu_active:
            gpu_active[gpu].append(mid)
        elif s == "sleep" and gpu is not None and gpu in gpu_sleep:
            gpu_sleep[gpu].append(mid)
        elif s in ("checkpoint", "wait"):
            checkpoint_models.append(mid)

    max_active = max((len(v) for v in gpu_active.values()), default=0)
    max_sleep = max((len(v) for v in gpu_sleep.values()), default=0)
    active_height = max(max_active, 1)
    sleep_height = max(max_sleep, 0)
    has_sleep = max_sleep > 0
    gpu_tier_height = active_height + (1 + sleep_height if has_sleep else 0)

    # title + separator + gpu_header + gpu_tier + separator + cpu_header +
    # checkpoint_entries (min 1)
    n_ckpt = max(len(checkpoint_models), 1)
    row = 1 + 1 + 1 + gpu_tier_height + 1 + 1 + n_ckpt

    n_saved_entries = max(len(models), 1)
    saved_tier_rows = 1 + n_saved_entries + 1 + 1
    saved_start = h - saved_tier_rows

    # subtract 2 for the requests separator + header
    return max(saved_start - row - 2, 0)


def _render(win, state: dict | None, connected: bool, tick: int,
            req_scroll: int = 0, follow_newest: bool = True,
            poll_interval: float = 0.5,
            req_rows: int | None = None,
            panel_title: str | None = None) -> tuple[int, int, int]:
    """Render the dashboard panel.

    Returns ``(total_requests, avail_rows, effective_req_scroll)``.

    *req_rows*, when given, forces the request list to exactly that many
    rows so two side-by-side panels line up.
    """
    win.erase()
    h, w = win.getmaxyx()
    if h < 6 or w < 30:
        _safe_addstr(win, 0, 0, "Terminal too small", curses.A_BOLD)
        win.noutrefresh()
        return 0, 0, 0

    border = curses.color_pair(C_BORDER) | curses.A_DIM

    # -- Title bar --
    title = f" {panel_title} " if panel_title else " Orchestrator Dashboard "
    _safe_addstr(win, 0, 0, "╔" + "═" * (w - 2) + "╗", border)
    _safe_addstr(win, 0, 2, title, curses.color_pair(C_TITLE) | curses.A_BOLD)

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

    hdr_row = row
    for i, gpu_id in enumerate(gpu_ids):
        cx = 1 + i * col_w
        max_col = 1 + (i + 1) * col_w
        header = f" GPU {gpu_id}"
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

    max_active = max((len(v) for v in gpu_active.values()), default=0)
    max_sleep = max((len(v) for v in gpu_sleep.values()), default=0)
    active_height = max(max_active, 1)
    sleep_height = max(max_sleep, 0)
    has_sleep = max_sleep > 0
    gpu_tier_height = active_height + (1 + sleep_height if has_sleep else 0)

    gpu_tier_start = row

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
            label = indicator + mid + _gpu_mem_str(info)
            _safe_addstr(win, gpu_tier_start + j, cx + 1, label, colour,
                         max_x=max_col)
        if not mlist:
            _safe_addstr(win, gpu_tier_start, cx + 1, "—",
                         curses.A_DIM, max_x=max_col)

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
                label = "○ " + mid + _gpu_mem_str(info)
                _safe_addstr(win, sleep_start + j, cx + 1, label,
                             curses.color_pair(C_UP),
                             max_x=max_col)

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

    # -- Compute bottom-pinned tiers first so CPU can absorb slack --
    requests: list[dict] = state.get("requests", [])
    total_requests = len(requests)
    all_models = list(models.items())
    all_models.sort(key=lambda x: x[1].get("pinned_cpu_bytes", 0), reverse=True)
    n_saved_entries = max(len(all_models), 1)
    saved_tier_rows = 1 + n_saved_entries + 1 + 1
    saved_start = h - saved_tier_rows

    # request tier: separator + header + data rows
    natural_req_data = max(saved_start - row - 2, 0)  # -2 for separator+header
    avail_rows = natural_req_data
    if req_rows is not None:
        avail_rows = min(avail_rows, req_rows)
    req_header_y = saved_start - avail_rows - 2  # separator line
    req_data_y = req_header_y + 2                 # first data row

    # -- CPU tier (fills from current row to just above requests separator) --
    cpu_models = sorted(checkpoint_models,
                        key=lambda x: x[1].get("pinned_cpu_bytes", 0), reverse=True)

    if cpu_models:
        for mid, info in cpu_models:
            if row >= req_header_y:
                break
            s = info.get("state", "checkpoint")
            if s == "wait":
                spin = _SPINNER[tick % len(_SPINNER)]
                label = f"{spin} {mid}"
                colour = curses.color_pair(C_WAIT)
            else:
                label = mid
                colour = curses.color_pair(C_CHECKPOINT)
            label += _pinned_str(info.get("pinned_cpu_bytes", 0))
            _safe_addstr(win, row, 2, label, colour)
            row += 1
    else:
        _safe_addstr(win, row, 2, "—", curses.A_DIM)
        row += 1

    # -- Requests separator + header (pinned position) --
    _safe_addstr(win, req_header_y, 0, "╠" + "═" * inner_w + "╣", border)
    active_requests = sum(
        1 for r in requests if r.get("state") in ("waiting", "generating")
    )
    req_label = " Requests "
    _safe_addstr(win, req_header_y + 1, 1, req_label,
                 curses.color_pair(C_SECTION) | curses.A_BOLD)
    req_header_extra = 0
    if active_requests:
        active_label = f" ({active_requests} active)"
        _safe_addstr(win, req_header_y + 1, 1 + len(req_label), active_label,
                     curses.color_pair(C_SECTION) | curses.A_DIM)
        req_header_extra = len(active_label)
    tail_tag = "  tail" if follow_newest else "  scroll"
    _safe_addstr(win, req_header_y + 1,
                 1 + len(req_label) + req_header_extra, tail_tag,
                 curses.color_pair(C_SECTION) | curses.A_DIM)

    # -- Image Cache tier (pinned to absolute bottom) --
    _safe_addstr(win, saved_start, 0, "╠" + "═" * inner_w + "╣", border)
    saved_label = " Image Cache "
    _safe_addstr(win, saved_start + 1, 1, saved_label,
                 curses.color_pair(C_SECTION) | curses.A_BOLD)
    image_cache = state.get("image_cache", "")
    if image_cache:
        _safe_addstr(win, saved_start + 1, 1 + len(saved_label),
                     f" {image_cache}",
                     curses.color_pair(C_SECTION) | curses.A_DIM)

    srow = saved_start + 2
    if all_models:
        for mid, info in all_models:
            vc = info.get("vllm_config", {})
            model_path = vc.get("model", "")
            label = f"{mid}{_pinned_str(info.get('pinned_cpu_bytes', 0))}  {model_path}"
            if info.get("state") == "saved":
                colour = curses.color_pair(C_SAVED)
            else:
                colour = curses.color_pair(C_SAVED) | curses.A_DIM
            _safe_addstr(win, srow, 2, label, colour)
            srow += 1
    else:
        _safe_addstr(win, srow, 2, "—", curses.A_DIM)
        srow += 1

    if srow < h:
        _safe_addstr(win, srow, 0, "╚" + "═" * inner_w + "╝", border)

    # -- Request data rows --
    row = req_data_y
    eff_scroll = 0

    if requests and avail_rows > 0:
        if follow_newest:
            eff_scroll = max(0, total_requests - avail_rows)
        else:
            eff_scroll = min(req_scroll, max(0, total_requests - avail_rows))
        visible = requests[eff_scroll:eff_scroll + avail_rows]
        row = req_data_y
        for req in visible:
            if row >= saved_start:
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
            model_col = mid
            parts = [f"#{rid + 1}"]
            if tlog:
                parts.append(tlog)
            parts.append(f"{model_col:<12s}")

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

    # -- Left/right borders (full height up to bottom border) --
    border_bottom = min(srow, h - 1)
    for y in range(1, border_bottom):
        _safe_addstr(win, y, 0, "║", border)
        _safe_addstr(win, y, w - 1, "║", border)

    win.noutrefresh()
    return total_requests, avail_rows, eff_scroll


# ---------------------------------------------------------------------------
# JSONL loader
# ---------------------------------------------------------------------------

def _load_recording(path: str) -> list[tuple[float, dict]]:
    entries: list[tuple[float, dict]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            entries.append((rec["t"], rec["state"]))
    return entries


# ---------------------------------------------------------------------------
# Curses helpers
# ---------------------------------------------------------------------------

def _safe(win, y: int, x: int, text: str, attr=0):
    h, w = win.getmaxyx()
    if y < 0 or y >= h or x >= w:
        return
    text = text[: w - x]
    try:
        win.addstr(y, x, text, attr)
    except curses.error:
        pass


def _draw_box(win, title: str = ""):
    h, w = win.getmaxyx()
    border = curses.color_pair(C_CHART_BORDER) | curses.A_DIM
    _safe(win, 0, 0, "┌" + "─" * (w - 2) + "┐", border)
    for row in range(1, h - 1):
        _safe(win, row, 0, "│", border)
        _safe(win, row, w - 1, "│", border)
    _safe(win, h - 1, 0, "└" + "─" * (w - 2) + "┘", border)
    if title:
        _safe(win, 0, max(1, (w - len(title)) // 2), title,
              curses.color_pair(C_CHART_TITLE) | curses.A_BOLD)


# ---------------------------------------------------------------------------
# Braille scatter helpers
# ---------------------------------------------------------------------------

_BRAILLE_BASE = 0x2800
_BRAILLE_DOT = [
    [0x01, 0x08],
    [0x02, 0x10],
    [0x04, 0x20],
    [0x40, 0x80],
]


def _plot_braille_dot(cells, cell_attr, px, py, x_max, y_max_display,
                      dot_w, dot_h, attr):
    dx = int(px / max(x_max, 0.001) * (dot_w - 1))
    dy = int(py / max(y_max_display, 0.001) * (dot_h - 1))
    dy = dot_h - 1 - dy
    dx = max(0, min(dot_w - 1, dx))
    dy = max(0, min(dot_h - 1, dy))

    cell_c = dx // 2
    cell_r = dy // 4
    sub_r = dy % 4
    sub_c = dx % 2

    key = (cell_r, cell_c)
    cells[key] = cells.get(key, 0) | _BRAILLE_DOT[sub_r][sub_c]
    if key not in cell_attr or (attr & curses.A_BOLD):
        cell_attr[key] = attr


# ---------------------------------------------------------------------------
# Scatter plot -- wait_s only, two files overlaid
# ---------------------------------------------------------------------------

def _draw_scatter(win, state_a: dict | None, state_b: dict | None,
                  now_rel_a: float | None, now_rel_b: float | None):
    win.erase()
    h, w = win.getmaxyx()
    if h < 4 or w < 12:
        return

    _draw_box(win, " Time to First Token (TTFT) ")

    y_label_w = 6
    plot_x0 = 1 + y_label_w
    plot_y0 = 1
    plot_w = w - plot_x0 - 2
    plot_h = h - 3
    if plot_w < 4 or plot_h < 2:
        return

    def _extract(state):
        if state is None:
            return [], 0.0
        reqs = state.get("requests", [])
        has_sub = [r for r in reqs if r.get("submit_rel_s") is not None]
        if not has_sub:
            return [], 0.0
        t_origin = min(r["submit_rel_s"] for r in has_sub)
        return has_sub, t_origin

    reqs_a, origin_a = _extract(state_a)
    reqs_b, origin_b = _extract(state_b)

    if not reqs_a and not reqs_b:
        _safe(win, h // 2, max(1, (w - 10) // 2), "no data",
              curses.color_pair(C_CHART_LABEL) | curses.A_DIM)
        win.noutrefresh()
        return

    all_wait = ([r.get("wait_s") or 0 for r in reqs_a] +
                [r.get("wait_s") or 0 for r in reqs_b])
    all_x = ([r["submit_rel_s"] - origin_a for r in reqs_a] +
             [r["submit_rel_s"] - origin_b for r in reqs_b])
    if now_rel_a is not None:
        all_x.append(now_rel_a)
    if now_rel_b is not None:
        all_x.append(now_rel_b)

    x_max = max(max(all_x), 1.0) if all_x else 1.0
    y_max = max(max(all_wait), 1.0) if all_wait else 1.0
    y_max_display = y_max * 1.05 + 0.1

    # Y-axis ticks
    step = max(1, round(y_max_display / 5))
    for tick_val in range(0, int(y_max_display) + step, step):
        frac = tick_val / y_max_display
        row = plot_y0 + plot_h - 1 - int(frac * (plot_h - 1))
        if plot_y0 <= row < plot_y0 + plot_h:
            _safe(win, row, 1, f"{tick_val:>4}s ",
                  curses.color_pair(C_CHART_LABEL) | curses.A_DIM)
            _safe(win, row, plot_x0, "·" * plot_w,
                  curses.color_pair(C_CHART_AXIS) | curses.A_DIM)

    # X-axis
    _safe(win, h - 2, plot_x0, f"0 ─ {x_max:.0f}s",
          curses.color_pair(C_CHART_LABEL) | curses.A_DIM)

    cell_char: dict[tuple[int, int], str] = {}
    cell_attr: dict[tuple[int, int], int] = {}

    gen_attr = curses.color_pair(C_RUNNING) | curses.A_BOLD

    def _scatter_file(reqs, t_origin, wait_attr, done_attr, marker):
        for r in reqs:
            rx = r["submit_rel_s"] - t_origin
            wait_s = r.get("wait_s") or 0
            st = r.get("state", "done")
            col = int(rx / max(x_max, 0.001) * (plot_w - 1))
            row = int(wait_s / max(y_max_display, 0.001) * (plot_h - 1))
            row = plot_h - 1 - row
            col = max(0, min(plot_w - 1, col))
            row = max(0, min(plot_h - 1, row))
            key = (row, col)
            if st == "generating":
                attr = gen_attr
            elif st == "waiting":
                attr = wait_attr
            else:
                attr = done_attr
            cell_char[key] = marker
            cell_attr[key] = attr

    _scatter_file(reqs_a, origin_a,
                  curses.color_pair(C_CASE1) | curses.A_BOLD,
                  curses.color_pair(C_CASE1_DIM) | curses.A_DIM,
                  "●")
    _scatter_file(reqs_b, origin_b,
                  curses.color_pair(C_CASE2) | curses.A_BOLD,
                  curses.color_pair(C_CASE2_DIM) | curses.A_DIM,
                  "●")

    for (cr, cc), ch in cell_char.items():
        _safe(win, plot_y0 + cr, plot_x0 + cc, ch,
              cell_attr.get((cr, cc), curses.color_pair(C_CHART_LABEL)))

    # Legend
    leg1 = " ● Case 1"
    leg2 = "  ● Case 2"
    lx = max(plot_x0, plot_x0 + plot_w - len(leg1) - len(leg2))
    _safe(win, h - 2, lx, leg1,
          curses.color_pair(C_CASE1) | curses.A_BOLD)
    _safe(win, h - 2, lx + len(leg1), leg2,
          curses.color_pair(C_CASE2) | curses.A_BOLD)

    win.noutrefresh()


# ---------------------------------------------------------------------------
# Line chart -- GPU utilization only, two files overlaid
# ---------------------------------------------------------------------------

def _draw_util_chart(win,
                     util_a: list[tuple[float, float]],
                     util_b: list[tuple[float, float]],
                     n_gpus_a: int, n_gpus_b: int,
                     window: float = 30.0):
    win.erase()
    h, w = win.getmaxyx()
    if h < 4 or w < 12:
        return

    _draw_box(win, f" GPU Utilization ({window:.0f}s window) ")

    y_label_w = 5
    plot_x0 = 1 + y_label_w
    plot_y0 = 1
    plot_w = w - plot_x0 - 2
    plot_h = h - 3
    if plot_w < 4 or plot_h < 2:
        return

    # Y-axis: 0-100%
    for pct in (0, 25, 50, 75, 100):
        frac = pct / 100.0
        row = plot_y0 + plot_h - 1 - int(frac * (plot_h - 1))
        if plot_y0 <= row < plot_y0 + plot_h:
            _safe(win, row, 1, f"{pct:>3}% ",
                  curses.color_pair(C_CHART_LABEL) | curses.A_DIM)
            _safe(win, row, plot_x0, "·" * plot_w,
                  curses.color_pair(C_CHART_AXIS) | curses.A_DIM)

    if not util_a and not util_b:
        _safe(win, h // 2, max(1, (w - 10) // 2), "no data",
              curses.color_pair(C_CHART_LABEL) | curses.A_DIM)
        win.noutrefresh()
        return

    all_ts = ([t for t, _ in util_a] if util_a else []) + \
             ([t for t, _ in util_b] if util_b else [])
    t_max = max(all_ts) if all_ts else 1.0

    _BR_DOT = [
        [0x01, 0x08],
        [0x02, 0x10],
        [0x04, 0x20],
        [0x40, 0x80],
    ]

    def _draw_series(history, attr, cells, cell_attr):
        if not history or plot_w <= 0:
            return
        dot_w = plot_w * 2
        dot_h = plot_h * 4

        bins: list[list[float]] = [[] for _ in range(plot_w)]
        for t, v in history:
            col = int(t / max(t_max, 0.001) * (plot_w - 1))
            col = max(0, min(plot_w - 1, col))
            bins[col].append(v)

        # Build dense Y values with interpolation
        col_y: list[float | None] = [None] * plot_w
        for ci, vals in enumerate(bins):
            if vals:
                col_y[ci] = sum(vals) / len(vals)
        prev_ci = None
        for ci in range(plot_w):
            if col_y[ci] is not None:
                if prev_ci is not None and ci - prev_ci > 1:
                    y0, y1 = col_y[prev_ci], col_y[ci]
                    for fi in range(prev_ci + 1, ci):
                        frac = (fi - prev_ci) / (ci - prev_ci)
                        col_y[fi] = y0 + (y1 - y0) * frac
                prev_ci = ci

        prev_dy = None
        for ci in range(plot_w):
            if col_y[ci] is None:
                prev_dy = None
                continue
            frac = max(0.0, min(1.0, col_y[ci] / 100.0))
            dy = int(frac * (dot_h - 1))
            dy = dot_h - 1 - dy
            dy = max(0, min(dot_h - 1, dy))

            # Interpolate sub-pixel between consecutive columns
            if prev_dy is not None and prev_dy != dy:
                steps = abs(dy - prev_dy)
                for si in range(steps + 1):
                    iy = round(prev_dy + (dy - prev_dy) * si / steps)
                    cell_r = iy // 4
                    sub_r = iy % 4
                    dx = ci * 2
                    cell_c = dx // 2
                    sub_c = dx % 2
                    key = (cell_r, cell_c)
                    cells[key] = cells.get(key, 0) | _BR_DOT[sub_r][sub_c]
                    cell_attr[key] = attr
            else:
                cell_r = dy // 4
                sub_r = dy % 4
                dx = ci * 2
                cell_c = dx // 2
                sub_c = dx % 2
                key = (cell_r, cell_c)
                cells[key] = cells.get(key, 0) | _BR_DOT[sub_r][sub_c]
                cell_attr[key] = attr
            prev_dy = dy

    br_cells: dict[tuple[int, int], int] = {}
    br_attr: dict[tuple[int, int], int] = {}

    _draw_series(util_a, curses.color_pair(C_CASE1) | curses.A_BOLD,
                 br_cells, br_attr)
    _draw_series(util_b, curses.color_pair(C_CASE2) | curses.A_BOLD,
                 br_cells, br_attr)

    for (cr, cc), bits in br_cells.items():
        ch = chr(0x2800 + bits)
        _safe(win, plot_y0 + cr, plot_x0 + cc, ch,
              br_attr.get((cr, cc), curses.color_pair(C_CHART_LABEL)))

    # Legend + X-axis
    legend_row = h - 2
    _safe(win, legend_row, plot_x0, f"0 ─ {t_max:.0f}s",
          curses.color_pair(C_CHART_LABEL) | curses.A_DIM)
    leg1 = " ● Case 1"
    leg2 = "  ● Case 2"
    lx = max(plot_x0, plot_x0 + plot_w - len(leg1) - len(leg2))
    _safe(win, legend_row, lx, leg1,
          curses.color_pair(C_CASE1) | curses.A_BOLD)
    _safe(win, legend_row, lx + len(leg1), leg2,
          curses.color_pair(C_CASE2) | curses.A_BOLD)

    win.noutrefresh()


# ---------------------------------------------------------------------------
# Line chart -- GPU efficiency, two files overlaid
# ---------------------------------------------------------------------------

def _draw_efficiency_chart(win,
                           eff_a: list[tuple[float, float]],
                           eff_b: list[tuple[float, float]],
                           window: float = 30.0):
    win.erase()
    h, w = win.getmaxyx()
    if h < 4 or w < 12:
        return

    _draw_box(win, f" Efficiency ({window:.0f}s window) ")

    y_label_w = 5
    plot_x0 = 1 + y_label_w
    plot_y0 = 1
    plot_w = w - plot_x0 - 2
    plot_h = h - 3
    if plot_w < 4 or plot_h < 2:
        return

    for pct in (0, 25, 50, 75, 100):
        frac = pct / 100.0
        row = plot_y0 + plot_h - 1 - int(frac * (plot_h - 1))
        if plot_y0 <= row < plot_y0 + plot_h:
            _safe(win, row, 1, f"{pct:>3}% ",
                  curses.color_pair(C_CHART_LABEL) | curses.A_DIM)
            _safe(win, row, plot_x0, "·" * plot_w,
                  curses.color_pair(C_CHART_AXIS) | curses.A_DIM)

    if not eff_a and not eff_b:
        _safe(win, h // 2, max(1, (w - 10) // 2), "no data",
              curses.color_pair(C_CHART_LABEL) | curses.A_DIM)
        win.noutrefresh()
        return

    all_ts = ([t for t, _ in eff_a] if eff_a else []) + \
             ([t for t, _ in eff_b] if eff_b else [])
    t_max = max(all_ts) if all_ts else 1.0

    _BR_DOT = [
        [0x01, 0x08],
        [0x02, 0x10],
        [0x04, 0x20],
        [0x40, 0x80],
    ]

    def _draw_series(history, attr, cells, cell_attr):
        if not history or plot_w <= 0:
            return
        dot_w = plot_w * 2
        dot_h = plot_h * 4

        bins: list[list[float]] = [[] for _ in range(plot_w)]
        for t, v in history:
            col = int(t / max(t_max, 0.001) * (plot_w - 1))
            col = max(0, min(plot_w - 1, col))
            bins[col].append(v)

        col_y: list[float | None] = [None] * plot_w
        for ci, vals in enumerate(bins):
            if vals:
                col_y[ci] = sum(vals) / len(vals)
        prev_ci = None
        for ci in range(plot_w):
            if col_y[ci] is not None:
                if prev_ci is not None and ci - prev_ci > 1:
                    y0, y1 = col_y[prev_ci], col_y[ci]
                    for fi in range(prev_ci + 1, ci):
                        frac = (fi - prev_ci) / (ci - prev_ci)
                        col_y[fi] = y0 + (y1 - y0) * frac
                prev_ci = ci

        prev_dy = None
        for ci in range(plot_w):
            if col_y[ci] is None:
                prev_dy = None
                continue
            frac = max(0.0, min(1.0, col_y[ci] / 100.0))
            dy = int(frac * (dot_h - 1))
            dy = dot_h - 1 - dy
            dy = max(0, min(dot_h - 1, dy))

            if prev_dy is not None and prev_dy != dy:
                steps = abs(dy - prev_dy)
                for si in range(steps + 1):
                    iy = round(prev_dy + (dy - prev_dy) * si / steps)
                    cell_r = iy // 4
                    sub_r = iy % 4
                    dx = ci * 2
                    cell_c = dx // 2
                    sub_c = dx % 2
                    key = (cell_r, cell_c)
                    cells[key] = cells.get(key, 0) | _BR_DOT[sub_r][sub_c]
                    cell_attr[key] = attr
            else:
                cell_r = dy // 4
                sub_r = dy % 4
                dx = ci * 2
                cell_c = dx // 2
                sub_c = dx % 2
                key = (cell_r, cell_c)
                cells[key] = cells.get(key, 0) | _BR_DOT[sub_r][sub_c]
                cell_attr[key] = attr
            prev_dy = dy

    br_cells: dict[tuple[int, int], int] = {}
    br_attr: dict[tuple[int, int], int] = {}

    _draw_series(eff_a, curses.color_pair(C_CASE1) | curses.A_BOLD,
                 br_cells, br_attr)
    _draw_series(eff_b, curses.color_pair(C_CASE2) | curses.A_BOLD,
                 br_cells, br_attr)

    for (cr, cc), bits in br_cells.items():
        ch = chr(0x2800 + bits)
        _safe(win, plot_y0 + cr, plot_x0 + cc, ch,
              br_attr.get((cr, cc), curses.color_pair(C_CHART_LABEL)))

    legend_row = h - 2
    _safe(win, legend_row, plot_x0, f"0 ─ {t_max:.0f}s",
          curses.color_pair(C_CHART_LABEL) | curses.A_DIM)
    leg1 = " ● Case 1"
    leg2 = "  ● Case 2"
    lx = max(plot_x0, plot_x0 + plot_w - len(leg1) - len(leg2))
    _safe(win, legend_row, lx, leg1,
          curses.color_pair(C_CASE1) | curses.A_BOLD)
    _safe(win, legend_row, lx + len(leg1), leg2,
          curses.color_pair(C_CASE2) | curses.A_BOLD)

    win.noutrefresh()


# ---------------------------------------------------------------------------
# Per-file replay state
# ---------------------------------------------------------------------------

class _ReplayStream:
    def __init__(self, entries: list[tuple[float, dict]]):
        self.entries = entries
        self.idx = 0
        self.util_history: list[tuple[float, float]] = []
        self.hist_wall_t0: float | None = None
        self.hist_wall_offset: float = 0.0
        self.now_rel: float | None = None
        self.req_scroll: int = 0
        self.prev_total_requests: int = 0
        self.follow_newest: bool = True

    @property
    def state(self) -> dict:
        return self.entries[self.idx][1]

    def advance(self, elapsed: float):
        while (self.idx < len(self.entries) - 1
               and self.entries[self.idx + 1][0] <= elapsed):
            self.idx += 1

    def update_util(self, window: float, speed: float = 1.0):
        state = self.state
        reqs = state.get("requests", [])

        submit_times = [r["submit_rel_s"] for r in reqs
                        if r.get("submit_rel_s") is not None]

        if not self.now_rel and not submit_times:
            return

        if submit_times:
            t_origin = min(submit_times)
            req_now = max(
                (r.get("submit_rel_s", 0) or 0)
                + (r.get("wait_s", 0) or 0)
                + (r.get("gen_s", 0) or 0)
                - t_origin
                for r in reqs
            )
            if self.hist_wall_t0 is None:
                self.hist_wall_t0 = time.monotonic()
                self.hist_wall_offset = req_now
                _backfill_util(state, self.util_history, window=window)
            self.now_rel = self.hist_wall_offset + (time.monotonic() - self.hist_wall_t0) * speed
            util = _compute_utilization(state, self.now_rel, window=window)
            if util is None and self.util_history:
                util = self.util_history[-1][1]
            if util is not None:
                self.util_history.append((self.now_rel, util))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _main(stdscr, args):
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(100)
    _init_colours()
    curses.mousemask(curses.ALL_MOUSE_EVENTS | curses.REPORT_MOUSE_POSITION)

    entries_a = _load_recording(args.file_a)
    entries_b = _load_recording(args.file_b)
    if not entries_a or not entries_b:
        msg = []
        if not entries_a:
            msg.append(f"Empty: {args.file_a}")
        if not entries_b:
            msg.append(f"Empty: {args.file_b}")
        stdscr.addstr(0, 0, "  ".join(msg))
        stdscr.getch()
        return

    sa = _ReplayStream(entries_a)
    sb = _ReplayStream(entries_b)
    window = args.window
    wall_t0 = time.monotonic()

    tick = 0
    SCROLL_STEP = 3
    both_done = False

    while True:
        key = stdscr.getch()
        if key == ord("q") or key == ord("Q") or key == 27:
            break
        if key in (ord("t"), ord("T")):
            sa.follow_newest = True
            sb.follow_newest = True
        if key == curses.KEY_RESIZE:
            curses.update_lines_cols()
        if key == curses.KEY_MOUSE:
            try:
                _, mx, _, _, bstate = curses.getmouse()
                term_h, term_w = stdscr.getmaxyx()
                col_w = term_w * 4 // 15
                if bstate & curses.BUTTON4_PRESSED:
                    if mx < col_w:
                        sa.follow_newest = False
                        sa.req_scroll = max(0, sa.req_scroll - SCROLL_STEP)
                    elif mx < col_w * 2:
                        sb.follow_newest = False
                        sb.req_scroll = max(0, sb.req_scroll - SCROLL_STEP)
                elif bstate & curses.BUTTON5_PRESSED:
                    if mx < col_w:
                        sa.follow_newest = False
                        sa.req_scroll += SCROLL_STEP
                    elif mx < col_w * 2:
                        sb.follow_newest = False
                        sb.req_scroll += SCROLL_STEP
            except curses.error:
                pass

        # Advance both streams in sync
        elapsed = (time.monotonic() - wall_t0) * args.speed
        sa.advance(elapsed)
        sb.advance(elapsed)
        sa.update_util(window, speed=args.speed)
        sb.update_util(window, speed=args.speed)

        # Layout: two narrower dashboard columns + wider right charts (2 panes)
        term_h, term_w = stdscr.getmaxyx()
        col_w = term_w * 4 // 15
        right_w = term_w - col_w * 2
        right_top_h = term_h // 2
        right_bot_h = term_h - right_top_h

        try:
            win_a = stdscr.subwin(term_h, col_w, 0, 0)
            win_b = stdscr.subwin(term_h, col_w, 0, col_w)
            win_scatter = stdscr.subwin(right_top_h, right_w, 0, col_w * 2)
            win_util = stdscr.subwin(right_bot_h, right_w,
                                     right_top_h, col_w * 2)
        except curses.error:
            stdscr.erase()
            stdscr.addstr(0, 0, "Terminal too small")
            stdscr.noutrefresh()
            curses.doupdate()
            time.sleep(0.1)
            tick += 1
            continue

        # Use the same request-list height for both panels
        rr_a = _measure_req_rows(win_a, sa.state)
        rr_b = _measure_req_rows(win_b, sb.state)
        shared_req_rows = min(rr_a, rr_b)

        # Dashboard A
        total_a, _, eff_a = _render(
            win_a, sa.state, connected=True, tick=tick,
            req_scroll=sa.req_scroll, follow_newest=sa.follow_newest,
            poll_interval=args.interval, req_rows=shared_req_rows,
            panel_title="Case 1",
        )
        if total_a > sa.prev_total_requests:
            sa.follow_newest = True
        sa.prev_total_requests = total_a
        sa.req_scroll = eff_a

        # Dashboard B
        total_b, _, eff_b = _render(
            win_b, sb.state, connected=True, tick=tick,
            req_scroll=sb.req_scroll, follow_newest=sb.follow_newest,
            poll_interval=args.interval, req_rows=shared_req_rows,
            panel_title="Case 2",
        )
        if total_b > sb.prev_total_requests:
            sb.follow_newest = True
        sb.prev_total_requests = total_b
        sb.req_scroll = eff_b

        # Scatter: wait_s only, both files
        _draw_scatter(win_scatter, sa.state, sb.state,
                      sa.now_rel, sb.now_rel)

        # Utilization: both files
        n_gpus_a = len(sa.state.get("gpu_ids", []))
        n_gpus_b = len(sb.state.get("gpu_ids", []))
        _draw_util_chart(win_util, sa.util_history, sb.util_history,
                         n_gpus_a, n_gpus_b, window=window)

        curses.doupdate()
        tick += 1

        a_done = sa.idx >= len(sa.entries) - 1
        b_done = sb.idx >= len(sb.entries) - 1
        if a_done and b_done:
            if not both_done:
                both_done = True
            else:
                while True:
                    k = stdscr.getch()
                    if k == ord("q") or k == ord("Q") or k == 27:
                        return
                    time.sleep(0.1)

        time.sleep(args.interval)


def main():
    parser = argparse.ArgumentParser(
        description="Compare two dashboard recordings side by side")
    parser.add_argument("file_a", help="JSONL recording A (dashboard.py --record)")
    parser.add_argument("file_b", help="JSONL recording B (dashboard.py --record)")
    parser.add_argument("--interval", type=float, default=0.1,
                        help="render interval in seconds (default: 0.1)")
    parser.add_argument("--window", type=float, default=30.0,
                        help="rolling window for utilization (default: 30s)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="replay speed multiplier (default: 1.0)")
    args = parser.parse_args()
    curses.wrapper(lambda stdscr: _main(stdscr, args))


if __name__ == "__main__":
    main()
