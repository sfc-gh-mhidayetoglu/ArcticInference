#!/usr/bin/env python3
"""Live terminal monitor for the Orchestrator.

Polls GET /state from the orchestrator's HTTP server and renders two panels:

  Top:    Request Latency scatter plot (elapsed time per request).
  Bottom: GPU Utilization line chart over time.

Usage:
    python monitor.py [--host HOST] [--port PORT] [--interval SECS]
"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
import urllib.error

import os
import sys

import plotext as plt
import plotext._build as _plotext_build

import inspect as _inspect
import textwrap as _textwrap
_src = _textwrap.dedent(_inspect.getsource(_plotext_build.build_class.build_plot))
_src = _src.replace(
    "take_3 = lambda data: (data[ : 3] * 3)[ : 3]",
    "take_3 = lambda data: (data[ : 3] * 3)[ : 3] if data else [ut.no_color] * 3",
)
_src = _src.replace(
    "color =  [[ut.no_color] + take_3(c[s]) for s in Signals if labelled(s)]",
    "color =  [[ut.no_color] + take_3(self.color[s]) for s in Signals if labelled(s)]",
)
_src = _src.replace(
    "style =  [[ut.no_color] + take_3(st[s]) for s in Signals if labelled(s)]",
    "style =  [[ut.no_color] + take_3(self.style[s]) for s in Signals if labelled(s)]",
)
_src = _src.replace(
    "col_start + S + i, row_end - 1 - s, marker[s][i]",
    "col_end - S - 3 - L + S + i, row_end - 1 - s, marker[s][i]",
)
_src = _src.replace(
    "col_start + S + 3, row_end - 1 - s, labels[s]",
    "col_end - L, row_end - 1 - s, labels[s]",
)
_src = _src.replace(
    "col_start, row_end - 1 - s, side[s]",
    "col_end - S - 3 - L, row_end - 1 - s, side[s]",
)
exec(compile(_src, _plotext_build.__file__, "exec"), _plotext_build.__dict__)
_plotext_build.build_class.build_plot = _plotext_build.__dict__["build_plot"]
del _src, _inspect, _textwrap

_CSI_HOME = "\033[H"
_CSI_HIDE_CURSOR = "\033[?25l"
_CSI_SHOW_CURSOR = "\033[?25h"
_CSI_ALT_SCREEN = "\033[?1049h"
_CSI_MAIN_SCREEN = "\033[?1049l"


def _fetch_state(url: str) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=2) as resp:
            return json.loads(resp.read())
    except Exception:
        return None





_ROLLING_WINDOW = 30.0


def _merge_intervals(intervals: list[tuple[float, float]]) -> float:
    """Return total wall-clock duration of a set of (start, end) intervals after merging overlaps."""
    if not intervals:
        return 0.0
    intervals.sort()
    total = 0.0
    cur_start, cur_end = intervals[0]
    for s, e in intervals[1:]:
        if s <= cur_end:
            cur_end = max(cur_end, e)
        else:
            total += cur_end - cur_start
            cur_start, cur_end = s, e
    total += cur_end - cur_start
    return total


def _compute_utilization(state: dict, now_rel: float, window: float = _ROLLING_WINDOW) -> float | None:
    """Compute GPU utilization with overlap-aware per-GPU interval merging.

    Concurrent requests on the same GPU are merged so overlapping generation
    time is only counted once per GPU.
    """
    requests = state.get("requests", [])
    gpu_ids = state.get("gpu_ids", [])
    n_gpus = len(gpu_ids)
    if n_gpus == 0:
        return None

    submit_times = [r["submit_rel_s"] for r in requests if r.get("submit_rel_s") is not None]
    if not submit_times:
        return None
    t_origin = min(submit_times)
    win_start = max(0.0, now_rel - window)

    models = state.get("models", {})
    per_gpu: dict[int, list[tuple[float, float]]] = {g: [] for g in gpu_ids}

    for r in requests:
        gs = r.get("gen_start_rel_s")
        if gs is None:
            continue
        gen_start = gs - t_origin
        done = r.get("done_rel_s")
        if done is not None:
            gen_end = done - t_origin
        else:
            gen_end = gen_start + (r.get("gen_s") or 0)

        clamped_start = max(gen_start, win_start)
        clamped_end = min(gen_end, now_rel)
        if clamped_end <= clamped_start:
            continue

        mid = r.get("model_id")
        gpu = models.get(mid, {}).get("gpu") if mid else None
        if gpu is not None and gpu in per_gpu:
            per_gpu[gpu].append((clamped_start, clamped_end))
        else:
            for g in per_gpu:
                per_gpu[g].append((clamped_start, clamped_end))
                break

    total_gen = sum(_merge_intervals(intervals) for intervals in per_gpu.values())
    return min(100.0, total_gen / (window * n_gpus) * 100.0)


def _compute_efficiency(state: dict, now_rel: float, window: float = _ROLLING_WINDOW) -> float | None:
    """Compute GPU efficiency: gen_time / (wait_time + gen_time) * 100.

    For each request whose lifetime overlaps the window, clamp wait_s and
    gen_s to the portion inside the window, then:
        eff = sum(clamped_gen) / sum(clamped_wait + clamped_gen) * 100

    A request that is only waiting contributes pure wait time (eff→0).
    """
    requests = state.get("requests", [])

    submit_times = [r["submit_rel_s"] for r in requests if r.get("submit_rel_s") is not None]
    if not submit_times:
        return 0.0
    t_origin = min(submit_times)
    win_start = max(0.0, now_rel - window)

    total_gen = 0.0
    total_time = 0.0

    for r in requests:
        submit = r.get("submit_rel_s")
        if submit is None:
            continue
        submit -= t_origin

        wait_s = r.get("wait_s") or 0
        gen_s = r.get("gen_s") or 0

        req_start = submit
        wait_end = submit + wait_s
        req_end = submit + wait_s + gen_s

        if req_end <= win_start or req_start >= now_rel:
            continue

        clamped_start = max(req_start, win_start)
        clamped_end = min(req_end, now_rel)

        clamped_wait = max(0.0, min(wait_end, clamped_end) - clamped_start)
        clamped_gen = max(0.0, clamped_end - max(wait_end, clamped_start))

        total_gen += clamped_gen
        total_time += clamped_wait + clamped_gen

    if total_time <= 0:
        return None
    return min(100.0, total_gen / total_time * 100.0)


def _backfill_util(state: dict, util_history: list[tuple[float, float]], window: float = _ROLLING_WINDOW) -> None:
    """Populate util_history with historical data from the current state snapshot."""
    requests = state.get("requests", [])
    submit_times = [r["submit_rel_s"] for r in requests if r.get("submit_rel_s") is not None]
    if not submit_times:
        return
    t_origin = min(submit_times)

    sample_points: set[float] = {0.0}
    for r in requests:
        sub = (r.get("submit_rel_s") or 0) - t_origin
        sample_points.add(sub)
        gs = r.get("gen_start_rel_s")
        if gs is not None:
            sample_points.add(gs - t_origin)
        dr = r.get("done_rel_s")
        if dr is not None:
            sample_points.add(dr - t_origin)

    t_max = max(sample_points)
    t = 0.0
    while t <= t_max:
        sample_points.add(round(t, 1))
        t += 1.0

    prev_val = 0.0
    for t in sorted(sample_points):
        util = _compute_utilization(state, t, window=window)
        if util is None:
            util = prev_val
        else:
            prev_val = util
        util_history.append((t, util))


def _backfill_efficiency(state: dict, eff_history: list[tuple[float, float]], window: float = _ROLLING_WINDOW) -> None:
    """Populate eff_history with historical data from the current state snapshot."""
    requests = state.get("requests", [])
    submit_times = [r["submit_rel_s"] for r in requests if r.get("submit_rel_s") is not None]
    if not submit_times:
        return
    t_origin = min(submit_times)

    sample_points: set[float] = {0.0}
    for r in requests:
        sub = (r.get("submit_rel_s") or 0) - t_origin
        sample_points.add(sub)
        gs = r.get("gen_start_rel_s")
        if gs is not None:
            sample_points.add(gs - t_origin)
        dr = r.get("done_rel_s")
        if dr is not None:
            sample_points.add(dr - t_origin)

    t_max = max(sample_points)
    t = 0.0
    while t <= t_max:
        sample_points.add(round(t, 1))
        t += 1.0

    prev_val = 0.0
    for t in sorted(sample_points):
        eff = _compute_efficiency(state, t, window=window)
        if eff is None:
            eff = prev_val
        else:
            prev_val = eff
        eff_history.append((t, eff))


def _build_frame(state: dict, connected: bool, util_history: list[tuple[float, float]] | None = None, eff_history: list[tuple[float, float]] | None = None, now_rel: float | None = None, interval: float = 0.5, window: float = _ROLLING_WINDOW) -> str:
    """Build a two-panel figure: scatter plot (top) + GPU util line (bottom)."""
    term = os.get_terminal_size()
    requests = state.get("requests", [])

    conn_tag = "" if connected else "  [disconnected]"

    margin = 2
    half_h = (term.lines - 3) // 2
    plot_w = term.columns - margin * 2

    # --- Top panel: Request Latency scatter ---
    plt.subplot(1, 1)
    plt.clear_data()
    plt.plotsize(plot_w, half_h)
    plt.theme("dark")

    submit_times = [r["submit_rel_s"] for r in requests if r.get("submit_rel_s") is not None]
    t_origin = min(submit_times) if submit_times else 0

    has_submit = [r for r in requests if r.get("submit_rel_s") is not None]
    waiting = [r for r in has_submit if r.get("state") == "waiting"]
    generating = [r for r in has_submit if r.get("state") == "generating"]
    done = [r for r in has_submit if r.get("state") == "done"]

    def _sub_t(r: dict) -> float:
        return r["submit_rel_s"] - t_origin

    bright_blue_x = [_sub_t(r) for r in waiting]
    bright_blue_y = [r.get("wait_s") or 0 for r in waiting]

    dim_blue_x = [_sub_t(r) for r in generating + done]
    dim_blue_y = [r.get("wait_s") or 0 for r in generating + done]

    bright_green_x = [_sub_t(r) for r in generating]
    bright_green_y = [(r.get("wait_s") or 0) + (r.get("gen_s") or 0) for r in generating]

    dim_green_x = [_sub_t(r) for r in done]
    dim_green_y = [(r.get("wait_s") or 0) + (r.get("gen_s") or 0) for r in done]

    all_x = bright_blue_x + dim_blue_x + bright_green_x + dim_green_x
    all_y = bright_blue_y + dim_blue_y + bright_green_y + dim_green_y

    if all_x:
        x_max = max(all_x)
        y_max = max(all_y) if all_y else 1
        offscreen_x, offscreen_y = [x_max + 1000], [y_max + 1000]

        plt.scatter(dim_blue_x or offscreen_x, dim_blue_y or offscreen_y,
                    color=(40, 70, 130))
        plt.scatter(bright_blue_x or offscreen_x, bright_blue_y or offscreen_y,
                    color=(80, 140, 255), label="waiting")
        plt.scatter(bright_green_x or offscreen_x, bright_green_y or offscreen_y,
                    color=(0, 255, 0), label="generating")
        plt.scatter(dim_green_x or offscreen_x, dim_green_y or offscreen_y,
                    color=(0, 100, 0))

        x_end = max(x_max, now_rel or 0) + 0.5
        plt.xlim(0, x_end)
        y_top = y_max + max(0.1, y_max * 0.05)
        plt.ylim(-0.3, y_top)
        step = max(1, round(y_top / 5))
        y_ticks = list(range(0, int(y_top) + step, step))
        plt.yticks(y_ticks, [str(t) for t in y_ticks])

    plt.title(f"Request Latency (poll {interval}s)" + conn_tag)
    plt.xlabel("time (s)")
    plt.ylabel("elapsed (s)")

    # --- Bottom panel: GPU Utilization line chart ---
    plt.subplot(2, 1)
    plt.clear_data()
    plt.plotsize(plot_w, half_h)
    plt.theme("dark")

    x_max_bottom = 0.0
    if util_history:
        ts = [t for t, _ in util_history]
        vals = [v for _, v in util_history]
        plt.plot(ts, vals, color="blue", label="utilization")
        x_max_bottom = max(x_max_bottom, max(ts))
    if eff_history:
        ts_e = [t for t, _ in eff_history]
        vals_e = [v for _, v in eff_history]
        plt.plot(ts_e, vals_e, color="cyan", label="efficiency")
        x_max_bottom = max(x_max_bottom, max(ts))
    if x_max_bottom > 0:
        plt.xlim(0, x_max_bottom + 0.5)
    plt.ylim(0, 100)
    n_gpus = len(state.get("gpu_ids", []))
    plt.title(f"GPU Utilization / Efficiency ({window:.0f}s window, {n_gpus} GPUs)")
    plt.xlabel("time (s)")
    plt.ylabel("%")

    return _pad_frame(plt.build(), term)


_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def _pad_frame(raw: str, term: os.terminal_size, margin: int = 2) -> str:
    """Add black margins (top/bottom/left/right) and a mid separator."""
    w = term.columns
    bg = "\033[40m"
    rst = "\033[0m"
    black = f"{bg}{' ' * w}{rst}"
    left = f"{bg}{' ' * margin}"

    padded = []
    for line in raw.split("\n"):
        visible_len = len(_ANSI_RE.sub("", line))
        right_fill = max(0, w - margin - visible_len)
        padded.append(f"{left}{line}{bg}{' ' * right_fill}{rst}")

    padded.insert(0, black)
    padded.insert(len(padded) // 2, black)
    while len(padded) < term.lines:
        padded.append(black)
    return "\n".join(padded[:term.lines])


def _paint(frame: str) -> None:
    """Overwrite the screen with *frame* in a single atomic write."""
    sys.stdout.write(_CSI_HOME + frame)
    sys.stdout.flush()


def _load_replay(path: str) -> list[tuple[float, dict]]:
    """Load a JSONL recording into a list of (t, state) pairs."""
    entries: list[tuple[float, dict]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            entries.append((rec["t"], rec["state"]))
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description="Orchestrator live monitor")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8157)
    parser.add_argument("--interval", type=float, default=0.5,
                        help="polling interval in seconds")
    parser.add_argument("--window", type=float, default=_ROLLING_WINDOW,
                        help="rolling window for GPU utilization in seconds")
    parser.add_argument("--record", metavar="FILE", default=None,
                        help="record each polled state snapshot to a JSONL file")
    parser.add_argument("--replay", metavar="FILE", default=None,
                        help="replay from a recorded JSONL file instead of polling")
    args = parser.parse_args()

    replay_entries: list[tuple[float, dict]] | None = None
    if args.replay:
        replay_entries = _load_replay(args.replay)
        if not replay_entries:
            print(f"replay file is empty: {args.replay}", file=sys.stderr)
            return

    url = f"http://{args.host}:{args.port}/state"

    record_file = None
    record_t0: float | None = None
    if args.record:
        record_file = open(args.record, "w")

    sys.stdout.write(_CSI_ALT_SCREEN + _CSI_HIDE_CURSOR)
    sys.stdout.flush()

    last_state: dict | None = None
    util_history: list[tuple[float, float]] = []
    eff_history: list[tuple[float, float]] = []
    wall_t0: float | None = None
    wall_offset: float = 0.0
    prev_t_origin: float | None = None
    prev_num_reqs: int = 0

    replay_idx = 0
    replay_wall_t0: float | None = None

    plt.subplots(2, 1)

    try:
        while True:
            if replay_entries is not None:
                if replay_wall_t0 is None:
                    replay_wall_t0 = time.monotonic()
                elapsed = time.monotonic() - replay_wall_t0
                while replay_idx < len(replay_entries) - 1 and replay_entries[replay_idx + 1][0] <= elapsed:
                    replay_idx += 1
                last_state = replay_entries[replay_idx][1]
                connected = True
            else:
                fresh = _fetch_state(url)
                if fresh is not None:
                    last_state = fresh
                    connected = True
                    if record_file is not None:
                        if record_t0 is None:
                            record_t0 = time.monotonic()
                        t = time.monotonic() - record_t0
                        record_file.write(json.dumps({"t": round(t, 3), "state": fresh}) + "\n")
                        record_file.flush()
                else:
                    connected = False

            if last_state is not None:
                if connected:
                    reqs = last_state.get("requests", [])
                    num_reqs = len(reqs)

                    restarted = False
                    if num_reqs < prev_num_reqs:
                        restarted = True

                    submit_times = [r["submit_rel_s"] for r in reqs if r.get("submit_rel_s") is not None]
                    if submit_times:
                        t_origin = min(submit_times)
                        if prev_t_origin is not None and t_origin != prev_t_origin:
                            restarted = True
                        prev_t_origin = t_origin

                    if restarted:
                        util_history.clear()
                        eff_history.clear()
                        wall_t0 = None

                    prev_num_reqs = num_reqs

                    now_rel = None
                    if submit_times:
                        req_now = max(
                            (r.get("submit_rel_s", 0) or 0) + (r.get("wait_s", 0) or 0) + (r.get("gen_s", 0) or 0) - t_origin
                            for r in reqs
                        )
                        if wall_t0 is None:
                            wall_t0 = time.monotonic()
                            wall_offset = req_now
                            _backfill_util(last_state, util_history, window=args.window)
                            _backfill_efficiency(last_state, eff_history, window=args.window)
                        now_rel = wall_offset + (time.monotonic() - wall_t0)
                        util = _compute_utilization(last_state, now_rel, window=args.window)
                        if util is None and util_history:
                            util = util_history[-1][1]
                        if util is not None:
                            util_history.append((now_rel, util))
                        eff = _compute_efficiency(last_state, now_rel, window=args.window)
                        if eff is None and eff_history:
                            eff = eff_history[-1][1]
                        if eff is not None:
                            eff_history.append((now_rel, eff))
                frame = _build_frame(last_state, connected, util_history, eff_history, now_rel, interval=args.interval, window=args.window)
            else:
                term = os.get_terminal_size()
                margin = 2
                half_h = (term.lines - 3) // 2
                plot_w = term.columns - margin * 2
                plt.subplot(1, 1)
                plt.clear_data()
                plt.plotsize(plot_w, half_h)
                plt.theme("dark")
                plt.title("Waiting for orchestrator...")
                plt.subplot(2, 1)
                plt.clear_data()
                plt.plotsize(plot_w, half_h)
                plt.theme("dark")
                frame = _pad_frame(plt.build(), term)
            _paint(frame)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write(_CSI_SHOW_CURSOR + _CSI_MAIN_SCREEN)
        sys.stdout.flush()
        if record_file is not None:
            record_file.close()


if __name__ == "__main__":
    main()
