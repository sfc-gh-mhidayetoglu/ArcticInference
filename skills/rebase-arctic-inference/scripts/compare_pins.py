#!/usr/bin/env python3
"""Cross-version consistency check for the ArcticInference rebase skill.

Discovers every pinned vLLM version that exists as a branch (`main` + each
`rebase/vllm_v*`, local or remote) plus (optionally) the in-progress working
tree, runs a set of *probes* against each, and prints a version-sorted table so
you can confirm a new pin continues each patched surface's trajectory sensibly.
It also emits advisory anomaly flags (a surface that changed every prior bump
but not this one -> likely a missed port; one that vanished or reverted).

The probes live in `evolution_probes.json` next to this script and are the
skill's durable memory of which recurring drift surfaces to track. ADD A PROBE
whenever you fix a novel recurring drift.

Usage:
    python compare_pins.py [--repo PATH] [--probes PATH] [--include-worktree]
                           [--write evolution.md]
"""
import argparse
import json
import os
import re
import subprocess
import sys


def git(repo, *args):
    r = subprocess.run(["git", "-C", repo, *args],
                       capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def parse_pin(text):
    """Return the plugin's own pin (the `vllm = [...]` extra), not a comment
    or the separate `embedding` extra's older vLLM pin."""
    if not text:
        return None
    m = re.search(
        r"(?m)^\s*vllm\s*=\s*\[\s*(?:#[^\n]*\n\s*)*['\"]vllm==([0-9][0-9A-Za-z.\-]*)",
        text)
    if m:
        return m.group(1)
    for line in text.splitlines():  # fallback: first quoted dep, skip comments
        s = line.strip()
        if s.startswith("#"):
            continue
        mm = re.search(r"['\"]vllm==([0-9][0-9A-Za-z.\-]*)", s)
        if mm:
            return mm.group(1)
    return None


def version_key(pin):
    parts = re.findall(r"\d+", pin or "")
    return tuple(int(p) for p in parts) if parts else (0,)


def discover_refs(repo):
    """{pin: ref} for main + every rebase/vllm_v* branch, preferring local heads."""
    candidates = []
    for line in (git(repo, "for-each-ref", "--format=%(refname)",
                     "refs/heads/", "refs/remotes/") or "").splitlines():
        short = line.replace("refs/heads/", "").replace("refs/remotes/", "")
        base = short.split("/", 1)[-1] if short.startswith(("origin/",)) else short
        if base == "main" or re.search(r"rebase/vllm[_/]?v?\d", base):
            candidates.append(short)
    # rank: local head > remote; main last within a pin tie is fine
    def rank(ref):
        return (0 if not ref.startswith(("origin/", "upstream/")) else 1, len(ref))
    by_pin = {}
    for ref in sorted(set(candidates), key=rank):
        pin = parse_pin(git(repo, "show", f"{ref}:pyproject.toml"))
        if pin and pin not in by_pin:
            by_pin[pin] = ref
    return by_pin


def discover_history(repo, ref="HEAD"):
    """{pin: commit} — the first commit that introduced each distinct pin along
    `ref`'s first-parent history (so every past bump becomes a column, even the
    ones with no surviving branch)."""
    commits = git(repo, "log", "--first-parent", "--reverse", "--pretty=%H",
                  ref, "--", "pyproject.toml")
    by_pin = {}
    for c in (commits or "").splitlines():
        pin = parse_pin(git(repo, "show", f"{c}:pyproject.toml"))
        if pin and pin not in by_pin:
            by_pin[pin] = c
    return by_pin


def extract(content, probe):
    if content is None:
        return None
    rx = re.compile(probe["regex"], re.DOTALL if probe.get("dotall") else 0)
    hits = []
    for m in rx.finditer(content):
        hits.append((m.group(1) if m.groups() else m.group(0)).strip())
        if not probe.get("all"):
            break
    if not hits:
        return None
    return " ¦ ".join(dict.fromkeys(hits))  # dedupe, preserve order


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.getcwd())
    ap.add_argument("--probes", default=os.path.join(here, "evolution_probes.json"))
    ap.add_argument("--include-worktree", action="store_true",
                    help="add a column for the current on-disk working tree")
    ap.add_argument("--history", nargs="?", const="HEAD", metavar="REF",
                    help="also mine past bumps from REF's first-parent history "
                         "(default HEAD) so every historical pin becomes a column")
    ap.add_argument("--write", metavar="MD",
                    help="also write the table to this markdown file (ledger)")
    args = ap.parse_args()

    probes = json.load(open(args.probes))["probes"]

    # pin -> (kind, source); branches win over historical commits for the same pin
    resolved = {}
    if args.history:
        for pin, sha in discover_history(args.repo, args.history).items():
            resolved[pin] = ("commit", sha)
    for pin, ref in discover_refs(args.repo).items():
        resolved[pin] = ("ref", ref)  # override any commit entry

    columns = []  # (label, kind, source_or_None)
    for pin in sorted(resolved, key=version_key):
        columns.append((pin, resolved[pin][0], resolved[pin][1]))

    if args.include_worktree:
        wt_pin = parse_pin(open(os.path.join(args.repo, "pyproject.toml")).read())
        label = f"{wt_pin} (worktree)" if wt_pin else "worktree"
        # place after its version among refs
        idx = len(columns)
        for i, (p, _, _) in enumerate(columns):
            if version_key(wt_pin) < version_key(p):
                idx = i
                break
        columns.insert(idx, (label, "worktree", None))

    if not columns:
        print("No pinned vLLM branches found (main / rebase/vllm_v*).")
        return 1

    # gather content per (column, file) lazily
    cache = {}

    def content_for(kind, ref, path):
        key = (kind, ref, path)
        if key not in cache:
            if kind == "worktree":
                fp = os.path.join(args.repo, path)
                cache[key] = open(fp).read() if os.path.exists(fp) else None
            else:
                cache[key] = git(args.repo, "show", f"{ref}:{path}")
        return cache[key]

    rows = []  # (probe_label, [values per column])
    for probe in probes:
        vals = []
        for (_lbl, kind, ref) in columns:
            vals.append(extract(content_for(kind, ref, probe["file"]), probe))
        rows.append((probe["label"], vals))

    # ---- render table ----
    col_labels = [c[0] for c in columns]
    print("resolved versions:")
    for (lbl, kind, src) in columns:
        if kind == "worktree":
            desc = "(working tree, on disk)"
        elif kind == "commit":
            subj = (git(args.repo, "log", "-1", "--date=short",
                        "--pretty=%h %ad %s", src) or src).strip()
            desc = f"commit {subj}"
        else:
            desc = src
        print(f"  {lbl:<22} {desc}")
    print()
    def cell(v):
        return "—" if v is None else (v if len(v) <= 46 else v[:43] + "...")
    label_w = max([len("probe")] + [len(r[0]) for r in rows])
    col_w = [max(len(col_labels[i]),
                 *[len(cell(r[1][i])) for r in rows]) for i in range(len(columns))]

    def line(label, cells):
        out = label.ljust(label_w) + " | "
        out += " | ".join(cells[i].ljust(col_w[i]) for i in range(len(cells)))
        return out

    print(line("probe", col_labels))
    print("-" * len(line("probe", col_labels)))
    for lbl, vals in rows:
        print(line(lbl, [cell(v) for v in vals]))

    # ---- anomaly flags (advisory; need >=3 versions of history) ----
    flags = []
    for lbl, vals in rows:
        n = len(vals)
        if n < 3:
            continue
        prior_pairs = [(vals[i], vals[i + 1]) for i in range(n - 2)]
        changed_prior = [a != b for a, b in prior_pairs if a is not None and b is not None]
        last_a, last_b = vals[-2], vals[-1]
        if last_b is None and last_a is not None:
            flags.append(f"[VANISHED] '{lbl}': present in {col_labels[-2]} but missing in {col_labels[-1]} — ported/removed?")
        elif last_a is not None and last_b is not None:
            if changed_prior and all(changed_prior) and last_a == last_b:
                flags.append(f"[STALLED] '{lbl}': changed on every prior bump but is UNCHANGED {col_labels[-2]}→{col_labels[-1]} — verify this surface didn't need a port.")
            if last_b in vals[:-2] and last_b != last_a:
                flags.append(f"[REVERTED] '{lbl}': {col_labels[-1]} value matches an older version but differs from {col_labels[-2]} — possible regression.")

    print()
    if flags:
        print("ANOMALY FLAGS (advisory — a human confirms):")
        for f in flags:
            print("  " + f)
    else:
        print("No anomalies: every tracked surface continues its trajectory consistently.")

    # ---- optional markdown ledger ----
    if args.write:
        with open(args.write, "w") as fh:
            fh.write("# vLLM pin evolution ledger (generated by compare_pins.py)\n\n")
            fh.write("| probe | " + " | ".join(col_labels) + " |\n")
            fh.write("|" + "---|" * (len(col_labels) + 1) + "\n")
            for lbl, vals in rows:
                fh.write("| " + lbl + " | " +
                         " | ".join(("—" if v is None else "`" + v + "`") for v in vals) +
                         " |\n")
            if flags:
                fh.write("\n**Anomaly flags:**\n\n")
                for f in flags:
                    fh.write("- " + f + "\n")
        print(f"\nWrote ledger -> {args.write}")

    return 1 if flags else 0


if __name__ == "__main__":
    raise SystemExit(main())
