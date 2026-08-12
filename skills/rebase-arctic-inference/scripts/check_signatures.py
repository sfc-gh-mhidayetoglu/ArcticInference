#!/usr/bin/env python3
"""Drift detector: compare ArcticPatch override signatures against the target vLLM base.

Usage:
    python check_signatures.py --arctic <arctic_inference_pkg> --vllm <vllm_pkg> \
        [--skip dynasor,embedding]

For every `class X(ArcticPatch[Base]):` in ArcticInference, this compares each
method the patch overrides against `Base.<method>` as defined in the TARGET vLLM
tree, and reports argument-list differences (added/removed/reordered params).

Heuristics / caveats:
- Base classes are indexed by name across the whole vLLM package; if two classes
  share a name the first-found wins (file is printed so you can disambiguate).
- Only positional/keyword arg *names* are compared (not defaults/annotations).
- A method present in the patch but absent from the base is reported as NEW
  (usually fine — Arctic adds helpers — but verify it isn't a typo'd override).
- `ArcticPatch[...]` can target a *module* (e.g. `parallel_state`), not just a
  class. Those show up under "BASE CLASS NOT FOUND"; confirm the module still
  exists rather than assuming a rename.
- Intentional `(self, *args, **kwargs)` passthrough wrappers will flag as a
  mismatch — that is expected; confirm the override forwards to `_orig_*`.
- Always eyeball flagged methods against the real diff; this is a first-pass filter.
"""
import argparse
import ast
import os


def arg_names(func):
    a = func.args
    names = [p.arg for p in a.posonlyargs] + [p.arg for p in a.args]
    if a.vararg:
        names.append("*" + a.vararg.arg)
    names += [p.arg for p in a.kwonlyargs]
    if a.kwarg:
        names.append("**" + a.kwarg.arg)
    return names


def index_vllm_classes(vllm_root, skip):
    """{class_name: (file, {method: [argnames]})} across the vllm package."""
    idx = {}
    for root, _, files in os.walk(vllm_root):
        for f in files:
            if not f.endswith(".py"):
                continue
            fp = os.path.join(root, f)
            try:
                tree = ast.parse(open(fp).read())
            except Exception:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    if node.name in idx:
                        continue
                    methods = {}
                    for b in node.body:
                        if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            methods[b.name] = arg_names(b)
                    idx[node.name] = (fp, methods)
    return idx


def patch_base_name(classdef):
    """Return Base from `class X(ArcticPatch[Base]):`, else None."""
    for base in classdef.bases:
        # ArcticPatch[Base]  -> Subscript(value=Name('ArcticPatch'), slice=Name('Base'))
        if isinstance(base, ast.Subscript):
            val = base.value
            if isinstance(val, ast.Name) and val.id == "ArcticPatch":
                sl = base.slice
                if isinstance(sl, ast.Name):
                    return sl.id
                if isinstance(sl, ast.Attribute):
                    return sl.attr
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arctic", required=True)
    ap.add_argument("--vllm", required=True)
    ap.add_argument("--skip", default="dynasor,embedding")
    args = ap.parse_args()
    skip = {s for s in args.skip.split(",") if s}

    base_idx = index_vllm_classes(args.vllm, skip)

    mismatches, new_methods, missing_base = [], [], []
    for root, _, files in os.walk(args.arctic):
        if any(s in root.split(os.sep) for s in skip):
            continue
        for f in files:
            if not f.endswith(".py"):
                continue
            fp = os.path.join(root, f)
            rel = os.path.relpath(fp, args.arctic)
            try:
                tree = ast.parse(open(fp).read())
            except Exception:
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef):
                    continue
                base = patch_base_name(node)
                if not base:
                    continue
                if base not in base_idx:
                    missing_base.append((rel, node.name, base))
                    continue
                _, base_methods = base_idx[base]
                for b in node.body:
                    if not isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        continue
                    if b.name.startswith("__") and b.name.endswith("__"):
                        pass  # still compare dunders like __init__/__post_init__
                    if b.name not in base_methods:
                        new_methods.append((rel, base, b.name))
                        continue
                    got, want = arg_names(b), base_methods[b.name]
                    if got != want:
                        mismatches.append((rel, base, b.name, got, want))

    if mismatches:
        print("SIGNATURE MISMATCHES (override vs target base):")
        for rel, base, meth, got, want in mismatches:
            print(f"  {rel}: {base}.{meth}")
            print(f"      arctic: ({', '.join(got)})")
            print(f"      base  : ({', '.join(want)})")
    if missing_base:
        print("\nBASE CLASS NOT FOUND in target vLLM (moved/renamed/removed?):")
        for rel, patch, base in missing_base:
            print(f"  {rel}: {patch} -> ArcticPatch[{base}]")
    if new_methods:
        print("\nNEW METHODS added by patch (verify none are typo'd overrides):")
        for rel, base, meth in new_methods:
            print(f"  {rel}: {base}.{meth}")
    if not (mismatches or missing_base):
        print("No signature mismatches against the target base classes.")
    return 1 if (mismatches or missing_base) else 0


if __name__ == "__main__":
    raise SystemExit(main())
