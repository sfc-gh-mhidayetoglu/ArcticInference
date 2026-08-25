#!/usr/bin/env python3
"""Static resolver: do all `vllm` imports in ArcticInference exist in a target vLLM tree?

Usage:
    python check_vllm_imports.py --arctic <arctic_inference_pkg> --vllm <vllm_pkg> \
        [--skip dynasor,embedding]

- <arctic_inference_pkg>: path to the `arctic_inference` package dir.
- <vllm_pkg>: path to the `vllm` package dir inside a checkout/worktree of the
  TARGET vLLM version (e.g. /tmp/vllm_v24/vllm).

Reports missing modules (for `import vllm.x.y`) and missing symbols
(for `from vllm.x import Y`). Symbol resolution is best-effort: it checks the
module file/package for a top-level def/class/assignment or a `*` re-export.
Manually confirm anything flagged — `*`-re-exports across submodules produce
false positives.
"""
import argparse
import ast
import os


def module_to_path(vllm_root, mod):
    # mod like "vllm.v1.attention.backend" -> vllm_root/v1/attention/backend
    rel = mod.split(".")[1:]
    return os.path.join(vllm_root, *rel)


def module_exists(vllm_root, mod):
    p = module_to_path(vllm_root, mod)
    return (
        os.path.isfile(p + ".py")
        or os.path.isdir(p)
        and os.path.isfile(os.path.join(p, "__init__.py"))
        or os.path.isdir(p)
    )


def module_source(vllm_root, mod):
    p = module_to_path(vllm_root, mod)
    if os.path.isfile(p + ".py"):
        return p + ".py"
    if os.path.isdir(p) and os.path.isfile(os.path.join(p, "__init__.py")):
        return os.path.join(p, "__init__.py")
    return None


def symbol_defined(src_file, name):
    """Heuristic: is `name` defined or star-imported at top level of src_file?"""
    try:
        tree = ast.parse(open(src_file).read())
    except Exception:
        return True  # can't parse -> don't cry wolf
    has_star = False
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if n.name == name:
                return True
        elif isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name) and t.id == name:
                    return True
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                if a.name == "*":
                    has_star = True
                elif (a.asname or a.name) == name:
                    return True
    return has_star  # star re-export -> assume present (manual confirm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arctic", required=True)
    ap.add_argument("--vllm", required=True)
    ap.add_argument("--skip", default="dynasor,embedding")
    args = ap.parse_args()
    skip = {s for s in args.skip.split(",") if s}

    missing_mod, missing_sym = set(), set()
    n_mod = n_sym = 0
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
            except Exception as e:
                print("PARSE FAIL", rel, e)
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for a in node.names:
                        if a.name == "vllm" or a.name.startswith("vllm."):
                            n_mod += 1
                            if not module_exists(args.vllm, a.name):
                                missing_mod.add((rel, a.name))
                elif isinstance(node, ast.ImportFrom):
                    if node.level or not node.module:
                        continue
                    if node.module != "vllm" and not node.module.startswith("vllm."):
                        continue
                    if not module_exists(args.vllm, node.module):
                        missing_mod.add((rel, node.module))
                        continue
                    src = module_source(args.vllm, node.module)
                    if src is None:
                        continue
                    for a in node.names:
                        if a.name == "*":
                            continue
                        n_sym += 1
                        if not symbol_defined(src, a.name):
                            missing_sym.add((rel, node.module, a.name))

    print(f"checked {n_mod} module imports, {n_sym} symbol imports "
          f"(skipped: {', '.join(sorted(skip)) or 'none'})")
    if missing_mod:
        print("\nMISSING MODULES:")
        for rel, m in sorted(missing_mod):
            print(f"  {rel}: {m}")
    if missing_sym:
        print("\nMISSING SYMBOLS (confirm manually — `*` re-exports cause false positives):")
        for rel, m, s in sorted(missing_sym):
            print(f"  {rel}: from {m} import {s}")
    if not missing_mod and not missing_sym:
        print("\nAll vllm imports resolve in the target tree.")
    return 1 if (missing_mod or missing_sym) else 0


if __name__ == "__main__":
    raise SystemExit(main())
