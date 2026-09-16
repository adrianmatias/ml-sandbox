#!/usr/bin/env python3
"""Complexity gate: a per-function cyclomatic budget, enforced, stdlib-only.

"Track complexity" is not a gate until something fails. This is the failing part: it walks the
package, computes McCabe complexity per function/method, and exits non-zero when one exceeds the
budget. When a function is over budget, split it — do not raise the budget for one file. One-off
exceptions rot into permanent rot.

Usage:
    python3 tools/complexity.py reference/python-tooling            # gate the arm
    python3 tools/complexity.py --budget 12 --test-budget 15 <path>...
    python3 tools/complexity.py --json <path>...                    # machine-readable, for CI tables

Decision set is the one radon uses (if/elif, for, while, and/or as separate decisions, ternary,
except handler, comprehension `if`s, `match` cases beyond the first), so `--json` output is
comparable with `radon cc` on the same tree. A nested def is its own unit and its decisions stay out
of the parent's count: folding them in measures closure factories, not complexity.

Scope is explicit: tests are gated too, at a looser budget, because a 40-branch test helper is a
maintenance problem even when it is not production code.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

SKIP_DIRS = {"__pycache__", ".venv", "venv", ".git", ".scala-build", "target", "node_modules"}


def complexity(fn: ast.AST) -> int:
    """McCabe complexity of one function body; nested defs are counted separately."""
    nested = {
        n for n in ast.walk(fn)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) and n is not fn
    }
    total, stack = 1, list(fn.body)
    while stack:
        node = stack.pop()
        if node in nested or isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if isinstance(node, (ast.If, ast.IfExp, ast.While, ast.For, ast.AsyncFor, ast.ExceptHandler)):
            total += 1
        elif isinstance(node, ast.BoolOp):
            total += len(node.values) - 1
        elif isinstance(node, ast.comprehension):
            total += len(node.ifs)
        elif isinstance(node, ast.Match):
            total += max(0, len(node.cases) - 1)
        stack.extend(ast.iter_child_nodes(node))
    return total


def blocks_in(path: Path) -> list[tuple[int, int, str]]:
    """(complexity, lineno, name) for every function in the file."""
    tree = ast.parse(path.read_text())
    out = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append((complexity(node), node.lineno, node.name))
    return out


def scan(paths: list[Path], budget: int, test_budget: int) -> tuple[list[dict], list[str]]:
    worst: list[dict] = []
    failures: list[str] = []
    for target in paths:
        files = [target] if target.is_file() else sorted(
            p for p in target.rglob("*.py") if not (SKIP_DIRS & set(p.parts))
        )
        for path in files:
            try:
                blocks = blocks_in(path)
            except SyntaxError as exc:
                failures.append(f"{path}: unparsable ({exc})")
                continue
            limit = test_budget if "tests" in path.parts or path.name.startswith("test_") else budget
            for value, lineno, name in blocks:
                row = {"file": str(path), "line": lineno, "name": name, "complexity": value,
                       "budget": limit}
                worst.append(row)
                if value > limit:
                    failures.append(f"{path}:{lineno}:{name} complexity {value} > {limit}")
    worst.sort(key=lambda r: -r["complexity"])
    return worst, failures


def main() -> int:
    ap = argparse.ArgumentParser(description="per-function cyclomatic budget")
    ap.add_argument("paths", nargs="+", help="files or directories to gate")
    ap.add_argument("--budget", type=int, default=12, help="production budget (default 12)")
    ap.add_argument("--test-budget", type=int, default=15, help="tests budget (default 15)")
    ap.add_argument("--top", type=int, default=8, help="how many functions to list")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a report")
    args = ap.parse_args()

    worst, failures = scan([Path(p) for p in args.paths], args.budget, args.test_budget)
    if args.json:
        print(json.dumps({"functions": worst, "failures": failures}, indent=2))
        return 1 if failures else 0

    print("top functions:")
    for row in worst[: args.top]:
        print(f"  {row['complexity']:3d}  {row['file']}:{row['line']}:{row['name']}")
    if failures:
        print("\nBUDGET EXCEEDED:")
        for line in failures:
            print(f"  {line}")
        return 1
    print(f"\ncomplexity: OK ({len(worst)} functions, budget {args.budget}/{args.test_budget})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
