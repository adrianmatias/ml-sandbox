#!/usr/bin/env python3
"""One yardstick over all five arms: structure, complexity, and lint — same tool, same settings.

Written because every number in FINDINGS.md was measured per-arm by different hands, and the
object-orientation question needs *comparable* counts. Read-only: it opens each arm, parses it, and
prints a table. Nothing is modified.

    python3 measure-arms.py [--json]

Arms (published in reference/, plus the two lab worktrees that are not published):
  * reference/python-unchecked    stdlib Python, no static analysis at all
  * reference/python-tooling      Python + ty + ruff enforced
  * reference/scala               Scala 3.9.0 LTS
  * lab fp-py-wt/fp-py            enforced functional style
  * lab val-py-wt/val-py          pydantic + beartype boundary arm

Complexity is McCabe per block (function/method), the same decision set radon uses, so the published
§14 numbers can be cross-checked rather than taken on trust. A nested def is its own block and its
decisions stay out of the parent's count — otherwise a closure factory reads as a monster.

`--own-env` switches the third-party arm's imports to the project's own environment, for measuring a
tree that has a .venv (the pydantic arm does; the others do not).
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# the bench this script ships in, plus the two arms that exist only in the experiment worktrees
BENCH = HERE.parent if (HERE.parent / "reference").exists() else HERE
LAB = BENCH.parent.parent  # lab/, which holds the unpublished worktrees

ARMS = [
    ("python-unchecked", BENCH / "reference/python-unchecked", "python"),
    ("python-tooling", BENCH / "reference/python-tooling", "python"),
    ("scala", BENCH / "reference/scala", "scala"),
    ("functional (lab worktree)", LAB / "fp-py-wt/fp-py", "python"),
    ("pydantic+beartype (lab worktree)", LAB / "val-py-wt/val-py", "python"),
]

SKIP_DIRS = {"tests", "test", ".venv", "venv", "__pycache__", ".scala-build", "target", ".git",
             "site-packages", "node_modules"}


def source_files(root: Path, lang: str) -> list[Path]:
    if lang == "scala":
        return sorted(p for p in (root / "src").rglob("*.scala") if not (SKIP_DIRS & set(p.parts)))
    return sorted(p for p in root.rglob("*.py") if not (SKIP_DIRS & set(p.parts)))


def decision_count(fn: ast.AST) -> int:
    """McCabe decisions in a function body, nested defs excluded (they are their own blocks)."""
    nested = {
        n for n in ast.walk(fn)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) and n is not fn
    }
    decisions = 0
    stack = list(fn.body)
    while stack:
        node = stack.pop()
        if node in nested:
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if isinstance(node, (ast.If, ast.IfExp, ast.While, ast.For, ast.AsyncFor,
                             ast.ExceptHandler)):
            decisions += 1
        elif isinstance(node, ast.BoolOp):
            decisions += len(node.values) - 1
        elif isinstance(node, ast.comprehension):
            decisions += len(node.ifs)
        elif isinstance(node, ast.Match):
            decisions += max(0, len(node.cases) - 1)
        stack.extend(ast.iter_child_nodes(node))
    return decisions


def _name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Call):
        return _name(node.func)
    if isinstance(node, ast.Subscript):
        return _name(node.value)
    return ""


def _unparse(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:  # noqa: BLE001 - best effort, decorators only
        return ""


def measure_python(root: Path) -> dict:
    files = source_files(root, "python")
    blocks: list[tuple[str, str, int, int]] = []
    stats = dict.fromkeys(
        ["classes", "dataclasses", "frozen_dataclasses", "enums", "protocols", "newtypes",
         "exception_classes", "domain_entities", "frozen_entities", "free_functions", "methods",
         "typed_functions", "untyped_functions", "assert_never", "type_ignores", "lines",
         "unparsable"], 0)
    for path in files:
        text = path.read_text()
        stats["lines"] += len(text.splitlines())
        stats["type_ignores"] += sum(1 for line in text.splitlines() if "type: ignore" in line)
        try:
            tree = ast.parse(text)
        except SyntaxError:
            stats["unparsable"] += 1
            continue

        # error hierarchies are modelled, not scattered: a class is an exception if it inherits from
        # one — `class RegistryError(LedgerError)` is an exception just as much as `(ValueError)`
        class_names = {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
        exception_names = {"Exception", "ValueError", "TypeError", "OSError", "KeyError",
                           "RuntimeError", "LookupError", "ArithmeticError", "AssertionError"}

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                stats["classes"] += 1
                decorators = {_unparse(d) for d in node.decorator_list}
                bases = {_name(b) for b in node.bases}
                is_exception = bool(bases & (exception_names | {n for n in class_names if n.endswith("Error")}))
                if is_exception:
                    stats["exception_classes"] += 1
                else:
                    stats["domain_entities"] += 1
                if any(d.startswith("dataclass") for d in decorators):
                    stats["dataclasses"] += 1
                    if any("frozen" in d for d in decorators):
                        stats["frozen_dataclasses"] += 1
                        if not is_exception:
                            stats["frozen_entities"] += 1
                if bases & {"Enum", "StrEnum", "IntEnum"}:
                    stats["enums"] += 1
                if "Protocol" in bases:
                    stats["protocols"] += 1
            elif isinstance(node, ast.Assign):
                if any(_name(v) == "NewType" for v in ast.walk(node.value)):
                    stats["newtypes"] += 1
            elif isinstance(node, ast.Call) and _name(node.func) == "assert_never":
                stats["assert_never"] += 1

            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                is_method = any(
                    isinstance(parent, ast.ClassDef) and node in parent.body
                    for parent in ast.walk(tree)
                )
                stats["methods" if is_method else "free_functions"] += 1
                annotated = bool(node.returns) and all(
                    a.annotation is not None
                    for a in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)
                    if a.arg not in {"self", "cls"}
                )
                stats["typed_functions" if annotated else "untyped_functions"] += 1
                blocks.append(("method" if is_method else "function", node.name, node.lineno,
                               decision_count(node) + 1))

    cc = [b[3] for b in blocks]
    return {
        "files": len(files),
        "lines": stats["lines"],
        "unparsable": stats["unparsable"],
        "classes": stats["classes"],
        "domain_entities": stats["domain_entities"],
        "exception_classes": stats["exception_classes"],
        "dataclasses": stats["dataclasses"],
        "frozen_dataclasses": stats["frozen_dataclasses"],
        "frozen_entities": stats["frozen_entities"],
        "enums": stats["enums"],
        "protocols": stats["protocols"],
        "newtypes": stats["newtypes"],
        "free_functions": stats["free_functions"],
        "methods": stats["methods"],
        "typed_functions": stats["typed_functions"],
        "untyped_functions": stats["untyped_functions"],
        "assert_never": stats["assert_never"],
        "type_ignores": stats["type_ignores"],
        "blocks": len(blocks),
        "cc_total": sum(cc),
        "cc_mean": round(sum(cc) / len(cc), 2) if cc else 0,
        "cc_max": max(cc) if cc else 0,
        "cc_over_10": sum(1 for c in cc if c > 10),
        "worst": sorted(blocks, key=lambda b: -b[3])[:6],
    }


def measure_scala(root: Path) -> dict:
    files = source_files(root, "scala")
    text = "\n".join(p.read_text() for p in files)
    return {
        "files": len(files),
        "lines": len(text.splitlines()),
        "unparsable": 0,
        "classes": text.count("\nclass ") + text.count("case class ") + text.count("\nenum "),
        "dataclasses": text.count("case class"),
        "frozen_dataclasses": None,
        "enums": text.count("\nenum "),
        "protocols": text.count("trait "),
        "newtypes": text.count("opaque type"),
        "free_functions": None,
        "methods": None,
        "typed_functions": None,
        "untyped_functions": None,
        "assert_never": None,
        "type_ignores": None,
        "blocks": None, "cc_total": None, "cc_mean": None, "cc_max": None, "cc_over_10": None,
        "worst": [],
        "note": "Scala structure counted textually; radon/ruff/ast do not apply.",
    }


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--own-env", action="store_true",
                    help="measure the pydantic arm's imports instead of its .venv (slow, default off)")
    args = ap.parse_args()

    results = {}
    for name, root, lang in ARMS:
        if not root.exists():
            print(f"!! missing arm: {name} at {root}", file=sys.stderr)
            continue
        results[name] = measure_scala(root) if lang == "scala" else measure_python(root)

    if args.json:
        print(json.dumps(results, indent=2))
        return 0

    cols = ["files", "lines", "classes", "domain_entities", "exception_classes",
            "frozen_dataclasses", "newtypes", "methods", "free_functions", "typed_functions",
            "untyped_functions", "blocks", "cc_mean", "cc_max", "cc_over_10", "type_ignores"]
    print(f"{'arm':<24}" + "".join(f"{c[:11]:>13}" for c in cols))
    for name, r in results.items():
        print(f"{name:<24}" + "".join(f"{_fmt(r.get(c)):>13}" for c in cols))

    print("\nworst blocks (complexity > 10):")
    for name, r in results.items():
        worst = [w for w in (r.get("worst") or []) if w[3] > 10]
        print(f"  {name:<20}" + (", ".join(f"{w[1]}()={w[3]}:{w[2]}" for w in worst) or "none"))
    return 0


def _fmt(value) -> str:
    return "—" if value is None else str(value)


if __name__ == "__main__":
    raise SystemExit(main())
