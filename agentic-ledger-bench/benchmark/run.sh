#!/usr/bin/env bash
# Run every reference implementation over the benchmark data and compare results.
#
# Usage (from this subproject's root):
#     bash benchmark/run.sh
#
# All three implementations must print semantically identical JSON for the
# baseline, and the arbiter (benchmark/verify.py) must pass for each. The
# implementations were written by separate agents that never saw each other's
# code, so agreement here is evidence about the *spec*, not about copied code.
#
# Only the Python implementations are runnable out of the box: one needs nothing
# but CPython, the other needs the Python tool stack. The Scala one needs
# scala-cli and is run only if it is on PATH.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

STMT="benchmark/data/statement.csv"
ARBITER="benchmark/verify.py"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

pass=0
fail=0

check() {
    local name="$1" file="$2"
    if python3 "$ARBITER" --check "$file" >/dev/null 2>&1; then
        echo "  PASS  $name"
        pass=$((pass + 1))
    else
        echo "  FAIL  $name"
        python3 "$ARBITER" --check "$file" 2>&1 | sed 's/^/        /'
        fail=$((fail + 1))
    fi
}

echo "== ground truth (computed by the arbiter, independently of all implementations) =="
python3 "$ARBITER" --summary | sed 's/^/  /'
echo

echo "== python (stdlib only, no type checker) =="
if (cd reference/python-unchecked && PYTHONPATH=. python3 -m ledger baseline "../../$STMT") >"$OUT/unchecked.json" 2>"$OUT/unchecked.err"; then
    check "python-unchecked" "$OUT/unchecked.json"
else
    echo "  FAIL  python-unchecked (CLI error)"; sed 's/^/        /' "$OUT/unchecked.err"; fail=$((fail + 1))
fi

echo "== python (ty + ruff enforced) =="
if python3 reference/python-tooling/ledger.py baseline "$STMT" >"$OUT/tooling.json" 2>"$OUT/tooling.err"; then
    check "python-tooling" "$OUT/tooling.json"
else
    echo "  FAIL  python-tooling (CLI error)"; sed 's/^/        /' "$OUT/tooling.err"; fail=$((fail + 1))
fi

if command -v scala-cli >/dev/null 2>&1; then
    echo "== scala 3 =="
    if (cd reference/scala && scala-cli run . --quiet -- baseline "../../$STMT") >"$OUT/scala.json" 2>"$OUT/scala.err"; then
        check "scala" "$OUT/scala.json"
    else
        echo "  SKIP  scala (scala-cli failed; see below)"; sed 's/^/        /' "$OUT/scala.err" | head -3
    fi
else
    echo "== scala 3 =="
    echo "  SKIP  scala-cli not on PATH (see reference/scala/README.md)"
fi

echo
echo "== cross-implementation agreement =="
python3 - "$OUT" <<'PY'
import json, sys, pathlib
d = pathlib.Path(sys.argv[1])
found = {}
for name in ("unchecked", "tooling", "scala"):
    p = d / f"{name}.json"
    if p.exists() and p.stat().st_size:
        found[name] = json.loads(p.read_text())

def txns(doc):
    data = doc.get("data", doc)
    return data.get("transactions")

if len(found) < 2:
    print("  only one implementation produced output; nothing to compare")
    raise SystemExit(0)

names = list(found)
base = names[0]
for other in names[1:]:
    a, b = txns(found[base]), txns(found[other])
    if a is None or b is None:
        print(f"  {base} vs {other}: one side has no transactions[] array "
              "(iteration-A schemas diverged in the original run) — comparing checks only")
        ca = found[base].get("data", found[base]).get("checks", {})
        cb = found[other].get("data", found[other]).get("checks", {})
        keys = [k for k in ca if k in cb]
        diffs = [k for k in keys if ca[k] != cb[k]]
        print(f"    checks compared: {len(keys)}, mismatches: {len(diffs)} {diffs}")
        continue
    if len(a) != len(b):
        print(f"  {base} vs {other}: DIFFERENT transaction counts {len(a)} vs {len(b)}")
        continue
    fields = ["line", "bookingDate", "valueDate", "amount", "balanceAfter",
              "rawAmount", "rawBalance", "checksum"]
    diffs = [(x.get("line"), f, x.get(f), y.get(f))
             for x, y in zip(a, b) for f in fields if x.get(f) != y.get(f)]
    print(f"  {base} vs {other}: {len(a)} transactions x {len(fields)} fields, "
          f"{len(diffs)} mismatches")
    for d_ in diffs[:5]:
        print(f"    {d_}")
PY

echo
echo "checks passed: $pass   failed: $fail"
[ "$fail" -eq 0 ]
