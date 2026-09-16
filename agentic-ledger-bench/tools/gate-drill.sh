#!/usr/bin/env bash
# Gate drill: a gate you have not seen fail is not a gate.
#
# Every gate gets two rows — it must be GREEN on the clean tree, and RED on a tree carrying the
# defect it exists to catch. A gate that never goes red proves nothing; a gate that is red on clean
# code is just as broken, and the clean rows catch that.
#
# The rows call the *configured* commands, not the raw tools, so a regression inside a gate script
# reddens the drill instead of hiding behind it. Plants are throwaway files under a scratch copy of
# the arm; nothing in the repository is modified, and the trap cleans up on any exit.
#
# Usage:  bash tools/gate-drill.sh [arm-dir]        (default: reference/python-tooling)
# Tools:  ruff, vulture, ty and radon from PATH, or via `uvx` if `uv` is installed.

set -uo pipefail
cd "$(dirname "$0")/.."
ARM="${1:-reference/python-tooling}"
[ -d "$ARM" ] || { echo "gate drill: no such arm: $ARM" >&2; exit 1; }

WORK="$(mktemp -d)"
cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT INT TERM

# --- tool resolution: PATH first, then uvx (keeps this runnable by a reader) --------------------
have() { command -v "$1" >/dev/null 2>&1; }
PYVER="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
resolve() {
    local tool="$1" pkg="${2:-$1}"
    if have "$tool"; then echo "$tool"; return 0; fi
    if have uvx; then echo "uvx --python $PYVER $pkg"; return 0; fi
    return 1
}
RUFF="$(resolve ruff || true)"
VULTURE="$(resolve vulture || true)"
COMPLEXITY="python3 tools/complexity.py"

# A tool that cannot parse the code must never be graded on it. vulture under an interpreter older
# than 3.12 prints one `invalid syntax` line for a PEP 695 `type X = ...` alias, skips the file, and
# reads as a clean scan — the silent-skip class this drill exists to catch. Here it is a SKIP.
syntax_skipped() { printf '%s' "$1" | grep -q "invalid syntax"; }

pass=0
missed=0

# grade <green|red> <label> <command...>
grade() {
    local want="$1" label="$2"; shift 2
    local got=green out
    if ! out="$("$@" 2>&1)"; then got=red; fi
    if [ "$got" = "$want" ]; then
        printf '  ok      %-5s %s\n' "$got" "$label"
        pass=$((pass + 1))
    else
        printf '  WRONG         %s — expected %s, got %s\n' "$label" "$want" "$got"
        printf '%s\n' "$out" | tail -4 | sed 's/^/                  | /'
        missed=$((missed + 1))
    fi
}

# expect_masked <label> <needle> <command...> — the command may be red for other reasons; the claim
# under test is that the needle never reaches its output.
expect_masked() {
    local label="$1" needle="$2"; shift 2
    local out
    out="$("$@" 2>&1 || true)"
    if printf '%s' "$out" | grep -q -- "$needle"; then
        printf '  NOT MASKED    %s — %s reached the output\n' "$label" "$needle"
        missed=$((missed + 1))
    else
        printf '  ok      masked %s\n' "$label"
        pass=$((pass + 1))
    fi
}

# scratch copy of the arm's importable package: plants stay out of the repository
PKG="$(basename "$ARM")"
cp -r "$ARM" "$WORK/$PKG"
cp "$ARM"/ruff.toml "$WORK/" 2>/dev/null || true

echo "gate drill: $ARM (clean tree green, planted defect red)"

if [ -n "$RUFF" ]; then
    echo "-- ruff lint"
    # config discovery: ruff.toml sits in $WORK, the planted package is $WORK/$PKG, so the gate runs
    # exactly as the arm's own `ruff check .` does from its root
    grade green "ruff (clean)" $RUFF check "$WORK/$PKG"
    cat >"$WORK/$PKG/_drill_plant.py" <<'PY'
import os  # unused import: the lint gate's defect


def value() -> int:
    return 1
PY
    grade red "ruff (planted unused import)" $RUFF check "$WORK/$PKG"
    rm -f "$WORK/$PKG/_drill_plant.py"
else
    echo "-- ruff lint: SKIPPED (no ruff and no uv/uvx)"
fi

echo "-- complexity budget"
grade green "complexity (clean)" $COMPLEXITY "$WORK/$PKG" --budget 12 --test-budget 15
{
    echo "def branchy(x: int) -> int:"
    echo "    total = 0"
    for i in $(seq 1 12); do echo "    if x == $i: total += 1"; done
    echo "    return total"
} >"$WORK/$PKG/_drill_plant.py"
grade red "complexity (planted 13-branch function)" $COMPLEXITY "$WORK/$PKG" --budget 12 --test-budget 15
rm -f "$WORK/$PKG/_drill_plant.py"

if [ -n "$VULTURE" ]; then
    echo "-- dead code"
    cat >"$WORK/$PKG/_drill_plant.py" <<'PY'
def documented_predicate(x: int) -> bool:
    """Written, documented, never called — the defect class a reachability gate exists for."""
    return x > 0
PY
    clean_out="$($VULTURE "$WORK/$PKG" 2>&1 || true)"
    if syntax_skipped "$clean_out"; then
        echo "  SKIP          dead code — $VULTURE cannot parse this tree (run the drill on"
        echo "                python3 >= 3.12, or install vulture on that interpreter)"
    else
        grade red "dead code (planted orphan)" $VULTURE "$WORK/$PKG"
        # the control that matters: the same orphan, under the threshold that silences it
        grade green "control: same orphan at --min-confidence 80" \
            $VULTURE "$WORK/$PKG" --min-confidence 80
    fi
    rm -f "$WORK/$PKG/_drill_plant.py"
else
    echo "-- dead code: SKIPPED (no vulture and no uv/uvx)"
fi

echo
if [ "$missed" -gt 0 ]; then
    echo "gate drill: $missed row(s) behaved wrongly — fix the gate before trusting it"
    exit 1
fi
echo "gate drill: $pass/$pass rows green or red as required"
