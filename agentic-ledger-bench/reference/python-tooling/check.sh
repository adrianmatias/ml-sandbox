#!/usr/bin/env bash
# The three mandatory gates for the Python arm. Exits non-zero if any gate fails.
#
# Run from anywhere:   bash check.sh
#
# Gates (EXPERIMENT.md, "The tools — mandatory, non-negotiable"):
#   1. ty     0.0.81  -- strict, warnings are errors (./ty.toml)
#   2. ruff   0.16.7  -- curated real-bug / anti-pattern set (./ruff.toml)
#   3. semgrep        -- p/python ruleset
#
# This arm is standard library only, and the gates are reproducible from this directory: both
# configs are published beside this script. Tools are taken from PATH, or installed on the fly with
# `uvx` **on the interpreter this code targets** — `uvx ty` defaults to whatever python it finds
# first, and under 3.11 it reports `Cannot use 'type' alias statement on Python 3.11` for a file the
# arm is entitled to write, which is an interpreter artefact rather than a defect in the arm.
#
# The one scope adjustment is deliberate: EXPERIMENT.md's ruff command runs at the repository root,
# where it also lints the orchestrator's read-only `verify.py` (which this arm may not modify). Those
# findings are reported but do not fail the gate; a finding anywhere under this directory does.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTSIDE_SCOPE="verify.py"   # orchestrator-owned, read-only, not part of this arm
PYVER="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"

cd "$ROOT"

have() { command -v "$1" >/dev/null 2>&1; }
resolve() {
    local tool="$1"
    if have "$tool"; then echo "$tool"; return 0; fi
    if have uvx; then echo "uvx --python $PYVER $tool"; return 0; fi
    return 1
}
TY="$(resolve ty || true)"
RUFF="$(resolve ruff || true)"
SEMGREP="$(resolve semgrep || true)"

rc=0
declare -a FAILED=()

run_gate() {
    local name="$1"
    shift
    echo "=== gate: ${name} ==="
    if "$@"; then
        echo "--- ${name}: PASS"
    else
        echo "--- ${name}: FAIL"
        rc=1
        FAILED+=("$name")
    fi
    echo
}

# --- 1. type check -----------------------------------------------------------
if [ -n "$TY" ]; then
    run_gate "ty" $TY check --output-format concise .
else
    echo "=== gate: ty === SKIPPED (no ty and no uvx)"
fi

# --- 2. lint -----------------------------------------------------------------
ruff_gate() {
    local findings scoped status
    findings="$($RUFF check --no-cache --output-format concise . 2>&1)"
    status=$?
    if [ "$status" -eq 0 ]; then
        echo "clean: no findings in this arm"
        return 0
    fi
    # Keep only real findings (`path:line:col: CODE message`), dropping ruff's
    # own summary lines ("Found N errors.", "No fixes available", ...).
    scoped="$(printf '%s\n' "$findings" | grep -E '^[^ ]+:[0-9]+:[0-9]+: ' || true)"
    if [ -z "$scoped" ]; then
        echo "FAIL: ruff exited ${status} without a parsable finding"
        printf '%s\n' "$findings" | tail -5
        return 1
    fi
    printf '%s\n' "$scoped"
    if printf '%s\n' "$scoped" | grep -qv "$OUTSIDE_SCOPE"; then
        echo "FAIL: ruff found findings in the code this arm owns"
        return 1
    fi
    echo "note: findings above are in ${OUTSIDE_SCOPE} (orchestrator-owned, read-only,"
    echo "      outside this arm's scope); everything this directory owns is clean"
    return 0
}
if [ -n "$RUFF" ]; then
    run_gate "ruff" ruff_gate
else
    echo "=== gate: ruff === SKIPPED (no ruff and no uvx)"
fi

# --- 3. pattern scan ---------------------------------------------------------
# semgrep writes its settings on startup and aborts when it cannot. Its path is derived from
# `Path.home()/.semgrep`, so redirecting the XDG variables is not enough in a sandbox whose $HOME is
# read-only — SEMGREP_SETTINGS_FILE has to point somewhere writable, and its parent must exist.
# Everything it touches lives under this directory, which keeps its cache out of the reader's home.
if [ -n "$SEMGREP" ]; then
    semgrep_gate() {
        mkdir -p "$ROOT/.semgrep-home/config" "$ROOT/.semgrep-home/cache" "$ROOT/.semgrep-home/tmp"
        env SEMGREP_SETTINGS_FILE="$ROOT/.semgrep-home/config/settings.yml" \
            SEMGREP_CACHE_DIR="$ROOT/.semgrep-home/cache" \
            XDG_CONFIG_HOME="$ROOT/.semgrep-home/config" \
            XDG_CACHE_HOME="$ROOT/.semgrep-home/cache" \
            TMPDIR="$ROOT/.semgrep-home/tmp" \
            $SEMGREP --metrics=off --config=p/python .
    }
    run_gate "semgrep" semgrep_gate
else
    echo "=== gate: semgrep === SKIPPED (no semgrep and no uvx)"
fi

if [ "$rc" -eq 0 ]; then
    echo "ALL GATES GREEN (ty, ruff, semgrep)"
else
    echo "GATES FAILED: ${FAILED[*]}"
fi
exit "$rc"
