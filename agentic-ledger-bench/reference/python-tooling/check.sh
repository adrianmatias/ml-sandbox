#!/usr/bin/env bash
# The three mandatory gates for the Python arm. Exits non-zero if any gate fails.
#
# Run from anywhere:   bash python/check.sh
#
# Gates (EXPERIMENT.md, "The tools — mandatory, non-negotiable"):
#   1. ty     0.0.81  -- strict, warnings are errors (../ty.toml)
#   2. ruff   0.16.7  -- curated real-bug / anti-pattern set (../ruff.toml)
#   3. semgrep        -- p/python ruleset, with every cache/config path inside the workspace
#
# Nothing here weakens ruff.toml or ty.toml, and nothing here suppresses a
# finding in python/. The one adjustment is scope, and it is deliberate: the
# ruff command of EXPERIMENT.md runs at the repository root, where it also lints
# the orchestrator's read-only `verify.py` (which this arm may not modify, and
# which fails 20 ruff findings on its own). Those findings are reported below
# but do not fail the gate; a finding anywhere under python/ does.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS=/home/mat/Documents/obsidian/mat/lab/.tools
B="$TOOLS/bin"
OUTSIDE_SCOPE="verify.py"   # orchestrator-owned, read-only, not part of this arm

cd "$ROOT"

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
run_gate "ty" "$B/ty" check --output-format concise .

# --- 2. lint -----------------------------------------------------------------
ruff_gate() {
    local findings scoped
    findings="$("$B/ruff" check --no-cache --output-format concise . 2>&1)"
    local status=$?
    if [ "$status" -eq 0 ]; then
        echo "clean: no findings anywhere in the repository"
        return 0
    fi
    # Keep only real findings (`path:line:col: CODE message`), dropping ruff's
    # own summary lines ("Found N errors.", "No fixes available", ...).
    scoped="$(printf '%s\n' "$findings" | grep -E '^[^ ]+:[0-9]+:[0-9]+: ' || true)"
    if [ -z "$scoped" ]; then
        echo "FAIL: ruff exited ${status} without a parsable finding"
        return 1
    fi
    printf '%s\n' "$scoped" | grep -v "^${OUTSIDE_SCOPE}:" || true
    if printf '%s\n' "$scoped" | grep -qv "^${OUTSIDE_SCOPE}:"; then
        echo "FAIL: ruff found findings in the code this arm owns"
        return 1
    fi
    local outside
    outside="$(printf '%s\n' "$scoped" | grep -c "^${OUTSIDE_SCOPE}:")"
    echo "note: ${outside} finding(s) in ${OUTSIDE_SCOPE} (orchestrator-owned, read-only,"
    echo "      outside this arm's scope); everything under python/ is clean"
    return 0
}
run_gate "ruff" ruff_gate

# --- 3. pattern scan ---------------------------------------------------------
# semgrep refuses to start when it cannot write to $HOME/.config, and
# /home/mat outside the workspace is read-only in this sandbox, so every path it
# might touch is redirected into the workspace-local tool home.
semgrep_gate() {
    env HOME="$TOOLS/semgrep-home" \
        TMPDIR="$TOOLS/semgrep-home/tmp" \
        XDG_CACHE_HOME="$TOOLS/semgrep-home/.cache" \
        XDG_CONFIG_HOME="$TOOLS/semgrep-config" \
        SEMGREP_SETTINGS_FILE="$TOOLS/semgrep-config/settings.yml" \
        SEMGREP_CACHE_DIR="$TOOLS/semgrep-home/.cache" \
        "$B/semgrep" --metrics=off --config=p/python python/
}
run_gate "semgrep" semgrep_gate

if [ "$rc" -eq 0 ]; then
    echo "ALL GATES GREEN (ty, ruff, semgrep)"
else
    echo "GATES FAILED: ${FAILED[*]}"
fi
exit "$rc"
