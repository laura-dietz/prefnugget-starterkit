#!/usr/bin/env bash
#
# Create golden traces for the approval test by running the ORIGINAL code
# (trec25/judges/prefnugget/) against the kiddie dataset.
#
# Runs original code from trec25/judges/ to produce golden traces for
# the refactored prefnugget-starterkit approval tests.
#
# Old code locations:
#   trec25/judges/prefnugget/     -> prefnugget--* traces
#   trec25/judges/ground_rubric/  -> grounded--* traces
#   trec25/judges/rubric/         -> queryonly--* traces
#
# Prerequisites:
#   source .venv/bin/activate
#   export CACHE_DIR="./prefnugget.cache"   # populated prompt cache
#
# Usage:
#   bash create_golden_traces.sh
#
set -euo pipefail

GOLDEN_DIR="tests/golden-traces"
TOPICS="data/kiddie/topics/kiddie-topics.jsonl"
RESPONSES="data/kiddie/runs/repgen/"
OUTDIR="/tmp/golden-trace-out"
OLD_PREFNUGGET="../trec25/judges/prefnugget/workflow.yml"
OLD_GROUNDED="../trec25/judges/ground_rubric/workflow.yml"
OLD_QUERYONLY="../trec25/judges/prefnugget/rubric_workflow.yml"

if [[ -z "${CACHE_DIR:-}" ]]; then
    echo "ERROR: CACHE_DIR not set. Export CACHE_DIR pointing to populated prompt cache."
    echo "  export CACHE_DIR=./prefnugget.cache"
    exit 1
fi

for wf in "$OLD_PREFNUGGET" "$OLD_GROUNDED" "$OLD_QUERYONLY"; do
    if [[ ! -f "$wf" ]]; then
        echo "ERROR: Old workflow not found at $wf"
        echo "  Run from prefnugget-starterkit/ directory."
        exit 1
    fi
done

mkdir -p "$GOLDEN_DIR" "$OUTDIR"

# Create a ties_allowed copy of the old prefnugget workflow (override default pref_judge)
TIES_PREFNUGGET="/tmp/golden-trace-out/workflow-prefnugget-ties.yml"
sed 's/pref_judge: "must_decide"/pref_judge: "ties_allowed"/' "$OLD_PREFNUGGET" > "$TIES_PREFNUGGET"

# Create a must_decide copy of the old grounded workflow (override default pref_judge)
DECIDE_GROUNDED="/tmp/golden-trace-out/workflow-grounded-decide.yml"
sed 's/pref_judge: "ties_allowed"/pref_judge: "must_decide"/' "$OLD_GROUNDED" > "$DECIDE_GROUNDED"

FAILURES=0

run_variant() {
    local workflow="$1"
    local old_variant="$2"
    local new_name="$3"
    local trace_file="${GOLDEN_DIR}/${new_name}.trace.jsonl"

    echo "--- ${old_variant} -> ${new_name} ---"
    rm -f "$trace_file"

    if MINIMA_TRACE_FILE="$trace_file" auto-judge run \
        --workflow "$workflow" \
        --variant "$old_variant" \
        --rag-responses "$RESPONSES" \
        --rag-topics "$TOPICS" \
        --out-dir "$OUTDIR"; then

        if [[ -f "$trace_file" ]]; then
            num_calls=$(wc -l < "$trace_file")
            echo "  OK: ${num_calls} calls recorded"
        else
            echo "  WARNING: no trace file produced"
            FAILURES=$((FAILURES + 1))
        fi
    else
        echo "  FAILED"
        FAILURES=$((FAILURES + 1))
    fi
    echo ""
}

# ── must_decide variants (old default) ────────────────────────────────────────

# ── prefnugget: must_decide (old default) ─────────────────────────────────────

echo "=== prefnugget must_decide variants ==="
run_variant "$OLD_PREFNUGGET" "iter20bothties-few"                "prefnugget--best-decide"
# run_variant "$OLD_PREFNUGGET" "iter20bothties-few-docs"           "prefnugget--best-decide-docs"
run_variant "$OLD_PREFNUGGET" "iter20bothties-few-random-pairs"   "prefnugget--random-decide"
# run_variant "$OLD_PREFNUGGET" "iter20bothties-few-docs-random-pairs" "prefnugget--random-decide-docs"

# ── prefnugget: ties_allowed (old code, pref_judge overridden) ────────────────

echo "=== prefnugget ties_allowed variants ==="
run_variant "$TIES_PREFNUGGET" "iter20bothties-few"                "prefnugget--best"
# run_variant "$TIES_PREFNUGGET" "iter20bothties-few-docs"           "prefnugget--best-docs"
run_variant "$TIES_PREFNUGGET" "iter20bothties-few-random-pairs"   "prefnugget--random"
# run_variant "$TIES_PREFNUGGET" "iter20bothties-few-docs-random-pairs" "prefnugget--random-docs"

# ── grounded: ties_allowed (old default) ──────────────────────────────────────

echo "=== grounded ties_allowed variants ==="
run_variant "$OLD_GROUNDED" "ground-response"        "grounded--best"
# run_variant "$OLD_GROUNDED" "ground-docs"            "grounded--best-docs"
run_variant "$OLD_GROUNDED" "ground-random-response" "grounded--random"
# run_variant "$OLD_GROUNDED" "ground-random-docs"     "grounded--random-docs"

# ── grounded: must_decide (old code, pref_judge overridden) ───────────────────

echo "=== grounded must_decide variants ==="
run_variant "$DECIDE_GROUNDED" "ground-response"        "grounded--best-decide"
# run_variant "$DECIDE_GROUNDED" "ground-docs"            "grounded--best-decide-docs"
run_variant "$DECIDE_GROUNDED" "ground-random-response" "grounded--random-decide"
# run_variant "$DECIDE_GROUNDED" "ground-random-docs"     "grounded--random-decide-docs"

# ── queryonly (old rubric_workflow.yml in prefnugget dir) ─────────────────────

echo "=== queryonly variants ==="
run_variant "$OLD_QUERYONLY" "prefnugget-rubric-response" "queryonly--response"
# run_variant "$OLD_QUERYONLY" "prefnugget-rubric-docs"     "queryonly--docs"

# ── Summary ───────────────────────────────────────────────────────────────────

echo "=== Done ==="
echo "Golden traces in ${GOLDEN_DIR}/:"
ls -la "$GOLDEN_DIR"/*.trace.jsonl 2>/dev/null || echo "  (none)"

if [[ $FAILURES -gt 0 ]]; then
    echo "${FAILURES} variant(s) failed."
    exit 1
fi

echo ""
echo "Uncomment -docs variants in this script to generate those golden traces too."
