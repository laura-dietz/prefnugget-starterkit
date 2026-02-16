#!/usr/bin/env bash
#
# Approval test: run all judge variants against the kiddie dataset,
# record prompt cache traces, and compare against a golden baseline.
#
# Each trace line is {"key": "<hash>", "canonical": "<prompt-json>"}.
# We compare cache keys (hashes) first. On mismatch, we show a diff of
# the full canonical prompts so you can see what changed.
#
# Usage:
#   source .venv/bin/activate
#   export CACHE_DIR="./prefnugget.cache"   # must contain populated cache
#
#   bash run_approval_test.sh --approve      # first run: record golden baseline
#   bash run_approval_test.sh                # subsequent: compare against golden
#   bash run_approval_test.sh --approve best # re-approve a single variant
#
set -euo pipefail

GOLDEN_DIR="tests/golden-traces"
TRACE_DIR="output-kiddie/traces"
TOPICS="data/kiddie/topics/kiddie-topics.jsonl"
RESPONSES="data/kiddie/runs/repgen/"
OUTDIR="output-kiddie"

# ── Variant definitions ─────────────────────────────────────────────────────
# Format: "workflow_file:variant_name"

PREFNUGGET_VARIANTS=(
    "judges/prefnugget/workflow.yml:best"
    "judges/prefnugget/workflow.yml:best-docs"
    "judges/prefnugget/workflow.yml:random"
    "judges/prefnugget/workflow.yml:random-docs"
    "judges/prefnugget/workflow.yml:best-decide"
    "judges/prefnugget/workflow.yml:best-decide-docs"
    "judges/prefnugget/workflow.yml:random-decide"
    "judges/prefnugget/workflow.yml:random-decide-docs"
)

GROUNDED_VARIANTS=(
    "judges/grounded/workflow.yml:best"
    "judges/grounded/workflow.yml:best-docs"
    "judges/grounded/workflow.yml:random"
    "judges/grounded/workflow.yml:random-docs"
    "judges/grounded/workflow.yml:best-decide"
    "judges/grounded/workflow.yml:best-decide-docs"
    "judges/grounded/workflow.yml:random-decide"
    "judges/grounded/workflow.yml:random-decide-docs"
)

QUERYONLY_VARIANTS=(
    "judges/queryonly/workflow.yml:response"
    "judges/queryonly/workflow.yml:docs"
)

ALL_VARIANTS=( "${PREFNUGGET_VARIANTS[@]}" "${GROUNDED_VARIANTS[@]}" "${QUERYONLY_VARIANTS[@]}" )

# ── Helpers ──────────────────────────────────────────────────────────────────

# Derive a safe filename from "workflow:variant" -> "prefnugget--best" etc.
trace_name() {
    local spec="$1"
    local workflow="${spec%%:*}"
    local variant="${spec##*:}"
    # Extract judge dir name from workflow path
    local judge_dir
    judge_dir=$(basename "$(dirname "$workflow")")
    echo "${judge_dir}--${variant}"
}

# Extract sorted cache keys from a trace file
extract_hashes() {
    python3 -c "
import json, sys
hashes = []
for line in open(sys.argv[1]):
    obj = json.loads(line)
    hashes.append(obj['key'])
print('\n'.join(hashes))
" "$1"
}

# Compare two trace files: show call-level summary + field-level diff for mismatches
compare_traces() {
    python3 -c "
import json, sys, difflib

golden_path, current_path = sys.argv[1], sys.argv[2]

golden = [json.loads(l) for l in open(golden_path)]
current = [json.loads(l) for l in open(current_path)]

print(f'  Golden: {len(golden)} calls, Current: {len(current)} calls')

# Walk both in order, report first mismatches
max_show = 3
shown = 0
for i in range(max(len(golden), len(current))):
    if i >= len(golden):
        print(f'  Call {i+1}: EXTRA in current (not in golden)')
        shown += 1
    elif i >= len(current):
        print(f'  Call {i+1}: MISSING in current (was in golden)')
        shown += 1
    elif golden[i]['key'] == current[i]['key']:
        continue
    else:
        print(f'  Call {i+1}: HASH MISMATCH')
        print(f'    golden:  {golden[i][\"key\"]}')
        print(f'    current: {current[i][\"key\"]}')
        # Decode canonical and diff field by field
        g_canon = json.loads(golden[i]['canonical'])
        c_canon = json.loads(current[i]['canonical'])
        # Compare top-level keys
        for key in sorted(set(list(g_canon.keys()) + list(c_canon.keys()))):
            if key == 'messages':
                continue  # handle below
            g_val = g_canon.get(key)
            c_val = c_canon.get(key)
            if g_val != c_val:
                print(f'    {key}: {g_val!r} -> {c_val!r}')
        # Compare messages
        g_msgs = g_canon.get('messages', [])
        c_msgs = c_canon.get('messages', [])
        if len(g_msgs) != len(c_msgs):
            print(f'    messages: {len(g_msgs)} -> {len(c_msgs)} messages')
        for mi in range(max(len(g_msgs), len(c_msgs))):
            if mi >= len(g_msgs):
                print(f'    msg[{mi}]: EXTRA in current ({c_msgs[mi].get(\"role\",\"?\")})')
            elif mi >= len(c_msgs):
                print(f'    msg[{mi}]: MISSING in current (was {g_msgs[mi].get(\"role\",\"?\")})')
            elif g_msgs[mi] != c_msgs[mi]:
                g_content = g_msgs[mi].get('content', '')
                c_content = c_msgs[mi].get('content', '')
                role = g_msgs[mi].get('role', '?')
                if g_msgs[mi].get('role') != c_msgs[mi].get('role'):
                    print(f'    msg[{mi}]: role {g_msgs[mi].get(\"role\")} -> {c_msgs[mi].get(\"role\")}')
                if g_content != c_content:
                    # Show unified diff of message content
                    g_lines = g_content.splitlines(keepends=True)
                    c_lines = c_content.splitlines(keepends=True)
                    diff = list(difflib.unified_diff(g_lines, c_lines,
                        fromfile=f'golden msg[{mi}] ({role})',
                        tofile=f'current msg[{mi}] ({role})',
                        n=2))
                    if diff:
                        print(f'    msg[{mi}] ({role}): content differs:')
                        for dl in diff[:30]:
                            print(f'      {dl}', end='' if dl.endswith('\n') else '\n')
                        if len(diff) > 30:
                            print(f'      ... ({len(diff)-30} more diff lines)')
        print()
        shown += 1
    if shown >= max_show:
        remaining = 0
        for j in range(i+1, max(len(golden), len(current))):
            if j >= len(golden) or j >= len(current) or golden[j]['key'] != current[j]['key']:
                remaining += 1
        if remaining:
            print(f'  ... and {remaining} more mismatched call(s)')
        break

if shown == 0:
    print('  ERROR: hashes file differs but call-by-call comparison found no mismatches')
" "$1" "$2"
}

run_variant() {
    local spec="$1"
    local workflow="${spec%%:*}"
    local variant="${spec##*:}"
    local name
    name=$(trace_name "$spec")
    local trace_file="${TRACE_DIR}/${name}.trace.jsonl"

    echo "--- Running ${name} ---"

    # Clean trace file
    rm -f "$trace_file"

    # Run with trace recording
    MINIMA_TRACE_FILE="$trace_file" auto-judge run \
        --workflow "$workflow" \
        --variant "$variant" \
        --rag-responses "$RESPONSES" \
        --rag-topics "$TOPICS" \
        --out-dir "$OUTDIR" \

    if [[ ! -f "$trace_file" ]]; then
        echo "  WARNING: No trace file produced for ${name}"
        return 1
    fi

    local num_calls
    num_calls=$(wc -l < "$trace_file")
    echo "  Recorded ${num_calls} LLM calls"
}

compare_variant() {
    local spec="$1"
    local name
    name=$(trace_name "$spec")
    local trace_file="${TRACE_DIR}/${name}.trace.jsonl"
    local golden_hashes="${GOLDEN_DIR}/${name}.hashes"
    local golden_trace="${GOLDEN_DIR}/${name}.trace.jsonl"

    if [[ ! -f "$golden_trace" ]]; then
        echo "  SKIP ${name}: no golden trace"
        return 0
    fi

    # Always regenerate .hashes from .trace.jsonl (golden trace may have been updated)
    extract_hashes "$golden_trace" > "$golden_hashes"

    if [[ ! -f "$trace_file" ]]; then
        echo "  FAIL ${name}: no trace file produced by current run"
        return 1
    fi

    # Extract current hashes
    local current_hashes="${TRACE_DIR}/${name}.hashes"
    extract_hashes "$trace_file" > "$current_hashes"

    # Compare hash lists
    if diff -q "$golden_hashes" "$current_hashes" > /dev/null 2>&1; then
        echo "  PASS ${name} ($(wc -l < "$current_hashes") calls match)"
        return 0
    else
        echo "  FAIL ${name}: prompt hashes differ"
        echo ""
        compare_traces "$golden_trace" "$trace_file"
        return 1
    fi
}

approve_variant() {
    local spec="$1"
    local name
    name=$(trace_name "$spec")
    local trace_file="${TRACE_DIR}/${name}.trace.jsonl"
    local golden_hashes="${GOLDEN_DIR}/${name}.hashes"
    local golden_trace="${GOLDEN_DIR}/${name}.trace.jsonl"

    if [[ ! -f "$trace_file" ]]; then
        echo "  SKIP ${name}: no trace file to approve"
        return 0
    fi

    mkdir -p "$GOLDEN_DIR"
    extract_hashes "$trace_file" > "$golden_hashes"
    cp "$trace_file" "$golden_trace"

    local num_calls
    num_calls=$(wc -l < "$golden_hashes")
    echo "  APPROVED ${name} (${num_calls} calls)"
}

# Build reverse lookup: trace_name -> spec
declare -A NAME_TO_SPEC
for spec in "${ALL_VARIANTS[@]}"; do
    NAME_TO_SPEC[$(trace_name "$spec")]="$spec"
done

# ── Main ─────────────────────────────────────────────────────────────────────

MODE="compare"
FILTER=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --approve)
            MODE="approve"
            shift
            ;;
        *)
            FILTER="$1"
            shift
            ;;
    esac
done

# Ensure required env
if [[ -z "${CACHE_DIR:-}" ]]; then
    echo "ERROR: CACHE_DIR not set. Export CACHE_DIR pointing to populated prompt cache."
    echo "  export CACHE_DIR=./prefnugget.cache"
    exit 1
fi

mkdir -p "$TRACE_DIR" "$OUTDIR"

# ── Decide which variants to run ────────────────────────────────────────────

if [[ "$MODE" == "approve" ]]; then
    # --approve: run all variants (or filtered subset), then save as golden
    RUN_VARIANTS=("${ALL_VARIANTS[@]}")
    if [[ -n "$FILTER" ]]; then
        FILTERED=()
        for spec in "${ALL_VARIANTS[@]}"; do
            name=$(trace_name "$spec")
            if [[ "$name" == *"$FILTER"* ]]; then
                FILTERED+=("$spec")
            fi
        done
        if [[ ${#FILTERED[@]} -eq 0 ]]; then
            echo "No variants matching '$FILTER'. Available:"
            for spec in "${ALL_VARIANTS[@]}"; do
                echo "  $(trace_name "$spec")"
            done
            exit 1
        fi
        RUN_VARIANTS=("${FILTERED[@]}")
    fi
else
    # compare mode: only run variants that have a golden trace file
    RUN_VARIANTS=()
    MISSING=()

    for spec in "${ALL_VARIANTS[@]}"; do
        name=$(trace_name "$spec")
        golden_trace="${GOLDEN_DIR}/${name}.trace.jsonl"
        if [[ -f "$golden_trace" ]]; then
            RUN_VARIANTS+=("$spec")
        else
            MISSING+=("$name")
        fi
    done

    # Also pick up golden files not in ALL_VARIANTS (manually placed)
    if [[ -d "$GOLDEN_DIR" ]]; then
        for golden_file in "$GOLDEN_DIR"/*.trace.jsonl; do
            [[ -f "$golden_file" ]] || continue
            fname=$(basename "$golden_file" .trace.jsonl)
            if [[ -n "${NAME_TO_SPEC[$fname]+x}" ]]; then
                : # already in RUN_VARIANTS from the loop above
            else
                echo "  WARNING: golden file ${fname}.trace.jsonl has no matching variant definition"
            fi
        done
    fi

    if [[ ${#MISSING[@]} -gt 0 ]]; then
        echo "WARNING: No golden baseline for ${#MISSING[@]} variant(s):"
        for name in "${MISSING[@]}"; do
            echo "  - ${GOLDEN_DIR}/${name}.trace.jsonl"
        done
        echo ""
    fi

    if [[ ${#RUN_VARIANTS[@]} -eq 0 ]]; then
        echo "No golden baselines found in ${GOLDEN_DIR}/."
        echo "Run with --approve first to record baselines."
        exit 0
    fi
fi

echo "=== Approval Test: ${#RUN_VARIANTS[@]} variants, mode=${MODE} ==="
echo "    CACHE_DIR=${CACHE_DIR}"
echo ""

# ── Run ─────────────────────────────────────────────────────────────────────

for spec in "${RUN_VARIANTS[@]}"; do
    run_variant "$spec" || true
done

echo ""

# ── Compare or approve ──────────────────────────────────────────────────────

if [[ "$MODE" == "approve" ]]; then
    echo "=== Approving golden baselines ==="
    for spec in "${RUN_VARIANTS[@]}"; do
        approve_variant "$spec"
    done
    echo ""
    echo "Golden baselines saved to ${GOLDEN_DIR}/"
    echo "Commit these files to track prompt stability."
else
    echo "=== Comparing against golden baselines ==="
    FAILURES=0
    PASSES=0
    for spec in "${RUN_VARIANTS[@]}"; do
        if compare_variant "$spec"; then
            PASSES=$((PASSES + 1))
        else
            FAILURES=$((FAILURES + 1))
        fi
    done

    echo ""
    echo "Results: ${PASSES} passed, ${FAILURES} failed, ${#MISSING[@]} skipped (no golden)"
    if [[ $FAILURES -gt 0 ]]; then
        echo "Run with --approve to update golden baselines after reviewing changes."
        exit 1
    fi
fi
