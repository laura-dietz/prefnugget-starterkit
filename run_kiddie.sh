#!/usr/bin/env bash
#
# End-to-end smoke test: run one judge variant on the kiddie dataset,
# then meta-evaluate against the (fake) kiddie truth leaderboard.
#
# Usage:
#   source .venv/bin/activate
#   bash run_kiddie.sh
#
set -euo pipefail

OUTDIR="./output-kiddie"
TOPICS="data/kiddie/topics/kiddie-topics.jsonl"
RESPONSES="data/kiddie/runs/repgen/"
TRUTH="data/kiddie/eval/kiddie_fake.eval.ir_measures.txt"

# --- Run the PrefNugget judge (best, response-graded) ---
WORKFLOW="judges/prefnugget/workflow.yml"
VARIANT="iter20bothties-few"

# Other PrefNugget variants:
#   iter20bothties-few-docs              (best, document-graded)
#   iter20bothties-few-random-pairs      (random pairs, response-graded)
#   iter20bothties-few-docs-random-pairs (random pairs, document-graded)
#
# GroundedNugget (use judges/grounded/workflow.yml):
#   ground-response        ground-docs
#   ground-random-response ground-random-docs
#
# QueryOnlyNugget (use judges/queryonly/workflow.yml):
#   prefnugget-rubric-response  prefnugget-rubric-docs

echo "=== Running ${VARIANT} on kiddie ==="
auto-judge run \
    --workflow "${WORKFLOW}" \
    --variant "${VARIANT}" \
    --rag-responses "${RESPONSES}" \
    --rag-topics "${TOPICS}" \
    --out-dir "${OUTDIR}"

echo ""
echo "=== Output files ==="
ls -1 "${OUTDIR}/"

# --- Local meta-evaluation ---
EVAL_FILE="${OUTDIR}/${VARIANT}.eval.txt"

if command -v auto-judge-evaluate &>/dev/null; then
    echo ""
    echo "=== Meta-evaluation (correlation with kiddie truth) ==="
    auto-judge-evaluate meta-evaluate \
        --truth-leaderboard "${TRUTH}" \
        --truth-format ir_measures --truth-header \
        --eval-format tot \
        --on-missing default \
        "${EVAL_FILE}"
else
    echo ""
    echo "Skipping meta-evaluation (auto-judge-evaluate not installed)."
    echo "Install with: uv pip install -e '.[evaluate]'"
fi

echo ""
echo "Done. Upload ${EVAL_FILE} to Auto-Judge Evaluation service for meta-evaluation on real datasets."