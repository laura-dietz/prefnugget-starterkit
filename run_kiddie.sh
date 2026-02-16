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
VARIANT="best"

# Other PrefNugget variants:
#   best-docs                (best, document-graded)
#   random                   (random pairs, response-graded)
#   random-docs              (random pairs, document-graded)
#   best-decide              (best, must_decide, response-graded)
#   best-decide-docs         (best, must_decide, document-graded)
#   random-decide            (random pairs, must_decide, response-graded)
#   random-decide-docs       (random pairs, must_decide, document-graded)
#
# GroundedNugget (use judges/grounded/workflow.yml):
#   best       best-docs
#   random     random-docs
#
# QueryOnlyNugget (use judges/queryonly/workflow.yml):
#   response   docs

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
