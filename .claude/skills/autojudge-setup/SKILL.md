---
name: autojudge-setup
description: Set up a development environment for the PrefNugget judges. Use when the user wants to get this repo running, install dependencies, or asks "how do I start / set up / run these judges". Walks through venv, install, a verification run on the kiddie dataset, and picking a judge variant.
---

# Set up the PrefNugget development environment

Walk the developer through this **interactively, one step at a time**. After each step, report what you found or did, then confirm before moving on.

The **canonical instructions live in the [TREC AutoJudge Participant HowTo](https://github.com/trec-auto-judge/.github/blob/main/profile/howto/README.md)** — this skill drives its pages [setup-environment](https://github.com/trec-auto-judge/.github/blob/main/profile/howto/01-setup-environment.md), [configure-llm-endpoint](https://github.com/trec-auto-judge/.github/blob/main/profile/howto/02-configure-llm-endpoint.md), and [run-workflows](https://github.com/trec-auto-judge/.github/blob/main/profile/howto/04-run-workflows.md); defer to them and do not contradict them. This repo's README holds the judge-specific reference (variants, pseudocode, prompts). For submitting, use `/autojudge-submit`.

## Step 1 — Create and activate a virtual environment
```bash
uv venv
source .venv/bin/activate
```
Common pitfall: `uv venv` creates the venv but does **not** activate it — if activation is skipped, the next `uv pip install` may land in the wrong environment.

## Step 2 — Install
```bash
uv pip install -e '.[all]'
```
The `.[all]` extra covers develop + test + evaluate + submit. The lightweight `uv pip install -e .` (judge only, no tira/pytest) works for a first look; switch to `.[all]` before testing or submitting.

## Step 3 — Configure the LLM endpoint
These judges call an LLM, so set the endpoint per [configure-llm-endpoint](https://github.com/trec-auto-judge/.github/blob/main/profile/howto/02-configure-llm-endpoint.md):
```bash
export OPENAI_BASE_URL=...  OPENAI_MODEL=...  OPENAI_API_KEY=...
export CACHE_DIR="./cache"   # optional, enables prompt caching
```

## Step 4 — Verify on the kiddie dataset
```bash
bash run_kiddie.sh
pytest
```
Any failure here signals an environment problem to fix before going further.

## Step 5 — Pick a judge and variant
The README's "Workflow files and variant names" table maps each judge (PrefNugget, GroundedNugget, QueryOnlyNugget) to its `workflow.yml` and `--variant` names. Run one with:
```bash
auto-judge run --workflow judges/prefnugget/workflow.yml --variant best \
    --rag-responses data/kiddie/runs/repgen/ \
    --rag-topics data/kiddie/topics/kiddie-topics.jsonl \
    --out-dir ./output-kiddie/
```
For dev flags, variants, and output files see [run-workflows](https://github.com/trec-auto-judge/.github/blob/main/profile/howto/04-run-workflows.md); when ready to submit, use `/autojudge-submit`.
