---
name: autojudge-develop
description: Modify, run, and evaluate the PrefNugget judges. Use when the user wants to change prompts or settings, add a variant, run workflows, debug prompt-cache misses, or meta-evaluate against ground truth. Covers the develop, run, cache, and meta-evaluation activities for this repo's judges.
---

# Develop, run, and evaluate a PrefNugget judge

Walk the developer through this **interactively, one step at a time**. After each step, report what you found or did, then confirm before moving on.

The **canonical instructions live in the [TREC AutoJudge Participant HowTo](https://github.com/trec-auto-judge/.github/blob/main/profile/howto/README.md)** — this skill drives its pages [develop-an-autojudge](https://github.com/trec-auto-judge/.github/blob/main/profile/howto/03-develop-an-autojudge.md), [run-workflows](https://github.com/trec-auto-judge/.github/blob/main/profile/howto/04-run-workflows.md), [prompt-cache](https://github.com/trec-auto-judge/.github/blob/main/profile/howto/05-prompt-cache.md), and [meta-evaluation](https://github.com/trec-auto-judge/.github/blob/main/profile/howto/06-meta-evaluation.md); defer to them and do not contradict them. Environment or endpoint problems belong to `/autojudge-setup`; submitting belongs to `/autojudge-submit`.

## Step 1 — Orient in this repo's judge family
Three judges share a three-phase architecture (rank → extract nuggets → grade): `judges/prefnugget/` (contrastive extraction from winner/loser pairs), `judges/grounded/` (extraction from top-ranked responses), `judges/queryonly/` (query-only generation). Shared logic lives in `judges/shared/` (DSPy signatures, batching, grading); the README's variant tables map judges to `workflow.yml` + `--variant` names. Ask what the developer wants to change — a prompt, a setting, a new variant, or a new judge — and locate the right file before editing.

## Step 2 — Change settings and variants, not constants
Hyperparameters (nugget caps, pair sampling, grading mode) live in `workflow.yml` `settings`/`nugget_settings`/`judge_settings`; add a named block under `variants:` for a new configuration instead of editing code. Prompt changes go into the DSPy signatures in `judges/shared/` — keep explicit `reasoning` and `confidence` output fields, and mind that any prompt change invalidates the prompt cache for affected calls.

## Step 3 — Run on kiddie, iterate fast
```bash
auto-judge run --workflow judges/prefnugget/workflow.yml --variant best \
    --rag-responses data/kiddie/runs/repgen/ \
    --rag-topics data/kiddie/topics/kiddie-topics.jsonl \
    --out-dir ./output-kiddie/
```
During iteration prefer `--limit-topics 2` or `--topic ID`, and `-S/-N/-J KEY=VALUE` for quick overrides ([full flag table](https://github.com/trec-auto-judge/.github/blob/main/profile/howto/04-run-workflows.md)). Confirm the expected outputs appear (`<variant>.eval.txt`, `.nuggets.jsonl`, `.config.yml`); in `.nuggets.jsonl`, questions live under `nugget_bank` as a mapping keyed by nugget id, not a list. Keep `pytest` green as you go — the suite checks every git-tracked judge for minimum framework compatibility, and passing tests are a submission requirement.

## Step 4 — Keep the prompt cache warm
With `CACHE_DIR` set, re-runs should be near-instant. If they are not, diagnose per [prompt-cache](https://github.com/trec-auto-judge/.github/blob/main/profile/howto/05-prompt-cache.md): check prompt determinism first (responses sorted by `run_id` before pairing), then trace with `MINIMA_TRACE_FILE=trace.jsonl` and diff the canonical JSON between runs. Use `CACHE_FORCE_REFRESH=1` when fresh answers are wanted.

## Step 5 — Meta-evaluate
```bash
auto-judge-evaluate meta-evaluate \
    --truth-leaderboard data/kiddie/eval/kiddie_fake.eval.ir_measures.txt \
    --truth-format ir_measures --truth-header \
    --eval-format ir_measures --on-missing default \
    output-kiddie/*.eval.txt
```
Remind the developer that kiddie truth is synthetic — a pipeline check, not a quality signal; real correlations come from the [meta-evaluation service](https://github.com/trec-auto-judge/.github/blob/main/profile/howto/06-meta-evaluation.md) or real assessments. When ready to submit, hand off to `/autojudge-submit`.
