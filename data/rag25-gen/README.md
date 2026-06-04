---
configs:
- config_name: inputs
  data_files:
  - split: test
    path: ["runs/*/*.jsonl", "topics/*.jsonl"]
- config_name: truths
  data_files:
  - split: test
    path: ["eval/rag--anon-scores.gen.nist-post-edit.txt", "eval/rag--anon-scores.gen.nuggetizer_eval.txt"]

tira_configs:
  resolve_inputs_to: "."
  resolve_truths_to: "."
  baseline:
    link: https://github.com/trec-auto-judge/auto-judge-starter-kit/tree/main/
    command: auto-judge run --workflow /auto-judge/judges/naive/workflow.yml --rag-responses $inputDataset/runs/*/ --rag-topics $inputDataset/topics/*.jsonl --out-dir $outputDir
    file: judges/naive/Dockerfile
    format:
      name: ["trec-eval-leaderboard"]
  input_format:
    name: "trec-rag-runs"
  truth_format:
    name: "arbitrary"
  evaluator:
    image: ghcr.io/trec-auto-judge/auto-judge-code/cli:0.0.3
    command: /evaluator.py $inputDataset/leaderboard.eval.txt ${inputRun} ${outputDir}
---

# TBD

# Admin Section

Submit to TIRA via:

```
tira-cli dataset-submission --path rag25-gen --task trec-auto-judge --split test --dry-run
```
