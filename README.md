# PrefNugget AutoJudge

Nugget-based LLM judge implementations for evaluating RAG systems, as described in our paper ["Too Many Questions?"](https://anonymous.4open.science/r/too-many-questions/).

Built on the [TREC AutoJudge](https://trec-auto-judge.cs.unh.edu/) framework.

## Quick Start

```bash
# Install
uv venv && source .venv/bin/activate
uv pip install -e .

# Run a judge variant against a dataset
auto-judge run \
    --workflow judges/prefnugget/workflow.yml \
    --variant iter20bothties-few \
    --rag-responses /path/to/responses/ \
    --rag-topics /path/to/topics.jsonl \
    --out-dir ./output/

# Run against the included kiddie test dataset
auto-judge run \
    --workflow judges/prefnugget/workflow.yml \
    --variant iter20bothties-few \
    --rag-responses data/kiddie/runs/repgen/ \
    --rag-topics data/kiddie/topics/kiddie-topics.jsonl \
    --out-dir ./output/
```

### LLM Configuration

Set environment variables:
```bash
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_MODEL="gpt-4o-mini"
export OPENAI_API_KEY="sk-..."
export CACHE_DIR="./cache"  # optional, enables prompt caching
```

Or use a config file: `auto-judge run --llm-config llm-config.yml ...`

## Judge Variants

All judges share a three-phase architecture: (1) rank responses, (2) extract nugget questions, (3) grade responses against nuggets. Variants differ in how nuggets are extracted.

| Judge | graded | nugget | preference | Phase 1: Ranking | Phase 2: Nugget Extraction | Phase 3: Grading |
| --- | --- | --- | --- | --- | --- | --- |
| PrefNugget best | response | pairwise | best | [PrefJudgment](#phase-1-preference-judging-prefjudgment) | [IterativeExtractDifferentiatingNuggets](#phase-2a-contrastive-nugget-extraction-iterativeextractdifferentiatingnuggets) | [GradeNuggetAnswer](#phase-3-nugget-based-grading-gradenuggetanswer) |
| PrefNugget docs | docs | pairwise | best | [PrefJudgment](#phase-1-preference-judging-prefjudgment) | [IterativeExtractDifferentiatingNuggets](#phase-2a-contrastive-nugget-extraction-iterativeextractdifferentiatingnuggets) | [GradeNuggetAnswer](#phase-3-nugget-based-grading-gradenuggetanswer) |
| PrefNugget random | response | pairwise | random | -- | [IterativeExtractDifferentiatingNuggets](#phase-2a-contrastive-nugget-extraction-iterativeextractdifferentiatingnuggets) | [GradeNuggetAnswer](#phase-3-nugget-based-grading-gradenuggetanswer) |
| PrefNugget random docs | docs | pairwise | random | -- | [IterativeExtractDifferentiatingNuggets](#phase-2a-contrastive-nugget-extraction-iterativeextractdifferentiatingnuggets) | [GradeNuggetAnswer](#phase-3-nugget-based-grading-gradenuggetanswer) |
| GroundedNugget best | response | grounded | best | [PrefJudgment](#phase-1-preference-judging-prefjudgment) | [GroundedIterativeNuggets](#phase-2b-grounded-nugget-extraction-groundediterativenuggets) | [GradeNuggetAnswer](#phase-3-nugget-based-grading-gradenuggetanswer) |
| GroundedNugget best docs | docs | grounded | best | [PrefJudgment](#phase-1-preference-judging-prefjudgment) | [GroundedIterativeNuggets](#phase-2b-grounded-nugget-extraction-groundediterativenuggets) | [GradeNuggetAnswer](#phase-3-nugget-based-grading-gradenuggetanswer) |
| GroundedNugget random | response | grounded | random | -- | [GroundedIterativeNuggets](#phase-2b-grounded-nugget-extraction-groundediterativenuggets) | [GradeNuggetAnswer](#phase-3-nugget-based-grading-gradenuggetanswer) |
| GroundedNugget random docs | docs | grounded | random | -- | [GroundedIterativeNuggets](#phase-2b-grounded-nugget-extraction-groundediterativenuggets) | [GradeNuggetAnswer](#phase-3-nugget-based-grading-gradenuggetanswer) |
| QueryOnlyNugget | response | query | na | -- | [IterativeGenerateNuggetQuestionsReportRequest](#phase-2c-query-only-nugget-generation-iterativegeneratenuggetquestionsreportrequest) | [GradeNuggetAnswer](#phase-3-nugget-based-grading-gradenuggetanswer) |
| QueryOnlyNugget docs | docs | query | na | -- | [IterativeGenerateNuggetQuestionsReportRequest](#phase-2c-query-only-nugget-generation-iterativegeneratenuggetquestionsreportrequest) | [GradeNuggetAnswer](#phase-3-nugget-based-grading-gradenuggetanswer) |

### Workflow files and variant names

| Judge | Workflow | Variant |
| --- | --- | --- |
| PrefNugget best | `judges/prefnugget/workflow.yml` | `iter20bothties-few` |
| PrefNugget docs | `judges/prefnugget/workflow.yml` | `iter20bothties-few-docs` |
| PrefNugget random | `judges/prefnugget/workflow.yml` | `iter20bothties-few-random-pairs` |
| PrefNugget random docs | `judges/prefnugget/workflow.yml` | `iter20bothties-few-docs-random-pairs` |
| GroundedNugget best | `judges/grounded/workflow.yml` | `ground-response` |
| GroundedNugget best docs | `judges/grounded/workflow.yml` | `ground-docs` |
| GroundedNugget random | `judges/grounded/workflow.yml` | `ground-random-response` |
| GroundedNugget random docs | `judges/grounded/workflow.yml` | `ground-random-docs` |
| QueryOnlyNugget | `judges/queryonly/workflow.yml` | `prefnugget-rubric-response` |
| QueryOnlyNugget docs | `judges/queryonly/workflow.yml` | `prefnugget-rubric-docs` |

## Pseudocode

### Phase 1: Pairwise Preference Elicitation

```
for each topic:
    pairs = stratified_sample(responses, num_others=4)
    for (resp_A, resp_B) in pairs:
        winner  = LLM(PrefJudgment, passage_1=A, passage_2=B)
        winner2 = LLM(PrefJudgment, passage_1=B, passage_2=A)  # swapped
        record results; drop ties
    borda[resp] = wins - losses
    rank responses by borda score (best first)
```

### Phase 2a: PrefNugget -- Contrastive Nugget Extraction

```
for each topic:
    questions = []
    pairs = winner_loser_pairs sorted by borda(winner) + 0.99*borda(loser) desc
    for (winner, loser) in pairs[:100], taken 2 per topic per batch:
        if len(questions) >= 20: stop
        new_qs = LLM(IterativeExtractDifferentiatingNuggets,
                      winner_passage, loser_passage,
                      given_exam_questions=questions)[:2]   # max 2 new per pair
        questions += deduplicate(new_qs)
    nugget_bank[topic] = questions[:20]
```

### Phase 2b: GroundedNugget -- Single-Response Extraction

```
for each topic:
    questions = []
    responses sorted by borda score (best first)
    for response in responses[:100], taken 2 per topic per batch:
        if len(questions) >= 20: stop
        new_qs = LLM(GroundedIterativeNuggets,
                      response_passage=response,
                      given_exam_questions=questions)[:2]   # max 2 new per response
        questions += deduplicate(new_qs)
    nugget_bank[topic] = questions[:20]
```

### Phase 2c: QueryOnlyNugget -- Parametric Generation

```
for each topic:
    questions = LLM(IterativeGenerateNuggetQuestionsReportRequest,
                     query_title, query_background, query_problem)
    nugget_bank[topic] = deduplicate(questions)[:20]
```

### Phase 3: Response Grading

```
for each (response, nugget_question) in responses x nugget_bank[topic]:

    if grading == "response":
        grade = LLM(GradeNuggetAnswer,
                     question=nugget_question,
                     passage=response.text)                   # -> 0-5

    elif grading == "document_paragraphs":
        for paragraph in response.cited_documents.paragraphs:
            g = LLM(GradeNuggetAnswer,
                     question=nugget_question,
                     passage=paragraph)
        grade = max(g for all paragraphs)                     # best paragraph wins

    nugget_grades[response][nugget] = grade

# Aggregate per response
MAX_GRADE = max(grade for all nuggets)
```

## Prompt Definitions

### Phase 1: Preference Judging (`PrefJudgment`)

> You are a highly experienced and accurate assessor for TREC.
>
> Select the passage that answers the query better. Just answer 1 or 2, without any explanation or extra verbiage.
> If both passages are similar, select the simplest and clearest.

| Direction | Field | Description |
| --- | --- | --- |
| In | `query_title` | Query title |
| In | `query_background` | Background context for the query |
| In | `query_problem` | Problem statement to be addressed |
| In | `passage_1` | Passage 1 |
| In | `passage_2` | Passage 2 |
| Out | `better_passage` | 1 or 2 |
| Out | `confidence` | Score 0.0--1.0 |

### Phase 2a: Contrastive Nugget Extraction (`IterativeExtractDifferentiatingNuggets`)

> Compare Winner vs Loser RAG responses for a query. Focus on relevance, correctness, completeness.
>
> From given_exam_questions, identify or generate questions the Winner addresses much better than the Loser. Reuse questions where possible. New differentiating_questions must be brief, atomic questions about information the Winner handles much better.
>
> Avoid generic quality questions. Make questions self-contained (e.g., "Capital of France?" not "The capital?").

| Direction | Field | Description |
| --- | --- | --- |
| In | `query_title` | Query title |
| In | `query_background` | Background context for the query |
| In | `winner_passage` | The passage that won the comparison |
| In | `loser_passage` | The passage that lost the comparison |
| In | `given_exam_questions` | Given exam questions (from prior iterations) |
| Out | `differentiating_questions` | JSON array of new questions |
| Out | `reasoning` | Brief explanation of the analysis |
| Out | `confidence` | Score 0.0--1.0 |

### Phase 2b: Grounded Nugget Extraction (`GroundedIterativeNuggets`)

> Analyze the RAG response passage for a query. Focus on relevance, correctness, completeness.
>
> From given_exam_questions, identify or generate questions the response addresses best. Reuse questions where possible. New_questions must be brief, atomic questions about information the response handles best.
>
> Avoid generic quality questions. Make questions self-contained (e.g., "Capital of France?" not "The capital?").

| Direction | Field | Description |
| --- | --- | --- |
| In | `query_title` | Query title |
| In | `query_background` | Background context for the query |
| In | `response_passage` | RAG response passage |
| In | `given_exam_questions` | Given exam questions (from prior iterations) |
| Out | `new_questions` | JSON array of new questions |
| Out | `reasoning` | Brief explanation of the analysis |
| Out | `confidence` | Score 0.0--1.0 |

### Phase 2c: Query-Only Nugget Generation (`IterativeGenerateNuggetQuestionsReportRequest`)

> For a query as title, problem statement, and user background, imagine a good RAG response. Focus on relevance, correctness, completeness. Generate brief, atomic questions that target query-essential information which a good response should answer well.
>
> Avoid generic quality questions. Make questions self-contained (e.g., "Capital of France?" not "The capital?").

| Direction | Field | Description |
| --- | --- | --- |
| In | `query_title` | Query title |
| In | `query_background` | Background context for the query |
| In | `query_problem` | Problem statement to be addressed |
| Out | `questions` | List of concise questions |
| Out | `reasoning` | Brief explanation of the reasoning |
| Out | `confidence` | Score 0.0--1.0 |

### Phase 3: Nugget-Based Grading (`GradeNuggetAnswer`)

> Grade how well a passage answers a specific question.
>
> Can the question be answered based on the available context? Choose one:
> - 5: The answer is highly relevant, complete, and accurate.
> - 4: The answer is mostly relevant and complete but may have minor gaps or inaccuracies.
> - 3: The answer is partially relevant and complete, with noticeable gaps or inaccuracies.
> - 2: The answer has limited relevance and completeness, with significant gaps or inaccuracies.
> - 1: The answer is minimally relevant or complete, with substantial shortcomings.
> - 0: The answer is not relevant or complete at all.

| Direction | Field | Description |
| --- | --- | --- |
| In | `question` | The question to be answered |
| In | `passage` | The passage that may contain the answer |
| Out | `grade` | Grade 0--5 |
| Out | `reasoning` | Brief explanation of the grade |
| Out | `confidence` | Score 0.0--1.0 |

## Project Structure

```
prefnugget-starterkit/
├── pyproject.toml
├── README.md
├── judges.yml                     # Analysis config (judge name mappings)
├── judges/
│   ├── shared/
│   │   ├── pref_common.py         # Preference judging utilities
│   │   └── rubric_common.py       # Nugget grading utilities
│   ├── prefnugget/
│   │   ├── prefnugget_judge.py    # PrefNugget judge (contrastive extraction)
│   │   └── workflow.yml
│   ├── grounded/
│   │   ├── groundnugget_judge.py  # GroundedNugget judge (single-response extraction)
│   │   └── workflow.yml
│   └── queryonly/
│       ├── rubric_autojudge.py    # QueryOnlyNugget judge (parametric generation)
│       └── workflow.yml
├── data/
│   └── kiddie/                    # Synthetic test dataset
├── tests/
│   ├── test_prompt_snapshots.py   # Assert prompt text is identical to originals
│   └── test_imports.py            # Smoke test that all classes import
└── run_all_datasets.py            # Multi-dataset runner
```

## Meta-Evaluation

Install with evaluation tools:
```bash
uv pip install -e ".[evaluate]"
```

Run meta-evaluation:
```bash
auto-judge-evaluate meta-evaluate \
    --truth-leaderboard /path/to/truth.eval.ir_measures.txt \
    --truth-format ir_measures \
    --eval-format tot -i ./output/*eval.txt \
    --correlation kendall --on-missing default
```

## Dependencies

- [autojudge-base](https://github.com/trec-auto-judge/auto-judge-base) -- core protocols and data models
- [minima-llm](https://github.com/trec-auto-judge/minima-llm) -- LLM integration
- [DSPy](https://github.com/stanfordnlp/dspy) -- structured prediction

## License

MIT