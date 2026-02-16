"""
Shared base class for nugget-based judges (PrefNugget, GroundNugget).

Extracts ~500 lines of duplicated code into:
- NuggetJudgeBase: Abstract base with shared judge/create_nuggets logic
- QuestionTracker: Tracks unique questions per topic during iterative extraction
- Utility functions: build_response_lookups, run_preference_phase, chunking, etc.
- Preference checkpoint I/O: save_preferences, load_preferences
"""
import abc
import collections
import json
import sys
from itertools import groupby
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Set, Type

import dspy
from pydantic import BaseModel

from autojudge_base import (
    AutoJudge,
    Leaderboard,
    LeaderboardBuilder,
    LeaderboardSpec,
    LlmConfigBase,
    MeasureSpec,
    NuggetBanksProtocol,
    Qrels,
    Report,
    Request,
    format_preview,
)
from autojudge_base.nugget_data import NuggetBanks, write_nugget_banks
from minima_llm import MinimaLlmConfig
from minima_llm.dspy_adapter import run_dspy_batch_generic

from judges.shared.pref_common import (
    PrefAggregateResult,
    PrefJudgment,
    PrefTiesJudgment,
    compute_pref_aggregates,
    prepare_prompts,
    run_pref_judgment_batch,
    PrefJudgeData,
)
from judges.shared.rubric_common import (
    NuggetGradeData,
    GradeNuggetAnswer,
    prepare_nugget_grade_data,
    prepare_nugget_grade_data_for_documents,
    compute_nugget_aggregates,
    compute_nugget_aggregates_for_documents,
    build_nugget_banks,
    collect_nugget_relevant_docs,
    write_nugget_docs_collaborator,
    nugget_docs_to_nugget_banks,
)


# =============================================================================
# Leaderboard Spec (shared by PrefNugget and GroundNugget)
# =============================================================================

PREFNUGGET_SPEC = LeaderboardSpec(measures=(
    MeasureSpec("NUGGET_COVERAGE"),
    MeasureSpec("AVG_GRADE"),
    MeasureSpec("MAX_GRADE"),
    MeasureSpec("COVERED_COUNT"),
))


# =============================================================================
# Utility: LLM config conversion
# =============================================================================

def _to_minima_config(llm_config: LlmConfigBase) -> MinimaLlmConfig:
    """Convert LlmConfigBase to MinimaLlmConfig (env vars as base, raw dict overlaid)."""
    return MinimaLlmConfig.from_dict(llm_config.raw or {})


# =============================================================================
# QuestionTracker
# =============================================================================

class QuestionTracker:
    """Track unique questions and their occurrence counts per topic."""

    def __init__(self):
        # counts[query_id][question] = count
        self._counts: Dict[str, Dict[str, int]] = collections.defaultdict(lambda: collections.defaultdict(int))
        self._topics_done: Set[str] = set()  # Topics that have collected enough questions

    def add(self, query_id: str, question: str, count: int = 1) -> None:
        """Add a question, incrementing its count."""
        self._counts[query_id][question] += count

    def add_all(self, query_id: str, questions: List[str], count: int = 1) -> None:
        """Add multiple questions, incrementing each count."""
        for q in questions:
            self._counts[query_id][q] += count

    def questions(self, query_id: str) -> List[str]:
        """Get list of unique questions for a topic."""
        return list(self._counts[query_id].keys())

    def count(self, query_id: str, question: str) -> int:
        """Get count for a specific question."""
        return self._counts[query_id].get(question, 0)

    def counts_dict(self, query_id: str) -> Dict[str, int]:
        """Get all counts for a topic."""
        return dict(self._counts[query_id])

    def num_questions(self, query_id: str) -> int:
        """Get number of unique questions for a topic."""
        return len(self._counts[query_id])

    def items(self):
        """Iterate over (query_id, questions_dict) pairs."""
        return self._counts.items()

    def top_questions(self, query_id: str, n: int) -> List[str]:
        """Get top n questions by count, sorted descending."""
        return sorted(
            self._counts[query_id].keys(),
            key=lambda q: self._counts[query_id][q],
            reverse=True
        )[:n]

    def is_done(self, query_id: str) -> bool:
        """Check if topic has collected enough questions."""
        return query_id in self._topics_done

    def mark_done(self, query_id: str) -> None:
        """Mark topic as having collected enough questions."""
        self._topics_done.add(query_id)

    def check_and_mark_done(self, query_id: str, stop_at_count: int) -> bool:
        """Mark done if threshold exceeded. Returns True if now done."""
        if self.num_questions(query_id) > stop_at_count:
            self._topics_done.add(query_id)
            return True
        return False

    def check_all_done(self, stop_at_count: int) -> None:
        """Check all topics and mark done if threshold exceeded."""
        for query_id in self._counts.keys():
            if self.num_questions(query_id) > stop_at_count:
                self._topics_done.add(query_id)


def _print_tracker(tracker: QuestionTracker) -> str:
    lines = []
    for query_id, counts in sorted(tracker.items())[:5]:
        # Sort by count descending, then alphabetically for determinism
        sorted_qs = sorted(counts.keys(), key=lambda q: (-counts[q], q))[:5]
        formatted = [f"  - {q} ({counts[q]})" for q in sorted_qs]
        lines.append(f"{query_id}: {len(counts)} questions:\n" + "\n".join(formatted))
    return "\n".join(lines)


# =============================================================================
# Response lookup utilities
# =============================================================================

def build_response_lookups(
    rag_responses: Sequence[Report],
    rag_topics: Sequence[Request],
) -> tuple[Dict[str, Request], Dict[str, List[Report]]]:
    """Build lookup structures for responses and topics.

    Returns:
        rag_topic_dict: topic_id -> Request
        rag_response_by_topic: topic_id -> List[Report]
    """
    rag_topic_dict: Dict[str, Request] = {t.request_id: t for t in rag_topics}
    rag_response_by_topic: Dict[str, List[Report]] = {
        topic: list(responses)
        for topic, responses in groupby(
            sorted(rag_responses, key=lambda r: r.metadata.topic_id),
            key=lambda r: r.metadata.topic_id,
        )
    }
    return rag_topic_dict, rag_response_by_topic


# =============================================================================
# Phase 1: Preference comparisons
# =============================================================================

def run_preference_phase(
    rag_topic_dict: Dict[str, Request],
    rag_response_by_topic: Dict[str, List[Report]],
    llm_config: LlmConfigBase,
    num_pivot: int,
    num_others: int,
    no_dupes: bool,
    pref_judge: Literal['must_decide', 'ties_allowed'] = 'must_decide',
    judge_name: str = "NuggetJudge",
) -> Optional[tuple[list, Dict[str, PrefAggregateResult]]]:
    """Run pairwise preference comparisons and compute aggregates.

    Returns:
        Tuple of (grade_data, aggregates) or None if no comparison pairs generated.
        grade_data: List of comparison results (flipped for position bias, ties dropped)
        aggregates: Dict of computed aggregates with better_than/worse_than lists
    """
    print(f"{judge_name}: Running pairwise comparisons (num_pivot={num_pivot}, num_others={num_others})...")
    grade_data = prepare_prompts(
        rag_topic_dict=rag_topic_dict,
        rag_response_by_topic=rag_response_by_topic,
        num_pivot=num_pivot,
        num_others=num_others,
        no_dupes=no_dupes
    )

    if not grade_data:
        print(f"{judge_name}: No comparison pairs generated")
        return None

    pref_signature = PrefTiesJudgment if pref_judge == "ties_allowed" else PrefJudgment
    grade_data = run_pref_judgment_batch(grade_data, llm_config, signature=pref_signature)
    print(f"{judge_name}: Completed {len(grade_data)} pairwise comparisons")

    # Include pairs in reverse for position bias handling
    grade_data = grade_data + [data.flip() for data in grade_data]
    # Drop ties (only keep pairs with a clear winner)
    grade_data = [d for d in grade_data if d.better_passage in [1, 2]]

    # Compute aggregates (better_than/worse_than lists)
    aggregates = compute_pref_aggregates(grade_data)

    return grade_data, aggregates


# =============================================================================
# Phase 1 checkpoint persistence
# =============================================================================

def save_preferences(
    grade_data: List[PrefJudgeData],
    aggregates: Dict[str, PrefAggregateResult],
    path: Path,
) -> None:
    """Save Phase 1 checkpoint to JSONL."""
    with open(path, "w") as f:
        for d in grade_data:
            record = d.model_dump()
            record["_type"] = "comparison"
            f.write(json.dumps(record) + "\n")
        for key, agg in aggregates.items():
            record = agg.model_dump()
            record["_type"] = "aggregate"
            f.write(json.dumps(record) + "\n")
    print(f"Saved Phase 1 preferences to {path} ({len(grade_data)} comparisons, {len(aggregates)} aggregates)")


def load_preferences(path: Path) -> tuple[List[PrefJudgeData], Dict[str, PrefAggregateResult]]:
    """Load Phase 1 checkpoint from JSONL."""
    grade_data: List[PrefJudgeData] = []
    aggregates: Dict[str, PrefAggregateResult] = {}

    with open(path) as f:
        for line in f:
            record = json.loads(line)
            record_type = record.pop("_type", None)
            if record_type == "comparison":
                grade_data.append(PrefJudgeData(**record))
            elif record_type == "aggregate":
                agg = PrefAggregateResult(**record)
                key = f"{agg.run_id}:{agg.topic_id}"
                aggregates[key] = agg

    print(f"Loaded Phase 1 preferences from {path} ({len(grade_data)} comparisons, {len(aggregates)} aggregates)")
    return grade_data, aggregates


# =============================================================================
# Chunking utilities
# =============================================================================

def chunk_by_query(
    lst: List[Any],
    borda_scores: Dict[str, int],
    nugget_gen_order: Literal["both", "winner", "as_provided"],
    sort_key_fn: Callable[[Any, Dict[str, int]], float],
    num_per_query: int = 2,
    max_pairs_considered: int = -1,
) -> List[List[Any]]:
    """Split list into chunks with at most `num_per_query` items per query_id.

    Items with higher sort keys are prioritized first.

    Args:
        lst: List of data items with query_id attribute
        borda_scores: Mapping of "run_id:topic_id" -> borda_score
        nugget_gen_order: Sorting strategy
        sort_key_fn: Callable(item, borda_scores) -> sort value for 'both' ordering
        num_per_query: Maximum items per query_id in each chunk
        max_pairs_considered (k): only look at the top-k pairs

    Returns:
        List of batches, each respecting the per-query limit
    """
    if not lst:
        return []

    # First split by topic (convert groupby iterator to dict)
    sorted_per_topic: Dict[str, list] = {
        k: list(g)
        for k, g in groupby(sorted(lst, key=lambda d: d.query_id), key=lambda d: d.query_id)
    }

    for query_id in sorted_per_topic:
        topic_lst = sorted_per_topic[query_id]

        # Sort by quality within each topic
        if nugget_gen_order == 'both':
            topic_lst = sorted(
                topic_lst,
                key=lambda x: sort_key_fn(x, borda_scores),
                reverse=True,
            )
        elif nugget_gen_order == 'winner':
            topic_lst = sorted(
                topic_lst,
                key=lambda x: borda_scores.get(f"{x.winner_run_id}:{x.query_id}", 0),
                reverse=True,
            )
        elif nugget_gen_order == 'as_provided':
            topic_lst = topic_lst

        # Limit to top-k pairs per topic (if max_pairs_considered > 0)
        if max_pairs_considered > 0:
            sorted_per_topic[query_id] = topic_lst[:max_pairs_considered]
        else:
            sorted_per_topic[query_id] = topic_lst

    # Build chunks by popping `n` per topic
    chunks = []
    while any(x for x in sorted_per_topic.values()):
        chunk = []

        for query_id in sorted_per_topic:
            for _rounds in range(num_per_query):
                topic_lst = sorted_per_topic[query_id]
                if topic_lst:
                    elem = topic_lst.pop(0)
                    sorted_per_topic[query_id] = topic_lst
                    chunk.append(elem)

        chunks.append(chunk)

    return chunks


# =============================================================================
# NuggetJudgeBase
# =============================================================================

class NuggetJudgeBase(AutoJudge, abc.ABC):
    """Abstract base class for nugget-based judges (PrefNugget, GroundNugget).

    Provides shared implementations of:
    - create_qrels() -> None
    - filter_non_topic_responses()
    - judge() -> Leaderboard (Phase 3 grading)
    - _build_leaderboard()
    - create_nuggets() template method with abstract hooks
    """

    nugget_banks_type: Type[NuggetBanksProtocol] = NuggetBanks

    def __init__(self):
        pass

    # -- Abstract hooks for create_nuggets --

    @abc.abstractmethod
    def _get_judge_name(self) -> str:
        """Return human-readable judge name for log messages."""

    @abc.abstractmethod
    def _extract_data_from_aggregates(
        self,
        aggregates: Dict[str, PrefAggregateResult],
        rag_responses: Sequence[Report],
        rag_topic_dict: Dict[str, Request],
    ) -> list:
        """Extract data items from preference aggregates for Phase 2."""

    @abc.abstractmethod
    def _extract_data_random(
        self,
        rag_responses: Sequence[Report],
        rag_topic_dict: Dict[str, Request],
    ) -> list:
        """Extract data items using random pairing for Phase 2."""

    @abc.abstractmethod
    def _get_sort_key_fn(self) -> Callable[[Any, Dict[str, int]], float]:
        """Return sort key function for chunk_by_query 'both' ordering."""

    @abc.abstractmethod
    def _get_extraction_signature(self) -> type:
        """Return the DSPy signature class for iterative extraction."""

    @abc.abstractmethod
    def _make_convert_output(self, max_questions_per_pair: int) -> Callable:
        """Return a convert_output function for DSPy batch processing."""

    @abc.abstractmethod
    def _get_extracted_questions(self, data: Any) -> List[str]:
        """Get the list of extracted questions from a data item."""

    @abc.abstractmethod
    def _init_exam_questions(self, extraction_data: list) -> None:
        """Initialize given_exam_questions fields on extraction data items."""

    @abc.abstractmethod
    def _set_exam_questions(self, data: Any, questions: List[str]) -> None:
        """Set the given_exam_questions field on a data item."""

    @abc.abstractmethod
    def _print_extraction_debug(self, extraction_data: list) -> None:
        """Print debug info about extraction pairs."""

    @abc.abstractmethod
    def _supports_non_iterative(self) -> bool:
        """Whether this judge supports non-iterative extraction."""

    @abc.abstractmethod
    def _get_non_iterative_signature(self) -> Optional[type]:
        """Return the DSPy signature for non-iterative extraction, or None."""

    # -- Shared implementations --

    def create_qrels(
        self,
        rag_responses: Sequence[Report],
        rag_topics: Sequence[Request],
        llm_config: LlmConfigBase,
        nugget_banks: Optional[NuggetBanksProtocol] = None,
        **kwargs
    ) -> Optional[Qrels]:
        """Nugget judges do not produce qrels."""
        return None

    def filter_non_topic_responses(self, rag_responses: Sequence[Report], topic_ids: Set[str]) -> Sequence[Report]:
        broken: bool = False
        broken_run_ids = []
        for r in rag_responses:
            if r.metadata.run_id not in broken_run_ids:
                if r.metadata.topic_id not in topic_ids:
                    print(f"Warning, report of run {r.metadata.run_id} is about topic {r.metadata.request_id}, which is not in topic_ids {format_preview(list(topic_ids), limit=10)}", file=sys.stderr)
                    broken = True
                    broken_run_ids.append(r.metadata.run_id)

        if broken:
            return list(filter(lambda r: r.metadata.request_id in topic_ids, rag_responses))
        else:
            return rag_responses

    def _flatten_extraction_results(
        self,
        extraction_data: list,
        rag_topic_dict: Dict[str, Request],
    ) -> Dict[str, tuple[str, List[str]]]:
        """Convert extraction results to format expected by build_nugget_banks.

        Returns:
            Dict mapping topic_id -> (title, list of question strings)
        """
        questions_by_topic: Dict[str, List[str]] = {}
        for data in extraction_data:
            questions_by_topic.setdefault(data.query_id, []).extend(
                self._get_extracted_questions(data)
            )

        return {
            topic_id: (rag_topic_dict[topic_id].title or topic_id, questions)
            for topic_id, questions in questions_by_topic.items()
            if topic_id in rag_topic_dict
        }

    def create_nuggets(
        self,
        rag_responses: Sequence[Report],
        rag_topics: Sequence[Request],
        llm_config: LlmConfigBase,
        pref_judge: Literal['must_decide', 'ties_allowed'],
        iterative_nuggets: bool,
        max_nuggets_per_topic: int,
        stop_collecting_at_nuggets_per_topic: int,
        gen_batch_num_per_query: int,
        max_pairs_considered: int,
        nugget_gen_order: Literal["both", "winner"],
        max_questions_per_pair: int = 5,
        num_pivot: int = 0,
        num_others: int = 8,
        no_dupes: bool = True,
        random_pairs: bool = False,
        pref_input: Optional[str] = None,
        filebase: str = "default",
        **kwargs,
    ) -> Optional[NuggetBanksProtocol]:
        """Extract nuggets using the template method pattern.

        Phase 1: Pairwise preferences (or load from checkpoint)
        Phase 2: Iterative nugget extraction
        """
        judge_name = self._get_judge_name()

        # Build lookup structures
        rag_topic_dict, rag_response_by_topic = build_response_lookups(rag_responses, rag_topics)
        rag_responses = self.filter_non_topic_responses(rag_responses, rag_topic_dict.keys())

        extraction_data: list
        borda_scores: Dict[str, int]

        if pref_input:
            # Load Phase 1 checkpoint
            print(
                f"WARNING: Loading Phase 1 preferences from {pref_input}. "
                "Skipping preference LLM calls.",
                file=sys.stderr,
            )
            _grade_data, aggregates = load_preferences(Path(pref_input))
            extraction_data = self._extract_data_from_aggregates(aggregates, rag_responses, rag_topic_dict)
            borda_scores = {key: agg.borda_score for key, agg in aggregates.items()}
        elif not random_pairs:
            # Step 1: Run pairwise preference comparisons
            result = run_preference_phase(
                rag_topic_dict=rag_topic_dict,
                rag_response_by_topic=rag_response_by_topic,
                llm_config=llm_config,
                num_pivot=num_pivot,
                num_others=num_others,
                no_dupes=no_dupes,
                pref_judge=pref_judge,
                judge_name=judge_name,
            )
            if result is None:
                return None
            grade_data, aggregates = result

            # Save Phase 1 checkpoint
            save_preferences(grade_data, aggregates, Path(f"{filebase}.preferences.jsonl"))

            # Step 2: Extract data from aggregates
            extraction_data = self._extract_data_from_aggregates(aggregates, rag_responses, rag_topic_dict)
            borda_scores = {key: agg.borda_score for key, agg in aggregates.items()}
        else:
            extraction_data = self._extract_data_random(rag_responses, rag_topic_dict)
            borda_scores = dict()

        # Initialize given_exam_questions for iterative extraction
        self._init_exam_questions(extraction_data)

        if not extraction_data:
            print(f"{judge_name}: No extraction data found after comparison")
            return None

        # Debug: print extraction pairs
        self._print_extraction_debug(extraction_data)

        print(f"{judge_name}: Extracting nuggets from {len(extraction_data)} items...")

        convert_output = self._make_convert_output(max_questions_per_pair)

        if iterative_nuggets:
            extraction_data_chunks = chunk_by_query(
                extraction_data,
                borda_scores=borda_scores,
                nugget_gen_order="as_provided" if random_pairs else nugget_gen_order,
                sort_key_fn=self._get_sort_key_fn(),
                num_per_query=gen_batch_num_per_query,
                max_pairs_considered=max_pairs_considered,
            )
            tracker = QuestionTracker()
            extraction_result_data = list()

            for chunk_idx, extraction_chunk in enumerate(extraction_data_chunks):
                # Skip prompts for topics that already have enough questions
                extraction_chunk = [d for d in extraction_chunk if not tracker.is_done(d.query_id)]

                if not extraction_chunk:
                    continue

                # Set questions so far
                for data in extraction_chunk:
                    self._set_exam_questions(data, tracker.questions(data.query_id))

                # Run LLM extraction
                full_config = _to_minima_config(llm_config)
                extraction_chunk = run_dspy_batch_generic(
                    extraction_chunk,
                    self._get_extraction_signature(),
                    convert_output,
                    full_config,
                )

                for data in extraction_chunk:
                    tracker.add_all(data.query_id, self._get_extracted_questions(data))

                tracker.check_all_done(stop_at_count=stop_collecting_at_nuggets_per_topic)
                extraction_result_data.extend(extraction_chunk)

                print(f"-- {judge_name}: Finished extracting nuggets pass {chunk_idx}. Questions:\n{_print_tracker(tracker)}")

            print(f"{judge_name}: Finished extracting nuggets")
            print(f"Question counts: {dict(tracker.items())}")

        else:
            if self._supports_non_iterative():
                full_config = _to_minima_config(llm_config)
                extraction_result_data = run_dspy_batch_generic(
                    extraction_data,
                    self._get_non_iterative_signature(),
                    convert_output,
                    full_config,
                )
                print(f"{judge_name}: Finished extracting nuggets")
            else:
                print("not supported")
                print(f"{judge_name}: Finished extracting nuggets")
                extraction_result_data = []

        # Build NuggetBanks from results
        questions_by_topic = self._flatten_extraction_results(extraction_result_data, rag_topic_dict)
        return build_nugget_banks(questions_by_topic, max_per_topic=max_nuggets_per_topic)

    def judge(
        self,
        rag_responses: Sequence[Report],
        rag_topics: Sequence[Request],
        llm_config: LlmConfigBase,
        nugget_banks: Optional[NuggetBanksProtocol] = None,
        grade_threshold: int = 4,
        on_missing_evals: str = "fix_aggregate",
        grade_text: Literal["response", "document", "document_paragraphs"] = "response",
        filebase: str = "prefnugget",
        **kwargs
    ) -> Leaderboard:
        """Grade each response against all nuggets for its topic."""
        judge_name = self._get_judge_name()

        if nugget_banks is None:
            raise ValueError(f"{judge_name} requires nugget_banks. Run create_nuggets first or provide --nugget-banks.")

        self.expected_topic_ids = [t.request_id for t in rag_topics]

        # Prepare grading data
        print(f"{judge_name}: Preparing grade data...")
        if grade_text == "response":
            grade_data, nuggets_per_topic = prepare_nugget_grade_data(rag_responses, nugget_banks)
        else:
            grade_data, nuggets_per_topic = prepare_nugget_grade_data_for_documents(
                rag_responses, nugget_banks, use_paragraphs=(grade_text == "document_paragraphs")
            )

        # Run LLM grading
        print(f"{judge_name}: Grading responses...")
        full_config = _to_minima_config(llm_config)
        grade_data = run_dspy_batch_generic(
            grade_data,
            GradeNuggetAnswer,
            GradeNuggetAnswer.convert_prompt_output,
            full_config,
        )
        print(f"{judge_name}: Finished grading")

        # Aggregate grades
        if grade_text == "response":
            aggregates = compute_nugget_aggregates(grade_data, nuggets_per_topic, grade_threshold)
        else:
            aggregates = compute_nugget_aggregates_for_documents(grade_data, nuggets_per_topic, grade_threshold)

        # Export nugget-relevant documents (only for document-level grading)
        if grade_text != "response":
            nugget_doc_topics = collect_nugget_relevant_docs(grade_data, grade_threshold)
            if nugget_doc_topics:
                write_nugget_docs_collaborator(nugget_doc_topics, Path(f"{filebase}.nugget-docs"))
                doc_banks = nugget_docs_to_nugget_banks(nugget_doc_topics)
                write_nugget_banks(doc_banks, Path(f"{filebase}.nugget-docs.nuggets.jsonl"))

        # Update Report.evaldata
        for response in rag_responses:
            response_key = f"{response.metadata.run_id}:{response.metadata.topic_id}"
            if response_key in aggregates:
                agg = aggregates[response_key]
                response.evaldata = {
                    "nugget_grades": agg.nugget_grades,
                    "coverage_score": agg.coverage_score,
                    "avg_grade": agg.avg_grade,
                    "max_grade": agg.max_grade,
                    "covered_count": agg.covered_count,
                    "total_nuggets": agg.total_nuggets,
                    "graded_nuggets": agg.graded_nuggets,
                }

        # Build leaderboard
        leaderboard = self._build_leaderboard(aggregates, on_missing_evals)
        leaderboard.verify(warn=True, expected_topic_ids=self.expected_topic_ids, on_missing=on_missing_evals)

        return leaderboard

    def _build_leaderboard(self, aggregates: Dict[str, Any], on_missing_evals: str) -> Leaderboard:
        """Build leaderboard from aggregated response grades."""
        b = LeaderboardBuilder(PREFNUGGET_SPEC)

        for response_key, agg in aggregates.items():
            run_id, topic_id = response_key.split(":", 1)
            b.add(
                run_id=run_id,
                topic_id=topic_id,
                values={
                    "NUGGET_COVERAGE": agg.coverage_score,
                    "AVG_GRADE": agg.avg_grade,
                    "MAX_GRADE": agg.max_grade,
                    "COVERED_COUNT": float(agg.covered_count),
                }
            )

        leaderboard = b.build(expected_topic_ids=self.expected_topic_ids, on_missing=on_missing_evals)
        leaderboard.verify(expected_topic_ids=self.expected_topic_ids, warn=False, on_missing=on_missing_evals)
        return leaderboard
