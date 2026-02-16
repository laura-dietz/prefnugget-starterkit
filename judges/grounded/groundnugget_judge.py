#!/usr/bin/env python3
"""
GroundNuggetJudge: Extract nuggets from individual (winning) responses.

First runs pairwise comparisons (via pref_common) for ordering, then extracts
NuggetQuestion objects from individual winner responses (no loser needed).

This judge is primarily a nugget creator - judge() is inherited from NuggetJudgeBase.
"""
import collections
from random import Random
from textwrap import dedent
from typing import Dict, List, Optional, Sequence, Set, Type

import dspy
from pydantic import BaseModel

from autojudge_base import (
    NuggetBanksProtocol,
    Report,
    Request,
    auto_judge_to_click_command,
)
from autojudge_base.nugget_data import NuggetBanks

from judges.shared.pref_common import PrefAggregateResult
from judges.shared.nugget_judge_base import NuggetJudgeBase


# =============================================================================
# DSPy Signature (for nugget extraction - specific to GroundNuggetJudge)
# =============================================================================


class GroundedIterativeNuggets(dspy.Signature):
    __doc__ = dedent(
      """
      Analyze the RAG response passage for a query. Focus on relevance, correctness, completeness.

      From given_exam_questions, identify or generate questions the response addresses best.
      Reuse questions where possible. New_questions must be brief, 
      atomic questions about information the response handles best.

      Avoid generic quality questions. 
      Make questions self-contained (e.g., "Capital of France?" not "The capital?").
      """
    )

    query_title: str = dspy.InputField(desc="Query title")
    query_background: str = dspy.InputField(desc="Background context for the query")
    response_passage: str = dspy.InputField(desc="RAG response passage")
    given_exam_questions: list[str] = dspy.InputField(desc="Given exam questions")

    new_questions: Optional[List[str]] = dspy.OutputField(
        desc='Generated questions as a JSON array, e.g. ["Capital of USA?", "Process to cook steel?"]'
    )
    reasoning: str = dspy.OutputField(
        desc="Brief explanation of the analysis"
    )
    confidence: float = dspy.OutputField(
        desc="Confidence score from 0.0 to 1.0 indicating how certain you are"
    )


# =============================================================================
# Data Model (for nugget extraction - specific to GroundNuggetJudge)
# =============================================================================


class GroundNuggetData(BaseModel):
    """Data model for extracting nuggets from individual responses.

    For iterative extraction, given_exam_questions is set before each batch.
    """

    # Input fields (for DSPy signature)
    query_id: str
    query_title: str
    query_background: str = ""
    winner_run_id: str
    response_passage: str
    given_exam_questions: Optional[List[str]] = None  # Set only for iterative extraction

    # Output fields (populated by LLM)
    new_questions: List[str] = []


# =============================================================================
# Ground-specific extraction functions
# =============================================================================


def extract_random(
    rag_responses: Sequence[Report],
    rag_topic_dict: Dict[str, Request],
) -> List[GroundNuggetData]:
    """Use random ordering (null hypothesis)."""
    responses_by_topic: Dict[str, List[Report]] = collections.defaultdict(list)
    for r in rag_responses:
        responses_by_topic[r.metadata.topic_id].append(r)

    extraction_data: List[GroundNuggetData] = []

    for topic_id, responses in responses_by_topic.items():
        rng = Random(topic_id)
        rng.shuffle(responses)
        for i in range(len(responses)):
            winner_response = responses[i]

            if not winner_response:
                continue

            winner_run_id = winner_response.metadata.run_id

            request = rag_topic_dict.get(topic_id)
            if not request:
                continue

            extraction_data.append(
                GroundNuggetData(
                    query_id=topic_id,
                    query_title=request.title or "",
                    query_background=request.background or "",
                    winner_run_id=winner_run_id,
                    response_passage=winner_response.get_report_text(),
                )
            )

    return extraction_data


def extract_winners(
    aggregates: Dict[str, PrefAggregateResult],
    rag_responses: Sequence[Report],
    rag_topic_dict: Dict[str, Request],
) -> List[GroundNuggetData]:
    """Extract winners from preference aggregates (one entry per unique winner)."""
    responses_by_key: Dict[str, Report] = {
        f"{r.metadata.run_id}:{r.metadata.topic_id}": r for r in rag_responses
    }
    extraction_data: List[GroundNuggetData] = []
    seen_pairs: Set[tuple[str, str, str]] = set()

    for _key, agg in aggregates.items():
        topic_id = agg.topic_id
        winner_run_id = agg.run_id
        winner_key = f"{winner_run_id}:{topic_id}"
        winner_response = responses_by_key.get(winner_key)

        if not winner_response:
            continue

        request = rag_topic_dict.get(topic_id)
        if not request:
            continue

        for loser_run_id in agg.better_than:
            pair_key = (topic_id, winner_run_id, "")
            if pair_key not in seen_pairs:
                seen_pairs.add(pair_key)
                extraction_data.append(
                    GroundNuggetData(
                        query_id=topic_id,
                        query_title=request.title or "",
                        query_background=request.background or "",
                        winner_run_id=winner_run_id,
                        response_passage=winner_response.get_report_text(),
                    )
                )

    return extraction_data


# =============================================================================
# GroundNuggetJudge Implementation
# =============================================================================


class GroundNuggetJudge(NuggetJudgeBase):
    """
    AutoJudge that extracts nuggets from individual winning responses.

    Uses preference comparisons to identify winners, then extracts nuggets
    from the winner's response alone (no loser comparison).
    """

    nugget_banks_type: Type[NuggetBanksProtocol] = NuggetBanks

    def _get_judge_name(self) -> str:
        return "GroundNuggetJudge"

    def _extract_data_from_aggregates(self, aggregates, rag_responses, rag_topic_dict):
        return extract_winners(aggregates, rag_responses, rag_topic_dict)

    def _extract_data_random(self, rag_responses, rag_topic_dict):
        return extract_random(rag_responses, rag_topic_dict)

    def _get_sort_key_fn(self):
        def sort_key(x, borda_scores):
            return borda_scores.get(f"{x.winner_run_id}:{x.query_id}", 0)
        return sort_key

    def _get_extraction_signature(self):
        return GroundedIterativeNuggets

    def _make_convert_output(self, max_questions_per_pair):
        def convert_output(prediction: dspy.Prediction, data: GroundNuggetData) -> None:
            new_questions = getattr(prediction, "new_questions", [])
            data.new_questions = [
                q.strip() for q in (new_questions or [])
                if q and q.strip()
            ][:max_questions_per_pair]
        return convert_output

    def _get_extracted_questions(self, data):
        return data.new_questions

    def _init_exam_questions(self, extraction_data):
        for data in extraction_data:
            data.given_exam_questions = []

    def _set_exam_questions(self, data, questions):
        data.given_exam_questions = questions

    def _print_extraction_debug(self, extraction_data):
        for i, ed in enumerate(extraction_data[:20]):
            print(f"  [{i}] {ed.query_id}: {ed.winner_run_id} ")

    def _supports_non_iterative(self):
        return False

    def _get_non_iterative_signature(self):
        return None


if __name__ == "__main__":
    auto_judge_to_click_command(GroundNuggetJudge(), "prefnugget_judge")()
