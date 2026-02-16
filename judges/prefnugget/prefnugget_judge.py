#!/usr/bin/env python3
"""
PrefNuggetJudge: Extract differentiating nuggets from preference comparisons.

First runs pairwise comparisons (via pref_common), then extracts NuggetQuestion
objects explaining WHY the better response won.

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
# DSPy Signatures (for nugget extraction - specific to PrefNuggetJudge)
# =============================================================================


class ExtractDifferentiatingNuggets(dspy.Signature):
    __doc__ = dedent(
        """
        For a query as title, problem statement, and user background, you are given Winner and Loser RAG responses.
        Generate brief, atomic questions that target query-essential information which the Winner answers well
        and the Loser omits or mishandles.

        Only include differences that change the answer to the query (correctness, completeness,
        usefulness). Prefer short questions such as "Capital of USA?" or "Process of steel cooking?".
        Avoid generic quality questions.
        """
    )

    query_title: str = dspy.InputField(desc="Query title")
    query_background: str = dspy.InputField(desc="Background context for the query")
    winner_passage: str = dspy.InputField(desc="The passage that won the comparison")
    loser_passage: str = dspy.InputField(desc="The passage that lost the comparison")

    differentiating_questions: list[str] = dspy.OutputField(
        desc='JSON array, e.g. ["Capital of USA?", "Process to cook steel?"]'
    )
    reasoning: str = dspy.OutputField(
        desc="Brief explanation of why these questions differentiate the passages"
    )
    confidence: float = dspy.OutputField(
        desc="Confidence score from 0.0 to 1.0 indicating how certain you are"
    )



class IterativeExtractDifferentiatingNuggets(dspy.Signature):
    __doc__ = dedent(
      """
      Compare Winner vs Loser RAG responses for a query. Focus on relevance, correctness, completeness.

      From given_exam_questions, identify or generate questions the Winner addresses much better than the Loser.
      Reuse questions where possible. New differentiating_questions must be brief, 
      atomic questions about information the Winner handels much better.

      Avoid generic quality questions. 
      Make questions self-contained (e.g., "Capital of France?" not "The capital?").
      """
    )

    query_title: str = dspy.InputField(desc="Query title")
    query_background: str = dspy.InputField(desc="Background context for the query")
    winner_passage: str = dspy.InputField(desc="The passage that won the comparison")
    loser_passage: str = dspy.InputField(desc="The passage that lost the comparison")
    given_exam_questions: list[str] = dspy.InputField(desc="Given exam questions")

    differentiating_questions: Optional[List[str]] = dspy.OutputField(
        desc='Generated questions as a JSON array, e.g. ["Capital of USA?", "Process to cook steel?"]'
    )
    reasoning: str = dspy.OutputField(
        desc="Brief explanation of the analysis"
    )
    confidence: float = dspy.OutputField(
        desc="Confidence score from 0.0 to 1.0 indicating how certain you are"
    )


# =============================================================================
# Data Model (for nugget extraction - specific to PrefNuggetJudge)
# =============================================================================


class PrefNuggetData(BaseModel):
    """Data model for extracting differentiating nuggets from comparison pairs.

    Used by both iterative and non-iterative extraction paths.
    For iterative extraction, given_exam_questions is set before each batch.
    """

    # Input fields (for DSPy signature)
    query_id: str
    query_title: str
    query_background: str = ""
    winner_run_id: str
    loser_run_id: str
    winner_passage: str
    loser_passage: str
    given_exam_questions: Optional[List[str]] = None  # Set only for iterative extraction

    # Output fields (populated by LLM)
    differentiating_questions: List[str] = []


# =============================================================================
# PrefNugget-specific extraction functions
# =============================================================================


def extract_random_pairs(
    rag_responses: Sequence[Report],
    rag_topic_dict: Dict[str, Request],
) -> List[PrefNuggetData]:
    """Use random pairs (null hypothesis that winner/loser pairs are not helping)."""
    responses_by_topic: Dict[str, List[Report]] = collections.defaultdict(list)
    for r in rag_responses:
        responses_by_topic[r.metadata.topic_id].append(r)

    extraction_data: List[PrefNuggetData] = []

    for topic_id, responses in responses_by_topic.items():
        rng = Random(topic_id)
        rng.shuffle(responses)
        for i in range(len(responses)):
            for j in range(i+1, len(responses)):
                winner_response = responses[i]
                loser_response = responses[j]

                if not winner_response:
                    continue

                winner_run_id = winner_response.metadata.run_id
                loser_run_id = loser_response.metadata.run_id

                request = rag_topic_dict.get(topic_id)
                if not request:
                    continue

                extraction_data.append(
                    PrefNuggetData(
                        query_id=topic_id,
                        query_title=request.title or "",
                        query_background=request.background or "",
                        winner_run_id=winner_run_id,
                        loser_run_id=loser_run_id,
                        winner_passage=winner_response.get_report_text(),
                        loser_passage=loser_response.get_report_text(),
                    )
                )

    return extraction_data


def extract_winner_loser_pairs(
    aggregates: Dict[str, PrefAggregateResult],
    rag_responses: Sequence[Report],
    rag_topic_dict: Dict[str, Request],
) -> List[PrefNuggetData]:
    """Extract winner/loser pairs from preference aggregates."""
    responses_by_key: Dict[str, Report] = {
        f"{r.metadata.run_id}:{r.metadata.topic_id}": r for r in rag_responses
    }
    extraction_data: List[PrefNuggetData] = []
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
            pair_key = (topic_id, winner_run_id, loser_run_id)
            if pair_key not in seen_pairs:
                seen_pairs.add(pair_key)
                loser_key = f"{loser_run_id}:{topic_id}"
                loser_response = responses_by_key.get(loser_key)
                if loser_response:
                    extraction_data.append(
                        PrefNuggetData(
                            query_id=topic_id,
                            query_title=request.title or "",
                            query_background=request.background or "",
                            winner_run_id=winner_run_id,
                            loser_run_id=loser_run_id,
                            winner_passage=winner_response.get_report_text(),
                            loser_passage=loser_response.get_report_text(),
                        )
                    )

    return extraction_data


# =============================================================================
# PrefNuggetJudge Implementation
# =============================================================================


class PrefNuggetJudge(NuggetJudgeBase):
    """
    AutoJudge that extracts differentiating nuggets from PrefJudge comparisons.

    Requires responses to have evaldata with 'better_than' lists from PrefJudge.
    Produces NuggetBanks containing NuggetQuestion objects.
    """

    nugget_banks_type: Type[NuggetBanksProtocol] = NuggetBanks

    def _get_judge_name(self) -> str:
        return "PrefNuggetJudge"

    def _extract_data_from_aggregates(self, aggregates, rag_responses, rag_topic_dict):
        return extract_winner_loser_pairs(aggregates, rag_responses, rag_topic_dict)

    def _extract_data_random(self, rag_responses, rag_topic_dict):
        return extract_random_pairs(rag_responses, rag_topic_dict)

    def _get_sort_key_fn(self):
        def sort_key(x, borda_scores):
            return (
                borda_scores.get(f"{x.winner_run_id}:{x.query_id}", 0)
                + 0.99 * borda_scores.get(f"{x.loser_run_id}:{x.query_id}", 0)
            )
        return sort_key

    def _get_extraction_signature(self):
        return IterativeExtractDifferentiatingNuggets

    def _make_convert_output(self, max_questions_per_pair):
        def convert_output(prediction: dspy.Prediction, data: PrefNuggetData) -> None:
            differentiating_questions = getattr(prediction, "differentiating_questions", [])
            data.differentiating_questions = [
                q.strip() for q in (differentiating_questions or [])
                if q and q.strip()
            ][:max_questions_per_pair]
        return convert_output

    def _get_extracted_questions(self, data):
        return data.differentiating_questions

    def _init_exam_questions(self, extraction_data):
        for data in extraction_data:
            data.given_exam_questions = []

    def _set_exam_questions(self, data, questions):
        data.given_exam_questions = questions

    def _print_extraction_debug(self, extraction_data):
        for i, ed in enumerate(extraction_data[:20]):
            print(f"  [{i}] {ed.query_id}: {ed.winner_run_id} > {ed.loser_run_id}")

    def _supports_non_iterative(self):
        return True

    def _get_non_iterative_signature(self):
        return ExtractDifferentiatingNuggets


if __name__ == "__main__":
    auto_judge_to_click_command(PrefNuggetJudge(), "prefnugget_judge")()
