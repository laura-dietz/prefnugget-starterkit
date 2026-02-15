"""Prompt snapshot tests.

Assert that DSPy signature prompts match expected substrings.
These tests catch accidental edits to prompts during refactoring.
Full cache-miss verification (running against an existing prompt cache)
provides the ultimate check.
"""

import pytest


# ── Phase 1: Preference Judging ──────────────────────────────────────────────


def test_pref_judgment_prompt():
    """PrefJudgment (no ties) prompt text."""
    from judges.shared.pref_common import PrefJudgment

    doc = PrefJudgment.__doc__
    assert "highly experienced and accurate assessor" in doc
    assert "select the simplest and clearest" in doc
    assert "Just answer 1 or 2" in doc


def test_pref_ties_judgment_prompt():
    """PrefTiesJudgment (ties allowed) prompt text."""
    from judges.shared.pref_common import PrefTiesJudgment

    doc = PrefTiesJudgment.__doc__
    assert "highly experienced and accurate assessor" in doc
    assert "answer with 0" in doc


def test_pref_judgment_fields():
    """PrefJudgment has expected input/output fields."""
    from judges.shared.pref_common import PrefJudgment

    schema = PrefJudgment.model_fields
    # Inputs
    assert "query_title" in schema
    assert "query_background" in schema
    assert "query_problem" in schema
    assert "passage_1" in schema
    assert "passage_2" in schema
    # Outputs
    assert "better_passage" in schema
    assert "confidence" in schema


# ── Phase 2a: Contrastive Nugget Extraction ──────────────────────────────────


def test_iterative_extract_prompt():
    """IterativeExtractDifferentiatingNuggets prompt text."""
    from judges.prefnugget.prefnugget_judge import (
        IterativeExtractDifferentiatingNuggets,
    )

    doc = IterativeExtractDifferentiatingNuggets.__doc__
    assert "Compare Winner vs Loser" in doc
    assert "given_exam_questions" in doc.lower() or "given_exam_questions" in str(
        IterativeExtractDifferentiatingNuggets.model_fields
    )
    assert "atomic questions" in doc
    assert "self-contained" in doc


def test_extract_differentiating_nuggets_prompt():
    """ExtractDifferentiatingNuggets (non-iterative) prompt text."""
    from judges.prefnugget.prefnugget_judge import ExtractDifferentiatingNuggets

    doc = ExtractDifferentiatingNuggets.__doc__
    assert "Winner and Loser RAG responses" in doc
    assert "atomic questions" in doc


def test_iterative_extract_fields():
    """IterativeExtractDifferentiatingNuggets has expected fields."""
    from judges.prefnugget.prefnugget_judge import (
        IterativeExtractDifferentiatingNuggets,
    )

    schema = IterativeExtractDifferentiatingNuggets.model_fields
    assert "query_title" in schema
    assert "winner_passage" in schema
    assert "loser_passage" in schema
    assert "given_exam_questions" in schema
    assert "differentiating_questions" in schema
    assert "reasoning" in schema
    assert "confidence" in schema


# ── Phase 2b: Grounded Nugget Extraction ─────────────────────────────────────


def test_grounded_iterative_nuggets_prompt():
    """GroundedIterativeNuggets prompt text."""
    from judges.grounded.groundnugget_judge import GroundedIterativeNuggets

    doc = GroundedIterativeNuggets.__doc__
    assert "Analyze the RAG response passage" in doc
    assert "atomic questions" in doc
    assert "self-contained" in doc


def test_grounded_iterative_nuggets_fields():
    """GroundedIterativeNuggets has expected fields."""
    from judges.grounded.groundnugget_judge import GroundedIterativeNuggets

    schema = GroundedIterativeNuggets.model_fields
    assert "query_title" in schema
    assert "response_passage" in schema
    assert "given_exam_questions" in schema
    assert "new_questions" in schema
    assert "reasoning" in schema
    assert "confidence" in schema


# ── Phase 2c: Query-Only Nugget Generation ───────────────────────────────────


def test_iterative_generate_nugget_questions_prompt():
    """IterativeGenerateNuggetQuestionsReportRequest prompt text."""
    from judges.queryonly.rubric_autojudge import (
        IterativeGenerateNuggetQuestionsReportRequest,
    )

    doc = IterativeGenerateNuggetQuestionsReportRequest.__doc__
    assert "imagine a good RAG response" in doc
    assert "atomic questions" in doc
    assert "self-contained" in doc


def test_iterative_generate_nugget_questions_fields():
    """IterativeGenerateNuggetQuestionsReportRequest has expected fields."""
    from judges.queryonly.rubric_autojudge import (
        IterativeGenerateNuggetQuestionsReportRequest,
    )

    schema = IterativeGenerateNuggetQuestionsReportRequest.model_fields
    assert "query_title" in schema
    assert "query_background" in schema
    assert "query_problem" in schema
    assert "questions" in schema
    assert "reasoning" in schema
    assert "confidence" in schema


# ── Phase 3: Nugget-Based Grading ────────────────────────────────────────────


def test_grade_nugget_answer_prompt():
    """GradeNuggetAnswer prompt text."""
    from judges.shared.rubric_common import GradeNuggetAnswer

    doc = GradeNuggetAnswer.__doc__
    assert "Grade how well a passage answers" in doc
    assert "5:" in doc  # grade scale
    assert "0:" in doc  # grade scale
    assert "not relevant or complete at all" in doc


def test_grade_nugget_answer_fields():
    """GradeNuggetAnswer has expected fields."""
    from judges.shared.rubric_common import GradeNuggetAnswer

    schema = GradeNuggetAnswer.model_fields
    assert "question" in schema
    assert "passage" in schema
    assert "grade" in schema
    assert "reasoning" in schema
    assert "confidence" in schema