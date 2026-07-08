"""Behavior tests for the three nugget grade aggregators in rubric_common.

Covers:
- compute_nugget_aggregates           (response-only)
- compute_nugget_aggregates_for_documents (max-pool over docs/paragraphs)
- compute_nugget_aggregates_combined  (response gate * max-pooled doc grade)
"""

import pytest

from judges.shared.rubric_common import (
    NuggetGradeData,
    compute_nugget_aggregates,
    compute_nugget_aggregates_combined,
    compute_nugget_aggregates_for_documents,
)


def _make(run_id, nugget_id, grade, *, doc_id=None, paragraph_idx=None, query_id="t1"):
    return NuggetGradeData(
        run_id=run_id,
        query_id=query_id,
        nugget_id=nugget_id,
        question=f"Q for {nugget_id}?",
        passage="...",
        grade=grade,
        doc_id=doc_id,
        paragraph_idx=paragraph_idx,
    )


# =============================================================================
# compute_nugget_aggregates (response-only)
# =============================================================================


def test_response_aggregator_basic_coverage():
    """coverage_score = covered/total_in_bank, covered uses grade_threshold."""
    grade_data = [
        _make("run-A", "n1", 5),
        _make("run-A", "n2", 4),
        _make("run-A", "n3", 1),
    ]
    aggs = compute_nugget_aggregates(grade_data, {"t1": 3}, grade_threshold=4)
    a = aggs["run-A:t1"]
    assert a.covered_count == 2  # n1=5, n2=4 (>=4)
    assert a.coverage_score == pytest.approx(2 / 3)
    assert a.avg_grade == pytest.approx((5 + 4 + 1) / 3)
    assert a.max_grade == 5
    assert a.total_nuggets == 3
    assert a.graded_nuggets == 3
    assert set(a.nugget_grades.keys()) == {"n1", "n2", "n3"}
    assert a.nugget_grades["n1"]["grade"] == 5


def test_response_aggregator_uses_bank_denominator():
    """Avg/coverage divide by total_in_bank, not just graded count."""
    grade_data = [
        _make("run-A", "n1", 5),
        _make("run-A", "n2", 5),
        _make("run-A", "n3", 5),
    ]
    aggs = compute_nugget_aggregates(grade_data, {"t1": 5}, grade_threshold=4)
    a = aggs["run-A:t1"]
    assert a.coverage_score == pytest.approx(3 / 5)
    assert a.avg_grade == pytest.approx(15 / 5)
    assert a.graded_nuggets == 3
    assert a.total_nuggets == 5


def test_response_aggregator_zero_bank_zeroes_scores():
    """total_in_bank == 0 returns zeroed scores but preserves nugget_grades."""
    grade_data = [_make("run-A", "n1", 4)]
    aggs = compute_nugget_aggregates(grade_data, {}, grade_threshold=4)
    a = aggs["run-A:t1"]
    assert a.coverage_score == 0.0
    assert a.avg_grade == 0.0
    assert a.max_grade == 0
    assert a.total_nuggets == 0
    assert a.nugget_grades["n1"]["grade"] == 4


def test_response_aggregator_separates_runs():
    grade_data = [
        _make("run-A", "n1", 5),
        _make("run-B", "n1", 1),
    ]
    aggs = compute_nugget_aggregates(grade_data, {"t1": 1}, grade_threshold=4)
    assert aggs["run-A:t1"].covered_count == 1
    assert aggs["run-B:t1"].covered_count == 0


# =============================================================================
# compute_nugget_aggregates_for_documents (max-pool)
# =============================================================================


def test_doc_aggregator_max_pools_across_paragraphs():
    """For each (run, topic, nugget), the max grade across docs/paragraphs is used."""
    grade_data = [
        _make("run-A", "n1", 2, doc_id="d1", paragraph_idx=0),
        _make("run-A", "n1", 5, doc_id="d1", paragraph_idx=1),  # winner
        _make("run-A", "n1", 3, doc_id="d2", paragraph_idx=0),
        _make("run-A", "n2", 1, doc_id="d1", paragraph_idx=0),
    ]
    aggs = compute_nugget_aggregates_for_documents(grade_data, {"t1": 2}, grade_threshold=4)
    a = aggs["run-A:t1"]
    assert a.nugget_grades["n1"]["grade"] == 5
    assert a.nugget_grades["n1"]["doc_id"] == "d1"
    assert a.nugget_grades["n1"]["paragraph_idx"] == 1
    assert a.nugget_grades["n2"]["grade"] == 1
    assert a.covered_count == 1  # only n1 >= 4
    assert a.coverage_score == pytest.approx(1 / 2)
    assert a.max_grade == 5


def test_doc_aggregator_separates_runs():
    grade_data = [
        _make("run-A", "n1", 5, doc_id="d1"),
        _make("run-B", "n1", 2, doc_id="d1"),
    ]
    aggs = compute_nugget_aggregates_for_documents(grade_data, {"t1": 1}, grade_threshold=4)
    assert aggs["run-A:t1"].nugget_grades["n1"]["grade"] == 5
    assert aggs["run-B:t1"].nugget_grades["n1"]["grade"] == 2


# =============================================================================
# compute_nugget_aggregates_combined (response gate + max-pool * response)
# =============================================================================


def test_combined_below_response_threshold_yields_zero():
    """When response_grade < threshold, combined = 0 regardless of doc grade."""
    response_data = [_make("run-A", "n1", 2)]
    doc_data = [_make("run-A", "n1", 5, doc_id="d1")]  # would-be max, gate blocks
    aggs = compute_nugget_aggregates_combined(
        response_data, doc_data, {"t1": 1}, grade_threshold=4,
    )
    a = aggs["run-A:t1"]
    assert a.nugget_grades["n1"]["grade"] == 0
    assert a.covered_count == 0


def test_combined_product_when_gate_passes():
    """Combined = response_grade * max_doc_grade (int product)."""
    response_data = [
        _make("run-A", "n1", 5),
        _make("run-A", "n2", 4),
    ]
    doc_data = [
        _make("run-A", "n1", 5, doc_id="d1"),
        _make("run-A", "n2", 3, doc_id="d1"),
    ]
    aggs = compute_nugget_aggregates_combined(
        response_data, doc_data, {"t1": 2}, grade_threshold=4,
    )
    a = aggs["run-A:t1"]
    assert a.nugget_grades["n1"]["grade"] == 25  # 5 * 5
    assert a.nugget_grades["n2"]["grade"] == 12  # 4 * 3


def test_combined_max_pools_paragraphs_before_multiplying():
    """The doc side max-pools paragraphs before the multiply."""
    response_data = [_make("run-A", "n1", 5)]
    doc_data = [
        _make("run-A", "n1", 2, doc_id="d1", paragraph_idx=0),
        _make("run-A", "n1", 4, doc_id="d1", paragraph_idx=1),  # max
        _make("run-A", "n1", 1, doc_id="d2", paragraph_idx=0),
    ]
    aggs = compute_nugget_aggregates_combined(
        response_data, doc_data, {"t1": 1}, grade_threshold=4,
    )
    assert aggs["run-A:t1"].nugget_grades["n1"]["grade"] == 20  # 5 * max(2,4,1)


def test_combined_covered_uses_threshold_times_five():
    """Covered = combined >= grade_threshold * 5 (e.g. 4 -> 20)."""
    response_data = [
        _make("run-A", "n1", 5),
        _make("run-A", "n2", 4),
        _make("run-A", "n3", 4),
    ]
    doc_data = [
        _make("run-A", "n1", 5, doc_id="d1"),  # 25 ✓ covered
        _make("run-A", "n2", 5, doc_id="d1"),  # 20 ✓ covered
        _make("run-A", "n3", 4, doc_id="d1"),  # 16 ✗ not covered
    ]
    aggs = compute_nugget_aggregates_combined(
        response_data, doc_data, {"t1": 3}, grade_threshold=4,
    )
    a = aggs["run-A:t1"]
    assert a.covered_count == 2
    assert a.coverage_score == pytest.approx(2 / 3)


def test_combined_empty_doc_data_yields_zero():
    """No doc grades -> max_doc_grade = 0 -> combined = 0 even if gate passes."""
    response_data = [_make("run-A", "n1", 5)]
    aggs = compute_nugget_aggregates_combined(
        response_data, [], {"t1": 1}, grade_threshold=4,
    )
    assert aggs["run-A:t1"].nugget_grades["n1"]["grade"] == 0
    assert aggs["run-A:t1"].covered_count == 0


def test_combined_missing_doc_for_specific_nugget_yields_zero():
    """Gate-passing nugget without any doc grade -> combined = 0; siblings unaffected."""
    response_data = [
        _make("run-A", "n1", 5),
        _make("run-A", "n2", 5),  # gate passes, but no doc grade
    ]
    doc_data = [_make("run-A", "n1", 5, doc_id="d1")]
    aggs = compute_nugget_aggregates_combined(
        response_data, doc_data, {"t1": 2}, grade_threshold=4,
    )
    a = aggs["run-A:t1"]
    assert a.nugget_grades["n1"]["grade"] == 25
    assert a.nugget_grades["n2"]["grade"] == 0