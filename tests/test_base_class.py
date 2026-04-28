"""Tests for NuggetJudgeBase shared utilities.

Covers QuestionTracker and preference checkpoint round-trip.
"""

import json
import pytest
from pathlib import Path


# =============================================================================
# QuestionTracker tests
# =============================================================================


def test_question_tracker_add_and_count():
    """QuestionTracker tracks question counts correctly."""
    from judges.shared.nugget_judge_base import QuestionTracker

    tracker = QuestionTracker()
    tracker.add("topic-1", "What is the capital?")
    tracker.add("topic-1", "What is the capital?")
    tracker.add("topic-1", "What is the population?")

    assert tracker.count("topic-1", "What is the capital?") == 2
    assert tracker.count("topic-1", "What is the population?") == 1
    assert tracker.num_questions("topic-1") == 2


def test_question_tracker_add_all():
    """QuestionTracker.add_all adds multiple questions."""
    from judges.shared.nugget_judge_base import QuestionTracker

    tracker = QuestionTracker()
    tracker.add_all("topic-1", ["Q1", "Q2", "Q3"])

    assert tracker.num_questions("topic-1") == 3
    assert tracker.count("topic-1", "Q1") == 1


def test_question_tracker_questions_list():
    """QuestionTracker.questions returns unique question strings."""
    from judges.shared.nugget_judge_base import QuestionTracker

    tracker = QuestionTracker()
    tracker.add("topic-1", "Q1")
    tracker.add("topic-1", "Q1")  # duplicate
    tracker.add("topic-1", "Q2")

    questions = tracker.questions("topic-1")
    assert len(questions) == 2
    assert "Q1" in questions
    assert "Q2" in questions


def test_question_tracker_top_questions():
    """QuestionTracker.top_questions returns most frequent first."""
    from judges.shared.nugget_judge_base import QuestionTracker

    tracker = QuestionTracker()
    tracker.add("t", "rare", count=1)
    tracker.add("t", "common", count=5)
    tracker.add("t", "medium", count=3)

    top = tracker.top_questions("t", 2)
    assert top == ["common", "medium"]


def test_question_tracker_done_tracking():
    """QuestionTracker tracks topic completion."""
    from judges.shared.nugget_judge_base import QuestionTracker

    tracker = QuestionTracker()
    assert not tracker.is_done("topic-1")

    tracker.mark_done("topic-1")
    assert tracker.is_done("topic-1")


def test_question_tracker_check_and_mark_done():
    """QuestionTracker.check_and_mark_done marks done when threshold exceeded."""
    from judges.shared.nugget_judge_base import QuestionTracker

    tracker = QuestionTracker()
    tracker.add_all("t", [f"Q{i}" for i in range(5)])  # 5 questions

    assert not tracker.check_and_mark_done("t", stop_at_count=5)  # not > 5
    assert not tracker.is_done("t")

    tracker.add("t", "Q5")  # now 6 unique questions
    assert tracker.check_and_mark_done("t", stop_at_count=5)  # > 5
    assert tracker.is_done("t")


def test_question_tracker_check_all_done():
    """QuestionTracker.check_all_done marks all qualifying topics."""
    from judges.shared.nugget_judge_base import QuestionTracker

    tracker = QuestionTracker()
    tracker.add_all("t1", [f"Q{i}" for i in range(10)])  # 10 questions
    tracker.add_all("t2", [f"Q{i}" for i in range(3)])   # 3 questions

    tracker.check_all_done(stop_at_count=5)
    assert tracker.is_done("t1")      # 10 > 5
    assert not tracker.is_done("t2")  # 3 not > 5


def test_question_tracker_empty_topic():
    """QuestionTracker handles queries for unknown topics gracefully."""
    from judges.shared.nugget_judge_base import QuestionTracker

    tracker = QuestionTracker()
    assert tracker.questions("nonexistent") == []
    assert tracker.num_questions("nonexistent") == 0
    assert tracker.count("nonexistent", "Q1") == 0
    assert not tracker.is_done("nonexistent")


def test_print_tracker():
    """_print_tracker produces readable output."""
    from judges.shared.nugget_judge_base import QuestionTracker, _print_tracker

    tracker = QuestionTracker()
    tracker.add("topic-1", "Q1", count=3)
    tracker.add("topic-1", "Q2", count=1)

    output = _print_tracker(tracker)
    assert "topic-1" in output
    assert "Q1" in output
    assert "(3)" in output


# =============================================================================
# Preference checkpoint round-trip tests
# =============================================================================


def test_preference_checkpoint_round_trip(tmp_path):
    """save_preferences + load_preferences is a lossless round-trip."""
    from judges.shared.nugget_judge_base import save_preferences, load_preferences
    from judges.shared.pref_common import PrefJudgeData, PrefAggregateResult

    # Create test data
    grade_data = [
        PrefJudgeData(
            query_id="topic-1",
            query_title="Test Query",
            query_background="Background",
            query_problem="Problem",
            run_id="run-A",
            run_id2="run-B",
            passage_1="Response A",
            passage_2="Response B",
            better_passage=1,
            confidence=0.9,
        ),
        PrefJudgeData(
            query_id="topic-1",
            query_title="Test Query",
            query_background="Background",
            query_problem="Problem",
            run_id="run-B",
            run_id2="run-C",
            passage_1="Response B",
            passage_2="Response C",
            better_passage=2,
            confidence=0.7,
        ),
    ]

    aggregates = {
        "run-A:topic-1": PrefAggregateResult(
            run_id="run-A",
            topic_id="topic-1",
            borda_score=3,
            win_frac=0.75,
            better_than=["run-B", "run-C"],
            worse_than=[],
        ),
        "run-B:topic-1": PrefAggregateResult(
            run_id="run-B",
            topic_id="topic-1",
            borda_score=1,
            win_frac=0.25,
            better_than=["run-C"],
            worse_than=["run-A"],
        ),
    }

    # Save
    checkpoint_path = tmp_path / "test.preferences.jsonl"
    save_preferences(grade_data, aggregates, checkpoint_path)

    # Verify file exists and has correct number of lines
    lines = checkpoint_path.read_text().strip().split("\n")
    assert len(lines) == 4  # 2 comparisons + 2 aggregates

    # Verify each line is valid JSON with _type
    for line in lines:
        record = json.loads(line)
        assert "_type" in record
        assert record["_type"] in ("comparison", "aggregate")

    # Load
    loaded_grade_data, loaded_aggregates = load_preferences(checkpoint_path)

    # Verify comparisons
    assert len(loaded_grade_data) == 2
    assert loaded_grade_data[0].query_id == "topic-1"
    assert loaded_grade_data[0].better_passage == 1
    assert loaded_grade_data[0].confidence == 0.9
    assert loaded_grade_data[1].better_passage == 2

    # Verify aggregates
    assert len(loaded_aggregates) == 2
    assert "run-A:topic-1" in loaded_aggregates
    assert loaded_aggregates["run-A:topic-1"].borda_score == 3
    assert loaded_aggregates["run-A:topic-1"].better_than == ["run-B", "run-C"]
    assert loaded_aggregates["run-B:topic-1"].worse_than == ["run-A"]


def test_preference_checkpoint_empty(tmp_path):
    """save_preferences handles empty data."""
    from judges.shared.nugget_judge_base import save_preferences, load_preferences

    checkpoint_path = tmp_path / "empty.preferences.jsonl"
    save_preferences([], {}, checkpoint_path)

    loaded_grade_data, loaded_aggregates = load_preferences(checkpoint_path)
    assert loaded_grade_data == []
    assert loaded_aggregates == {}


# =============================================================================
# chunk_by_query tests
# =============================================================================


def test_chunk_by_query_basic():
    """chunk_by_query splits items into chunks respecting per-query limit."""
    from judges.shared.nugget_judge_base import chunk_by_query
    from pydantic import BaseModel

    class Item(BaseModel):
        query_id: str
        winner_run_id: str
        value: int = 0

    items = [
        Item(query_id="t1", winner_run_id="r1", value=1),
        Item(query_id="t1", winner_run_id="r2", value=2),
        Item(query_id="t1", winner_run_id="r3", value=3),
        Item(query_id="t2", winner_run_id="r1", value=4),
        Item(query_id="t2", winner_run_id="r2", value=5),
    ]

    def sort_key(x, borda_scores):
        return x.value

    chunks = chunk_by_query(
        items,
        borda_scores={},
        nugget_gen_order="as_provided",
        sort_key_fn=sort_key,
        num_per_query=2,
    )

    # Should produce 2 chunks: first with 2 from t1 + 2 from t2, second with 1 from t1
    assert len(chunks) == 2
    # First chunk has at most 2 per topic
    t1_in_chunk0 = [i for i in chunks[0] if i.query_id == "t1"]
    t2_in_chunk0 = [i for i in chunks[0] if i.query_id == "t2"]
    assert len(t1_in_chunk0) <= 2
    assert len(t2_in_chunk0) <= 2


def test_chunk_by_query_empty():
    """chunk_by_query handles empty list."""
    from judges.shared.nugget_judge_base import chunk_by_query

    chunks = chunk_by_query([], {}, "as_provided", lambda x, b: 0)
    assert chunks == []


def test_chunk_by_query_max_pairs():
    """chunk_by_query respects max_pairs_considered."""
    from judges.shared.nugget_judge_base import chunk_by_query
    from pydantic import BaseModel

    class Item(BaseModel):
        query_id: str
        winner_run_id: str

    items = [Item(query_id="t1", winner_run_id=f"r{i}") for i in range(10)]

    chunks = chunk_by_query(
        items,
        borda_scores={},
        nugget_gen_order="as_provided",
        sort_key_fn=lambda x, b: 0,
        num_per_query=2,
        max_pairs_considered=3,
    )

    # Only 3 items should be processed total
    total_items = sum(len(c) for c in chunks)
    assert total_items == 3


def test_chunk_by_query_runner_up():
    """runner_up mode sweeps the sorted list: sweep 1 = each winner vs its
    runner-up (best loser), sweep 2 = each winner vs its 2nd-best loser, etc."""
    from judges.shared.nugget_judge_base import chunk_by_query
    from pydantic import BaseModel

    class Item(BaseModel):
        query_id: str
        winner_run_id: str
        loser_run_id: str

    # 4 systems ranked r1 > r2 > r3 > r4 by borda.
    borda = {"r1:t1": 4, "r2:t1": 3, "r3:t1": 2, "r4:t1": 1}

    # All borda-consistent pairs in arbitrary input order.
    items = [
        Item(query_id="t1", winner_run_id="r2", loser_run_id="r4"),
        Item(query_id="t1", winner_run_id="r1", loser_run_id="r4"),
        Item(query_id="t1", winner_run_id="r3", loser_run_id="r4"),
        Item(query_id="t1", winner_run_id="r1", loser_run_id="r2"),
        Item(query_id="t1", winner_run_id="r2", loser_run_id="r3"),
        Item(query_id="t1", winner_run_id="r1", loser_run_id="r3"),
    ]

    chunks = chunk_by_query(
        items,
        borda_scores=borda,
        nugget_gen_order="runner_up",
        sort_key_fn=lambda x, b: 0,  # unused for runner_up
        num_per_query=2,
    )

    # Flatten chunks back into sequence; chunk_by_query preserves within-topic order.
    flat = [p for chunk in chunks for p in chunk]
    pairs = [(p.winner_run_id, p.loser_run_id) for p in flat]

    assert pairs == [
        # sweep 1: each winner vs its runner-up
        ("r1", "r2"),
        ("r2", "r3"),
        ("r3", "r4"),
        # sweep 2: each winner vs its 2nd-best loser
        ("r1", "r3"),
        ("r2", "r4"),
        # sweep 3: r1's 3rd-best loser
        ("r1", "r4"),
    ]
