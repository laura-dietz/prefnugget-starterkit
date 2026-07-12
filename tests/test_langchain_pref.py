"""Unit tests for the LangChainPrefJudge helpers (no LLM calls)."""

from judges.langchain_pref.langchain_judge import (
    PrefPool,
    parse_better,
    parse_grade,
    parse_questions,
)


def test_parse_better():
    assert parse_better("1") == 1
    assert parse_better("Passage 2 is better") == 2
    assert parse_better("") is None
    assert parse_better("neither") is None


def test_parse_grade():
    assert parse_grade("5") == 5
    assert parse_grade("Grade: 3") == 3
    assert parse_grade("no digit here") == 0
    assert parse_grade("") == 0


def test_parse_questions_tolerates_prose():
    text = 'Here are the questions: ["Capital of USA?", "Process to cook steel?"] hope this helps'
    assert parse_questions(text) == ["Capital of USA?", "Process to cook steel?"]
    assert parse_questions("[]") == []
    assert parse_questions("no json at all") == []


def test_pref_pool_offsets_dedupe_and_cap():
    pool = PrefPool(run_ids=["r1", "r2", "r3", "r4"])
    first = pool.pairs_at_offset(1)
    second = pool.pairs_at_offset(2)
    # 4 runs -> C(4,2)=6 unordered pairs, split 4 + 2 across offsets 1 and 2
    assert len(first) == 4 and len(second) == 2
    assert len({tuple(sorted(p)) for p in first + second}) == 6
    # offset 3 mirrors offset 1 -> no new pairs, and growth caps at n//2
    pool.judged_offsets = 2
    assert pool.pairs_at_offset(3) == []
    assert not pool.can_grow()


def test_borda_from_consistent_pairs():
    pool = PrefPool(run_ids=["a", "b", "c"])
    pool.consistent_pairs = [("a", "b"), ("a", "c"), ("b", "c")]
    scores = pool.borda()
    assert scores == {"a": 2, "b": 0, "c": -2}
