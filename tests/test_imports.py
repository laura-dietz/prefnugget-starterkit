"""Import smoke tests.

Verify that all judge classes and key utilities can be imported.
"""

import pytest


def test_prefnugget_judge_imports():
    """PrefNuggetJudge and its spec can be imported."""
    from judges.prefnugget import PrefNuggetJudge, PREFNUGGET_SPEC

    assert hasattr(PrefNuggetJudge, "create_nuggets")
    assert hasattr(PrefNuggetJudge, "judge")
    assert len(PREFNUGGET_SPEC.measures) == 4


def test_ground_nugget_judge_imports():
    """GroundNuggetJudge can be imported."""
    from judges.grounded import GroundNuggetJudge

    assert hasattr(GroundNuggetJudge, "create_nuggets")
    assert hasattr(GroundNuggetJudge, "judge")


def test_rubric_judge_imports():
    """RubricJudge can be imported."""
    from judges.queryonly import RubricJudge

    assert hasattr(RubricJudge, "create_nuggets")
    assert hasattr(RubricJudge, "judge")


def test_shared_pref_common_imports():
    """Shared pref_common utilities can be imported."""
    from judges.shared.pref_common import (
        PrefJudgment,
        PrefTiesJudgment,
        PrefJudgeData,
        PrefAggregateResult,
        prepare_prompts,
        compute_pref_aggregates,
        run_pref_judgment_batch,
        select_comparison_samples,
    )


def test_shared_rubric_common_imports():
    """Shared rubric_common utilities can be imported."""
    from judges.shared.rubric_common import (
        GradeNuggetAnswer,
        NuggetGradeData,
        NuggetAggregateResult,
        prepare_nugget_grade_data,
        prepare_nugget_grade_data_for_documents,
        compute_nugget_aggregates,
        compute_nugget_aggregates_for_documents,
        build_nugget_banks,
        collect_nugget_relevant_docs,
    )


def test_dspy_signatures_all_importable():
    """All DSPy signatures used across judges can be imported."""
    from judges.shared.pref_common import PrefJudgment, PrefTiesJudgment
    from judges.shared.rubric_common import GradeNuggetAnswer
    from judges.prefnugget.prefnugget_judge import (
        ExtractDifferentiatingNuggets,
        IterativeExtractDifferentiatingNuggets,
    )
    from judges.grounded.groundnugget_judge import GroundedIterativeNuggets
    from judges.queryonly.rubric_autojudge import (
        IterativeGenerateNuggetQuestionsReportRequest,
    )

    # All 7 signatures exist
    sigs = [
        PrefJudgment,
        PrefTiesJudgment,
        GradeNuggetAnswer,
        ExtractDifferentiatingNuggets,
        IterativeExtractDifferentiatingNuggets,
        GroundedIterativeNuggets,
        IterativeGenerateNuggetQuestionsReportRequest,
    ]
    for sig in sigs:
        assert sig.__doc__ is not None, f"{sig.__name__} missing __doc__"


def test_workflow_files_exist():
    """Workflow YAML files exist for all judges."""
    from pathlib import Path

    judges_dir = Path(__file__).parent.parent / "judges"
    assert (judges_dir / "prefnugget" / "workflow.yml").exists()
    assert (judges_dir / "grounded" / "workflow.yml").exists()
    assert (judges_dir / "queryonly" / "workflow.yml").exists()


def test_prefnugget_spec_measures():
    """PREFNUGGET_SPEC has expected measures."""
    from judges.prefnugget import PREFNUGGET_SPEC

    measure_names = [m.name for m in PREFNUGGET_SPEC.measures]
    assert "NUGGET_COVERAGE" in measure_names
    assert "AVG_GRADE" in measure_names
    assert "MAX_GRADE" in measure_names
    assert "COVERED_COUNT" in measure_names