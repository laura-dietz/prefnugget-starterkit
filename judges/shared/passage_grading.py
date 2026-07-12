"""
Extract addressed quotes from high-grade nugget passages.

Provides:
- ExtractAddressedQuote: DSPy signature for extracting verbatim quotes
- CLI for processing nugget-grades.jsonl and adding addressed_quote field
"""

import json
from pathlib import Path
from textwrap import dedent
from typing import List, Optional

import click
import dspy

from minima_llm import MinimaLlmConfig
from minima_llm.dspy_adapter import run_dspy_batch_generic

from .rubric_common import NuggetGradeData


# =============================================================================
# DSPy Signature
# =============================================================================


def _normalize(s: Optional[str]) -> str:
    """Normalize string for comparison."""
    if s is None:
        return ""
    return s.strip().lower()


class ExtractAddressedQuote(dspy.Signature):
    __doc__ = dedent(
        """
        Extract a contiguous verbatim quote from a passage that addresses a question.

        Given the question and passage, find a SINGLE CONTIGUOUS text span that
        best supports the answer to the question.

        CRITICAL REQUIREMENTS:
        - The quote MUST be a single contiguous span - one unbroken sequence of
          characters that appears exactly as-is in the passage
        - Do NOT combine sentences from different parts of the passage
        - Do NOT rearrange or reorder text
        - Do NOT include quotation marks around the extracted text
        - Copy the text character-for-character from the passage

        If multiple relevant sections exist, choose the SINGLE BEST contiguous
        span that most directly addresses the question. Do not try to combine them.

        - Emit **each field exactly once**, in the order shown.
        - Do NOT repeat headers. Do NOT include any field more than once.
        - If no relevant contiguous quote can be found, set extracted_quote to None.
        """
    )

    question: str = dspy.InputField(desc="The question to be answered")
    passage: str = dspy.InputField(desc="The passage that may contain the answer")

    extracted_quote: Optional[str] = dspy.OutputField(
        desc="Single contiguous text span copied exactly from the passage (no quotation marks, no rearranging)",
        default=None,
    )
    confidence: float = dspy.OutputField(
        desc="Confidence score from 0.0 to 1.0 indicating how well the quote addresses the question",
        default=0.0,
    )

    @classmethod
    def convert_output(
        cls, prediction: dspy.Prediction, data: NuggetGradeData
    ) -> None:
        """Convert DSPy Prediction output to NuggetGradeData.addressed_quote."""
        extracted = getattr(prediction, "extracted_quote", None)
        confidence = getattr(prediction, "confidence", 0.0) or 0.0

        # Normalize empty/none values
        if _normalize(extracted) in ["none", "", "null", "n/a"]:
            extracted = None
            confidence = 0.0

        # Only set if confidence is reasonable
        if extracted and confidence > 0.2:
            data.addressed_quote = extracted
        else:
            data.addressed_quote = None


# =============================================================================
# Processing Functions
# =============================================================================


def load_nugget_grades(path: Path) -> List[NuggetGradeData]:
    """Load nugget grades from JSONL file."""
    grades = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                grades.append(NuggetGradeData(**data))
    return grades


def save_nugget_grades(grades: List[NuggetGradeData], path: Path) -> None:
    """Save nugget grades to JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for grade in grades:
            f.write(grade.model_dump_json() + "\n")


# =============================================================================
# CLI
# =============================================================================


@click.command()
@click.option(
    "--input",
    "-i",
    "input_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Input nugget-grades.jsonl file",
)
@click.option(
    "--output",
    "-o",
    "output_path",
    required=True,
    type=click.Path(path_type=Path),
    help="Output nugget-grades.jsonl file with addressed_quote field",
)
@click.option(
    "--min-grade",
    type=int,
    default=4,
    help="Minimum grade to extract quotes for (default: 4)",
)
@click.option(
    "--min-passage-length",
    type=int,
    default=100,
    help="Minimum passage length to process (default: 100)",
)
def main(
    input_path: Path,
    output_path: Path,
    min_grade: int,
    min_passage_length: int,
) -> None:
    """Extract addressed quotes from high-grade nugget passages."""
    # Load LLM config
    config = MinimaLlmConfig.from_env()

    click.echo(f"Using LLM: {config.model}")

    # Load grades
    click.echo(f"Loading grades from {input_path}...")
    all_grades = load_nugget_grades(input_path)
    click.echo(f"Loaded {len(all_grades)} entries")

    # Filter to qualifying entries (high grade, sufficient passage length, no existing quote)
    to_process = [
        g for g in all_grades
        if g.grade >= min_grade
        and len(g.passage) >= min_passage_length
        and not g.addressed_quote
    ]
    click.echo(f"Processing {len(to_process)} entries with grade >= {min_grade}")

    if to_process:
        # Run batch extraction using run_dspy_batch_generic
        to_process = run_dspy_batch_generic(
            data=to_process,
            signature=ExtractAddressedQuote,
            converter=ExtractAddressedQuote.convert_output,
            llm_config=config,
        )

    # Save results (all grades, with updated addressed_quote for processed ones)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_nugget_grades(all_grades, output_path)
    click.echo(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()