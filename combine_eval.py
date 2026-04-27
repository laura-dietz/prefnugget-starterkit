#!/usr/bin/env python3
"""
Combine multiple eval.txt files (ir_measures format) into one.

Per (run, topic):
- AVG_GRADE      -> mean across files
- MAX_GRADE      -> max across files
- COVERED_COUNT  -> sum across files

Other measures are ignored.

Usage:
    python combine_eval.py --output combined.eval.txt eval1.txt eval2.txt [...]
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

from autojudge_base import Leaderboard
from autojudge_base.leaderboard.leaderboard import LeaderboardEntry

KEPT_MEASURES = ("AVG_GRADE", "MAX_GRADE", "COVERED_COUNT")
KEPT_SET = set(KEPT_MEASURES)


def warn_run_id_mismatches(per_file_runs: Dict[Path, Set[str]]) -> None:
    files = list(per_file_runs.keys())
    if len(files) < 2:
        return
    reference = per_file_runs[files[0]]
    for path in files[1:]:
        runs = per_file_runs[path]
        missing = reference - runs
        extra = runs - reference
        if missing or extra:
            print(
                f"warning: run_ids differ between {files[0].name} and {path.name}: "
                f"missing={sorted(missing)} extra={sorted(extra)}",
                file=sys.stderr,
            )


def combine(input_paths: List[Path]) -> Leaderboard:
    grouped: Dict[Tuple[str, str], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    per_file_runs: Dict[Path, Set[str]] = {}

    for path in input_paths:
        lb = Leaderboard.load(path, format="ir_measures")
        runs_in_file: Set[str] = set()
        for entry in lb.entries:
            runs_in_file.add(entry.run_id)
            for measure, raw_value in entry.values.items():
                if measure not in KEPT_SET:
                    continue
                try:
                    value = float(raw_value)
                except (TypeError, ValueError):
                    print(
                        f"warning: {path}: non-numeric value for "
                        f"{entry.run_id}/{entry.topic_id}/{measure}: {raw_value!r}",
                        file=sys.stderr,
                    )
                    continue
                grouped[(entry.run_id, entry.topic_id)][measure].append(value)
        per_file_runs[path] = runs_in_file

    warn_run_id_mismatches(per_file_runs)

    entries: List[LeaderboardEntry] = []
    for (run_id, topic_id), measure_values in grouped.items():
        values: Dict[str, float] = {}
        if "AVG_GRADE" in measure_values:
            vs = measure_values["AVG_GRADE"]
            values["AVG_GRADE"] = sum(vs) / len(vs)
        if "MAX_GRADE" in measure_values:
            values["MAX_GRADE"] = max(measure_values["MAX_GRADE"])
        if "COVERED_COUNT" in measure_values:
            values["COVERED_COUNT"] = sum(measure_values["COVERED_COUNT"])
        entries.append(LeaderboardEntry(run_id=run_id, topic_id=topic_id, values=values))

    entries.sort(key=lambda e: (e.run_id, e.topic_id))
    return Leaderboard(measures=KEPT_MEASURES, entries=tuple(entries))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output", required=True, help="output eval.txt path")
    parser.add_argument("inputs", nargs="+", help="input eval.txt files (ir_measures format)")
    args = parser.parse_args()

    input_paths = [Path(p) for p in args.inputs]
    for p in input_paths:
        if not p.exists():
            print(f"error: input file not found: {p}", file=sys.stderr)
            return 2

    combined = combine(input_paths)
    combined.write(Path(args.output), format="ir_measures")
    return 0


if __name__ == "__main__":
    sys.exit(main())