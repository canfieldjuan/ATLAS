#!/usr/bin/env python3
"""Run maturity_sweep.py against an explicit list of Python files."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import maturity_sweep


def _iter_python_files(paths):
    seen = set()
    for raw in paths:
        path = Path(raw)
        if not path.exists() or path.suffix != ".py":
            continue
        if maturity_sweep.is_test_path(path):
            continue
        key = path.as_posix()
        if key in seen:
            continue
        seen.add(key)
        yield path


def sweep_files(paths, tests_root):
    test_sources, all_test_text = maturity_sweep.index_tests(Path(tests_root))
    results = []
    for path in _iter_python_files(paths):
        result = maturity_sweep.analyze_file(path, test_sources, all_test_text)
        result.path = Path(result.path).as_posix()
        results.append(result)
    results.sort(key=lambda result: result.score, reverse=True)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Rank explicit Python files by production-fragility."
    )
    parser.add_argument("paths", nargs="+", help="Python files to sweep")
    parser.add_argument(
        "--tests-root",
        default="tests",
        help="directory to index tests from",
    )
    parser.add_argument("--top", type=int, default=20, help="how many to list")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    parser.add_argument(
        "--min-score",
        type=int,
        default=None,
        help="exit nonzero if any file scores at/above this",
    )
    parser.add_argument("--baseline", default=None, help="baseline JSON for ratchet mode")
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="write/refresh the baseline from the current sweep and exit 0",
    )
    parser.add_argument(
        "--sensitive-glob",
        action="append",
        default=[],
        help="glob for sensitive paths where new swallowed/bare except fails",
    )
    args = parser.parse_args(argv)

    results = sweep_files(args.paths, args.tests_root)

    if args.update_baseline:
        if not args.baseline:
            raise SystemExit("--update-baseline requires --baseline")
        maturity_sweep.write_baseline(args.baseline, results)
        print("wrote maturity sweep baseline: %s" % args.baseline)
        return 0

    baseline = maturity_sweep.load_baseline(args.baseline) if args.baseline else None
    ratchet = None
    if baseline is not None:
        ratchet = maturity_sweep.ratchet_failures(
            results,
            baseline,
            min_score=args.min_score,
            sensitive_globs=tuple(args.sensitive_glob),
        )

    if args.json:
        payload = [
            {
                "path": result.path,
                "score": result.score,
                "counts": result.counts,
                "findings": [asdict(finding) for finding in result.findings],
            }
            for result in results
            if result.score > 0
        ]
        print(json.dumps(payload, indent=2))
        if ratchet is not None:
            return 1 if ratchet else 0
        if args.min_score is not None:
            return 1 if any(result.score >= args.min_score for result in results) else 0
        return 0

    return maturity_sweep.print_report(
        results,
        args.top,
        args.min_score,
        ratchet=ratchet,
        baseline_path=args.baseline,
    )


if __name__ == "__main__":
    sys.exit(main())
