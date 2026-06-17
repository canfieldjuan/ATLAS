#!/usr/bin/env python3
"""
maturity_sweep.py - rank files by how likely they are to be brittle,
happy-path code that will not survive production.

The point of this tool is what it does NOT do. It never reads a diff, a PR
description, a plan doc, or commit intent. It looks only at the intrinsic
properties of each file and its tests. A reviewer that never sees the dev's
framing cannot drift into it. The output is an agenda for an adversarial
pass, ordered worst-first.

It does not decide a file is bad. It decides a file is suspect and tells you
exactly where to aim. Judgment stays with you (or a fresh reviewer session).

stdlib only, Python 3.9+.

Usage:
    python maturity_sweep.py path/to/lane
    python maturity_sweep.py path/to/lane --top 15
    python maturity_sweep.py path/to/lane --json > sweep.json
    python maturity_sweep.py path/to/lane --min-score 8   # CI gate
"""

import argparse
import ast
import fnmatch
import json
import os
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Tunable config. Bend these to your pipeline. For the Deflection Report lane
# the ingestion/clustering files are the ones you care about, so the boundary
# set below leans toward CSV/file/encoding entry points.
# ---------------------------------------------------------------------------

WEIGHTS = {
    "NO_TEST_FILE": 6,          # module has functions/classes and no test touches it
    "SWALLOWED_EXCEPT": 5,      # except: pass / return None / return [] - silent degradation
    "HAPPY_PATH_TESTS": 4,      # tests exist but almost none exercise failure
    "UNGUARDED_INPUT": 4,       # touches external input, no try, no raise in the function
    "BARE_EXCEPT": 4,           # except: with no type
    "NO_RAISES_TESTS": 3,       # test file never asserts that anything raises
    "ASSERT_AS_VALIDATION": 3,  # assert used as a guard in non-test code (vanishes under -O)
    "UNGUARDED_INDEX": 2,       # x[0] / split()[n] with no length check in the function
    "MUTABLE_DEFAULT": 2,       # def f(x=[])
    "HEURISTIC_COMMENT": 1,     # TODO/HACK/probably/best effort/brittle...
    "WEAK_CONTRACT": 1,         # public def with zero type annotations
    "PARSE_ERROR": 8,           # file does not parse
}

# Per-code caps so one noisy category cannot dominate a file's score.
CAPS = {
    "ASSERT_AS_VALIDATION": 3,
    "UNGUARDED_INDEX": 3,
    "HEURISTIC_COMMENT": 4,
    "WEAK_CONTRACT": 3,
    "MUTABLE_DEFAULT": 2,
}

# Calls that strongly indicate reading external / untrusted input. Kept tight
# on purpose - get/post/parse/decode are too common to be signal.
BOUNDARY_NAMES = {
    "open", "loads", "load", "reader", "DictReader",
    "read_csv", "read_excel", "strptime", "fromisoformat",
    "urlopen", "recv",
}

HEURISTIC_RE = re.compile(
    r"\b(todo|fixme|hack|xxx|for now|temporary|temp\b|best[ -]?effort|"
    r"good enough|probably|hopefully|assume|guess|kludge|workaround|"
    r"brittle|fragile|happy path|naive|quick fix)\b",
    re.IGNORECASE,
)

# test-name fragments that indicate the test is trying to break something
NEGATIVE_TEST_HINTS = (
    "invalid", "malformed", "empty", "missing", "bad", "error", "fail",
    "raises", "none", "null", "corrupt", "edge", "boundary", "duplicate",
    "garbage", "unicode", "encoding", "oversize", "too_", "partial",
    "truncat", "blank", "whitespace", "nan", "negative", "overflow",
)

SKIP_DIRS = {
    "__pycache__", ".venv", "venv", "env", "node_modules", ".git",
    "build", "dist", ".mypy_cache", ".pytest_cache", "migrations",
}

TEST_NAME_RE = re.compile(r"(^test_.+\.py$)|(.+_test\.py$)")
SENSITIVE_ZERO_TOLERANCE = ("BARE_EXCEPT", "SWALLOWED_EXCEPT")


@dataclass
class Finding:
    code: str
    lineno: int
    detail: str
    weight: int = 0

    def __post_init__(self):
        self.weight = WEIGHTS.get(self.code, 0)


@dataclass
class FileResult:
    path: str
    score: int = 0
    findings: list = field(default_factory=list)
    counts: dict = field(default_factory=dict)


@dataclass(frozen=True)
class GateFailure:
    path: str
    reason: str
    detail: str


# ---------------------------------------------------------------------------
# AST analysis
# ---------------------------------------------------------------------------

class Analyzer(ast.NodeVisitor):
    """Collects findings from a single module. Tracks whether we are inside a
    try block and whether the enclosing function ever raises, so that touching
    external input without either guard can be flagged."""

    def __init__(self, is_test):
        self.is_test = is_test
        self.findings = []
        self.try_depth = 0
        self.func_has_raise = []  # stack, one bool per enclosing function

    def add(self, code, lineno, detail):
        self.findings.append(Finding(code, lineno, detail))

    # -- functions -----------------------------------------------------------
    def visit_FunctionDef(self, node):
        self._check_mutable_defaults(node)
        self._check_contract(node)
        has_raise = any(isinstance(n, ast.Raise) for n in ast.walk(node))
        self.func_has_raise.append(has_raise)
        saved = self.try_depth
        self.try_depth = 0
        self.generic_visit(node)
        self.try_depth = saved
        self.func_has_raise.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def _check_mutable_defaults(self, node):
        defaults = list(node.args.defaults) + list(node.args.kw_defaults)
        for d in defaults:
            if isinstance(d, (ast.List, ast.Dict, ast.Set)):
                self.add("MUTABLE_DEFAULT", node.lineno,
                         "mutable default argument in %s" % node.name)
                break
            if isinstance(d, ast.Call) and isinstance(d.func, ast.Name) \
                    and d.func.id in ("list", "dict", "set"):
                self.add("MUTABLE_DEFAULT", node.lineno,
                         "mutable default argument in %s" % node.name)
                break

    def _check_contract(self, node):
        if node.name.startswith("_") or self.is_test:
            return
        annotated = node.returns is not None or any(
            a.annotation is not None for a in node.args.args
        )
        if not annotated and (node.args.args or node.returns is None):
            # only flag if it takes args or is a real public entry point
            if node.args.args:
                self.add("WEAK_CONTRACT", node.lineno,
                         "public def %s has no type annotations" % node.name)

    # -- try / except --------------------------------------------------------
    def visit_Try(self, node):
        self.try_depth += 1
        for n in node.body:
            self.visit(n)
        self.try_depth -= 1
        for h in node.handlers:
            self.visit(h)
        for n in node.orelse + node.finalbody:
            self.visit(n)

    def visit_ExceptHandler(self, node):
        if node.type is None:
            self.add("BARE_EXCEPT", node.lineno, "bare except")
        if self._is_swallowed(node.body):
            self.add("SWALLOWED_EXCEPT", node.lineno,
                     "exception swallowed (no re-raise, no handling)")
        self.generic_visit(node)

    @staticmethod
    def _is_swallowed(body):
        if len(body) != 1:
            return False
        stmt = body[0]
        if isinstance(stmt, (ast.Pass, ast.Continue)):
            return True
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) \
                and stmt.value.value is Ellipsis:
            return True
        if isinstance(stmt, ast.Return):
            v = stmt.value
            if v is None:
                return True
            if isinstance(v, ast.Constant) and v.value in (None, "", 0, False):
                return True
            if isinstance(v, (ast.List, ast.Dict, ast.Set, ast.Tuple)) and not getattr(v, "elts", None) \
                    and not getattr(v, "keys", None):
                return True
        return False

    # -- assert as validation ------------------------------------------------
    def visit_Assert(self, node):
        if not self.is_test:
            self.add("ASSERT_AS_VALIDATION", node.lineno,
                     "assert used as a guard (removed under python -O)")
        self.generic_visit(node)

    # -- boundary calls ------------------------------------------------------
    def visit_Call(self, node):
        name = self._call_name(node.func)
        if name in BOUNDARY_NAMES:
            in_try = self.try_depth > 0
            enclosing_raises = bool(self.func_has_raise) and self.func_has_raise[-1]
            if not in_try and not enclosing_raises:
                self.add("UNGUARDED_INPUT", node.lineno,
                         "reads external input via %s() with no try and no raise"
                         % name)
        self.generic_visit(node)

    @staticmethod
    def _call_name(func):
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            return func.attr
        return None

    # -- unguarded indexing (heuristic, low weight) --------------------------
    def visit_Subscript(self, node):
        idx = node.slice
        const_int = isinstance(idx, ast.Constant) and isinstance(idx.value, int)
        on_split = isinstance(node.value, ast.Call) and \
            self._call_name(node.value.func) == "split"
        if const_int or on_split:
            self.add("UNGUARDED_INDEX", node.lineno,
                     "fixed-index access; verify the sequence can be shorter")
        self.generic_visit(node)


# ---------------------------------------------------------------------------
# test scoring
# ---------------------------------------------------------------------------

def index_tests(root):
    """Return (test_sources, all_test_text). test_sources maps a test-file stem
    to its source; all_test_text is every test file concatenated, for the
    'is this module mentioned anywhere in tests' check."""
    sources = {}
    blobs = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if fn.endswith(".py") and (TEST_NAME_RE.match(fn) or "test" in Path(dirpath).name.lower()):
                p = Path(dirpath) / fn
                src = read_text(p)
                sources[p.stem] = src
                blobs.append(src)
    return sources, "\n".join(blobs)


def matching_test_sources(module_stem, test_sources):
    """All test sources whose file stem contains the module stem as a
    _-delimited segment run. Handles prefixed conventions like
    test_extracted_<module> and test_content_ops_<module>, and unions the
    multiple test files a module commonly has. The previous exact-stem-only
    match left HAPPY_PATH_TESTS/NO_RAISES_TESTS dark for prefix-named tests."""
    seg_re = re.compile(r"(^|_)%s(_|$)" % re.escape(module_stem.lower()))
    return [src for stem, src in sorted(test_sources.items())
            if seg_re.search(stem.lower())]


def score_tests(module_stem, test_sources, all_test_text):
    """Returns a list of findings about test coverage quality for one module."""
    findings = []
    matched = matching_test_sources(module_stem, test_sources)
    src = "\n".join(matched) if matched else None

    mentioned = re.search(r"\b%s\b" % re.escape(module_stem), all_test_text)

    if src is None:
        if not mentioned:
            findings.append(Finding("NO_TEST_FILE", 0,
                                    "no test file and module not referenced by any test"))
        return findings

    test_defs = re.findall(r"def\s+(test_\w+)", src)
    total = len(test_defs)
    if total == 0:
        return findings

    negatives = sum(
        1 for name in test_defs
        if any(h in name.lower() for h in NEGATIVE_TEST_HINTS)
    )
    raises = bool(re.search(r"pytest\.raises|assertRaises|with raises", src))

    ratio = negatives / total
    if ratio < 0.15:
        findings.append(Finding("HAPPY_PATH_TESTS", 0,
                                "%d tests, %d exercise failure paths (%.0f%%)"
                                % (total, negatives, ratio * 100)))
    if not raises and total >= 3:
        findings.append(Finding("NO_RAISES_TESTS", 0,
                                "no test asserts that anything raises"))
    return findings


# ---------------------------------------------------------------------------
# driving
# ---------------------------------------------------------------------------

def read_text(path):
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1", errors="replace")
    except OSError:
        return ""


def scan_comments(source):
    out = []
    for i, line in enumerate(source.splitlines(), start=1):
        m = HEURISTIC_RE.search(line)
        if m:
            out.append(Finding("HEURISTIC_COMMENT", i,
                               "language of a known shortcut: '%s'" % m.group(0).strip()))
    return out


def is_test_path(path):
    if TEST_NAME_RE.match(path.name):
        return True
    return any(part.lower() in ("tests", "test") for part in path.parts)


def looks_substantive(tree):
    """True if the module defines functions or classes worth testing."""
    return any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
               for n in tree.body)


def apply_caps(findings):
    """Sum weights with per-code caps so one category cannot bury the rest."""
    seen = {}
    total = 0
    counts = {}
    for f in findings:
        counts[f.code] = counts.get(f.code, 0) + 1
        cap = CAPS.get(f.code)
        if cap is not None:
            used = seen.get(f.code, 0)
            if used >= cap:
                continue
            seen[f.code] = used + 1
        total += f.weight
    return total, counts


def analyze_file(path, test_sources, all_test_text):
    source = read_text(path)
    result = FileResult(path=str(path))
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as e:
        result.findings.append(Finding("PARSE_ERROR", e.lineno or 0,
                                       "does not parse: %s" % e.msg))
        result.score, result.counts = apply_caps(result.findings)
        return result

    analyzer = Analyzer(is_test=False)
    analyzer.visit(tree)
    result.findings.extend(analyzer.findings)
    result.findings.extend(scan_comments(source))

    if looks_substantive(tree) and path.stem != "__init__":
        result.findings.extend(score_tests(path.stem, test_sources, all_test_text))

    result.score, result.counts = apply_caps(result.findings)
    return result


def sweep(root, tests_root=None):
    root = Path(root)
    # Tests may live outside the swept lane (e.g. a repo-level tests/ dir).
    # Index them from tests_root when given so coverage signals are real.
    test_sources, all_test_text = index_tests(Path(tests_root) if tests_root else root)
    results = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            path = Path(dirpath) / fn
            if is_test_path(path):
                continue
            results.append(analyze_file(path, test_sources, all_test_text))
    results.sort(key=lambda r: r.score, reverse=True)
    return results


# ---------------------------------------------------------------------------
# baseline ratchet gate
# ---------------------------------------------------------------------------

def baseline_entry(result):
    return {
        "score": int(result.score),
        "counts": {code: int(count) for code, count in sorted(result.counts.items())},
    }


def baseline_payload(results):
    return {
        result.path: baseline_entry(result)
        for result in sorted(results, key=lambda item: item.path)
        if result.score > 0
    }


def write_baseline(path, results):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(baseline_payload(results), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_baseline(path):
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit("baseline not found: %s" % path)
    except json.JSONDecodeError as exc:
        raise SystemExit("baseline is not valid JSON: %s: %s" % (path, exc))
    if not isinstance(raw, dict):
        raise SystemExit("baseline must be a JSON object: %s" % path)
    return raw


def _baseline_score(entry):
    if not isinstance(entry, dict):
        return 0
    try:
        return int(entry.get("score", 0))
    except (TypeError, ValueError):
        return 0


def _baseline_counts(entry):
    if not isinstance(entry, dict):
        return {}
    counts = entry.get("counts", {})
    if not isinstance(counts, dict):
        return {}
    out = {}
    for code, count in counts.items():
        try:
            out[str(code)] = int(count)
        except (TypeError, ValueError):
            out[str(code)] = 0
    return out


def _matches_sensitive(path, patterns):
    return any(
        fnmatch.fnmatch(path, pattern) or Path(path).match(pattern)
        for pattern in patterns
    )


def ratchet_failures(results, baseline, min_score=None, sensitive_globs=()):
    by_path = {result.path: result for result in results}
    failures = []
    for path, result in sorted(by_path.items()):
        entry = baseline.get(path)
        baseline_score = _baseline_score(entry)
        baseline_counts = _baseline_counts(entry)
        if entry is not None and result.score > baseline_score:
            failures.append(GateFailure(
                path=path,
                reason="score increased",
                detail="%d -> %d" % (baseline_score, result.score),
            ))
        if entry is None and min_score is not None and result.score >= min_score:
            failures.append(GateFailure(
                path=path,
                reason="new file at or above min-score",
                detail="score %d >= %d" % (result.score, min_score),
            ))
        if _matches_sensitive(path, sensitive_globs):
            for code in SENSITIVE_ZERO_TOLERANCE:
                old = baseline_counts.get(code, 0)
                new = int(result.counts.get(code, 0))
                if new > old:
                    failures.append(GateFailure(
                        path=path,
                        reason="new sensitive-path %s" % code,
                        detail="%d -> %d" % (old, new),
                    ))
    return failures


def print_ratchet_failures(failures, baseline_path):
    if not failures:
        print("ratchet gate passed: no new brittleness above baseline")
        return 0
    print("RATCHET GATE FAILED: %d file(s) introduced new brittleness" % len(failures))
    for failure in failures:
        print("- %s: %s (%s)" % (failure.path, failure.reason, failure.detail))
    print()
    print("To accept this intentionally, rerun with:")
    print("  python scripts/maturity_sweep.py <lane> --baseline %s --update-baseline" % baseline_path)
    return 1


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def print_report(results, top, min_score, ratchet=None, baseline_path=None):
    flagged = [r for r in results if r.score > 0]
    print("maturity sweep: %d files scanned, %d flagged\n" % (len(results), len(flagged)))
    print("score  file")
    print("-----  ----")
    for r in flagged[:top]:
        print("%5d  %s" % (r.score, r.path))

    print("\nworst offenders, with where to aim:\n")
    for r in flagged[:min(top, 8)]:
        print("[%d] %s" % (r.score, r.path))
        summary = ", ".join("%s x%d" % (c, n) for c, n in
                            sorted(r.counts.items(), key=lambda kv: -WEIGHTS.get(kv[0], 0)))
        print("     %s" % summary)
        ranked = sorted(r.findings, key=lambda f: -f.weight)
        for f in ranked[:6]:
            loc = ("L%d" % f.lineno) if f.lineno else "file"
            print("     %-5s %s" % (loc, f.detail))
        print()

    if ratchet is not None:
        return print_ratchet_failures(ratchet, baseline_path)

    if min_score is not None:
        over = [r for r in results if r.score >= min_score]
        if over:
            print("GATE FAILED: %d file(s) at or above min-score %d" % (len(over), min_score))
            return 1
        print("gate passed: nothing at or above min-score %d" % min_score)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="Rank files by production-fragility.")
    ap.add_argument("path", help="directory to sweep (point at the lane)")
    ap.add_argument("--tests-root", default=None,
                    help="directory to index tests from (default: the swept path); "
                         "point at the repo tests dir when source and tests are separate")
    ap.add_argument("--top", type=int, default=20, help="how many to list")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of text")
    ap.add_argument("--min-score", type=int, default=None,
                    help="exit nonzero if any file scores at/above this (CI gate)")
    ap.add_argument("--baseline", default=None,
                    help="baseline JSON for ratchet mode")
    ap.add_argument("--update-baseline", action="store_true",
                    help="write/refresh the baseline from the current sweep and exit 0")
    ap.add_argument("--sensitive-glob", action="append", default=[],
                    help="glob for sensitive paths where new swallowed/bare except fails")
    args = ap.parse_args(argv)

    results = sweep(args.path, args.tests_root)

    if args.update_baseline:
        if not args.baseline:
            raise SystemExit("--update-baseline requires --baseline")
        write_baseline(args.baseline, results)
        print("wrote maturity sweep baseline: %s" % args.baseline)
        return 0

    baseline = load_baseline(args.baseline) if args.baseline else None
    ratchet = None
    if baseline is not None:
        ratchet = ratchet_failures(
            results,
            baseline,
            min_score=args.min_score,
            sensitive_globs=tuple(args.sensitive_glob),
        )

    if args.json:
        payload = [
            {"path": r.path, "score": r.score, "counts": r.counts,
             "findings": [asdict(f) for f in r.findings]}
            for r in results if r.score > 0
        ]
        print(json.dumps(payload, indent=2))
        if ratchet is not None:
            return 1 if ratchet else 0
        if args.min_score is not None:
            return 1 if any(r.score >= args.min_score for r in results) else 0
        return 0

    return print_report(
        results,
        args.top,
        args.min_score,
        ratchet=ratchet,
        baseline_path=args.baseline,
    )


if __name__ == "__main__":
    sys.exit(main())
