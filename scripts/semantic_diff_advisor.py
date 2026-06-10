#!/usr/bin/env python3
"""
semantic_diff_advisor.py - advisory diff-pattern detector for matching/contract
edge risks.

Four recent review BLOCKERs shared one shape: a recognition surface changed and
the boundary case slipped (over-broad fold #1439, generic key over-capture
#1446, fail-open contract field #1453, stemmer/term-set asymmetry #1466). The
defect itself needs judgment, but the TRIGGER is mechanical. This tool detects
the trigger patterns in a diff and emits the matching adversarial question.

It never decides the change is wrong. It is advisory: exit code is 0 unless
--strict is passed. Judgment stays with the builder self-check, the reviewer,
and the live AI-reconciliation gate.

stdlib only, Python 3.9+.

Usage:
    python scripts/semantic_diff_advisor.py --base origin/main
    python scripts/semantic_diff_advisor.py --base origin/main --json
"""

import argparse
import ast
import difflib
import json
import re
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

PY_EXTS = {".py"}
JS_EXTS = {".js", ".jsx", ".ts", ".tsx", ".mjs"}

# Patterns -> the standing adversarial question each one should trigger.
QUESTIONS = {
    "RECOGNITION_SET_WIDENED": (
        "A recognition set gained members. Probe the GENERIC ones: does any "
        "added entry over-capture (SLA/metadata columns, auto-acks, API-auth "
        "vs user-login)? Ship a negative fixture per new entry. "
        "(class: #1446 first_response, #1439 auth fold)"
    ),
    "RECOGNITION_SET_ADDED": (
        "A new recognition set was introduced. Every member needs "
        "both-direction coverage: a fixture that should match and a boundary "
        "fixture that must NOT. (class: #1446/#1439)"
    ),
    "MATCHER_CHANGED": (
        "A compiled matcher changed. Probe BOTH directions: what does it now "
        "match that it should not, and what does it no longer match that it "
        "should? (class: #1432/#1437 HTML strip over/under-detection)"
    ),
    "DEFAULTED_CONTRACT_FIELD": (
        "A defaulting idiom appeared in projection/validation context. Do the "
        "sibling fields fail closed? A missing field must invalidate, not "
        "render a default. Add a field-deletion test. (class: #1453 fail-open)"
    ),
    "NORMALIZER_TERMSET_COUPLING": (
        "A normalizer changed in a module holding term sets. Every term's "
        "inflections must be recognized by the REAL matching path -- add or "
        "refresh the inflection-coverage invariant test. Checking that terms "
        "are fixed points is NOT enough; check recognition of inflected "
        "forms. (class: #1466 stemmer asymmetry)"
    ),
}

NORMALIZER_NAME_RE = re.compile(r"(token|stem|normal|fold|signal)", re.IGNORECASE)

# Defaulting idioms that make a new contract field fail open.
DEFAULT_IDIOM_RE = re.compile(
    r"(\?\?\s*(0|\"\"|''|null|\[\])"
    r"|\.get\([^()]*,\s*(0|\"\"|'')\)"
    r"|\bor\s+(0|\"\"|''))"
)
# Function context that marks projection / validation code.
CONTRACT_CTX_RE = re.compile(
    r"\b(?:function\s+\w*(?:project|parse|validate)\w*"
    r"|def\s+\w*(?:project|parse|validate|snapshot|payload)\w*"
    r"|(?:project|parse)\w*(?:Snapshot|Payload|Inspect)\w*)",
    re.IGNORECASE,
)
CONTEXT_WINDOW_LINES = 60
MIN_NEW_SET_MEMBERS = 3


@dataclass
class Finding:
    path: str
    code: str
    name: str
    lineno: int
    detail: str


# ---------------------------------------------------------------------------
# AST extraction (Python sources)
# ---------------------------------------------------------------------------

def _string_elts(node):
    """Return frozenset of string members if node is a literal collection of
    strings (set/tuple/list, dict of string keys, or set()/frozenset()/tuple()
    wrapping one). None when it is anything else."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
            and node.func.id in ("set", "frozenset", "tuple", "list") and node.args:
        return _string_elts(node.args[0])
    if isinstance(node, (ast.Set, ast.Tuple, ast.List)):
        vals = []
        for e in node.elts:
            if isinstance(e, ast.Constant) and isinstance(e.value, str):
                vals.append(e.value)
            else:
                return None
        return frozenset(vals)
    if isinstance(node, ast.Dict):
        keys = []
        for k in node.keys:
            if isinstance(k, ast.Constant) and isinstance(k.value, str):
                keys.append(k.value)
            else:
                return None
        return frozenset(keys)
    return None


def string_collections(tree):
    """name -> (members, lineno) for module-level constants that are
    collections of string literals."""
    out = {}
    for node in tree.body:
        target = None
        value = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            target, value = node.targets[0].id, node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) \
                and node.value is not None:
            target, value = node.target.id, node.value
        if target is None:
            continue
        members = _string_elts(value)
        if members is not None and members:
            out[target] = (members, node.lineno)
    return out


def regex_constants(tree):
    """name -> (pattern_source, lineno) for module-level NAME = re.compile(...)."""
    out = {}
    for node in tree.body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)):
            continue
        v = node.value
        if isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute) \
                and v.func.attr == "compile" \
                and isinstance(v.func.value, ast.Name) and v.func.value.id == "re":
            try:
                pattern = ast.unparse(v.args[0]) if v.args else ""
            except Exception:
                pattern = ""
            out[node.targets[0].id] = (pattern, node.lineno)
    return out


def normalizer_funcs(tree):
    """name -> unparsed body for module-level functions whose name suggests
    token/stem/normalize/fold work."""
    out = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and NORMALIZER_NAME_RE.search(node.name):
            try:
                out[node.name] = ast.unparse(node)
            except Exception:
                out[node.name] = node.name
    return out


# ---------------------------------------------------------------------------
# Detectors (pure: old/new source in, findings out)
# ---------------------------------------------------------------------------

def detect_python(old_src, new_src, path):
    findings = []
    try:
        new_tree = ast.parse(new_src)
    except SyntaxError:
        return findings
    old_tree = None
    if old_src:
        try:
            old_tree = ast.parse(old_src)
        except SyntaxError:
            old_tree = None

    new_sets = string_collections(new_tree)
    old_sets = string_collections(old_tree) if old_tree else {}

    for name, (members, lineno) in sorted(new_sets.items()):
        if name in old_sets:
            added = members - old_sets[name][0]
            if added:
                findings.append(Finding(
                    path, "RECOGNITION_SET_WIDENED", name, lineno,
                    "added: " + ", ".join(sorted(added)[:12])))
        elif old_tree is not None and len(members) >= MIN_NEW_SET_MEMBERS:
            findings.append(Finding(
                path, "RECOGNITION_SET_ADDED", name, lineno,
                "%d members" % len(members)))

    new_res = regex_constants(new_tree)
    old_res = regex_constants(old_tree) if old_tree else {}
    for name, (pattern, lineno) in sorted(new_res.items()):
        if name in old_res and old_res[name][0] != pattern:
            findings.append(Finding(
                path, "MATCHER_CHANGED", name, lineno, "pattern changed"))

    new_fns = normalizer_funcs(new_tree)
    old_fns = normalizer_funcs(old_tree) if old_tree else {}
    changed = sorted(n for n, body in new_fns.items() if old_fns.get(n) != body)
    if changed and new_sets:
        findings.append(Finding(
            path, "NORMALIZER_TERMSET_COUPLING", ", ".join(changed[:3]), 0,
            "term sets in module: " + ", ".join(sorted(new_sets)[:4])))

    findings.extend(detect_default_idiom(old_src, new_src, path))
    return findings


def detect_default_idiom(old_src, new_src, path):
    """Added lines with a defaulting idiom near projection/validation context.
    Works on any language (line-based)."""
    findings = []
    old_lines = (old_src or "").splitlines()
    new_lines = (new_src or "").splitlines()
    matcher = difflib.SequenceMatcher(a=old_lines, b=new_lines, autojunk=False)
    for tag, _i1, _i2, j1, j2 in matcher.get_opcodes():
        if tag not in ("insert", "replace"):
            continue
        for j in range(j1, j2):
            line = new_lines[j]
            if not DEFAULT_IDIOM_RE.search(line):
                continue
            lo = max(0, j - CONTEXT_WINDOW_LINES)
            context = "\n".join(new_lines[lo:j + 1])
            if CONTRACT_CTX_RE.search(context):
                findings.append(Finding(
                    path, "DEFAULTED_CONTRACT_FIELD", "", j + 1,
                    line.strip()[:100]))
    return findings


def detect(old_src, new_src, path):
    ext = Path(path).suffix.lower()
    if ext in PY_EXTS:
        return detect_python(old_src, new_src, path)
    if ext in JS_EXTS:
        return detect_default_idiom(old_src, new_src, path)
    return []


def is_test_path(path):
    p = Path(path)
    name = p.name.lower()
    if name.startswith("test_") or name.endswith("_test.py"):
        return True
    if ".test." in name or ".spec." in name:
        return True
    return any(part.lower() in ("tests", "test") for part in p.parts)


# ---------------------------------------------------------------------------
# Git layer
# ---------------------------------------------------------------------------

def _git(*args):
    proc = subprocess.run(["git", *args], capture_output=True, text=True)
    return proc.returncode, proc.stdout


def merge_base(base):
    code, out = _git("merge-base", base, "HEAD")
    return out.strip() if code == 0 else None


def changed_paths(mb):
    code, out = _git("diff", "--name-only", mb, "HEAD")
    if code != 0:
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def file_at(rev, path):
    code, out = _git("show", "%s:%s" % (rev, path))
    return out if code == 0 else ""


def read_worktree(path):
    try:
        return Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


def sweep_diff(base):
    mb = merge_base(base)
    if mb is None:
        print("semantic-diff-advisor: cannot resolve merge-base for %r; "
              "skipping (advisory)." % base)
        return []
    findings = []
    for path in changed_paths(mb):
        ext = Path(path).suffix.lower()
        if ext not in PY_EXTS | JS_EXTS:
            continue
        if is_test_path(path):
            continue
        new_src = read_worktree(path)
        if not new_src:
            continue  # deleted file
        old_src = file_at(mb, path)
        findings.extend(detect(old_src, new_src, path))
    return findings


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(findings):
    if not findings:
        print("semantic-diff-advisor: no matching/contract-edge patterns "
              "detected in this diff.")
        return
    print("semantic-diff-advisor: %d pattern(s) detected. These are "
          "QUESTIONS, not verdicts.\n" % len(findings))
    by_path = {}
    for f in findings:
        by_path.setdefault(f.path, []).append(f)
    asked = set()
    for path in sorted(by_path):
        print(path)
        for f in by_path[path]:
            loc = "L%d" % f.lineno if f.lineno else "file"
            label = (" %s" % f.name) if f.name else ""
            print("  [%s]%s %s -- %s" % (f.code, label, loc, f.detail))
        print()
    print("ask yourself, per pattern:")
    for f in findings:
        if f.code in asked:
            continue
        asked.add(f.code)
        print("- %s: %s" % (f.code, QUESTIONS[f.code]))


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Advisory matching/contract-edge pattern detector for a diff.")
    ap.add_argument("--base", default="origin/main",
                    help="base ref to diff against (default: origin/main)")
    ap.add_argument("--json", action="store_true", help="emit JSON")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 when patterns are found (default: advisory, exit 0)")
    args = ap.parse_args(argv)

    findings = sweep_diff(args.base)
    if args.json:
        print(json.dumps([asdict(f) for f in findings], indent=2))
    else:
        print_report(findings)
    if args.strict and findings:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
