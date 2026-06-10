"""Unit tests for scripts/semantic_diff_advisor.py.

Each detector test replays one of the four real review BLOCKERs (minimized)
that motivated the tool: #1446 (recognition set widened by a generic key),
#1439 (fold map widened), #1466 (normalizer changed beside term sets),
#1453 (fail-open default on a contract field). A control case asserts the
advisor stays quiet on an unrelated change.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "semantic_diff_advisor.py"

SPEC = importlib.util.spec_from_file_location("semantic_diff_advisor", SCRIPT)
assert SPEC is not None
assert SPEC.loader is not None
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def codes(findings):
    return {f.code for f in findings}


def test_recognition_set_widened_fires_on_the_1446_replay() -> None:
    old = (
        '_RESOLUTION_TEXT_KEYS = (\n'
        '    "resolution_text",\n'
        '    "agent_reply",\n'
        ')\n'
    )
    new = (
        '_RESOLUTION_TEXT_KEYS = (\n'
        '    "resolution_text",\n'
        '    "agent_reply",\n'
        '    "first_response",\n'
        '    "last_response",\n'
        ')\n'
    )
    findings = MOD.detect_python(old, new, "pkg/support_ticket_input_package.py")
    widened = [f for f in findings if f.code == "RECOGNITION_SET_WIDENED"]
    assert len(widened) == 1
    assert widened[0].name == "_RESOLUTION_TEXT_KEYS"
    assert "first_response" in widened[0].detail
    assert "last_response" in widened[0].detail


def test_fold_map_widened_fires_on_the_1439_replay() -> None:
    old = (
        '_TOKEN_FOLDS = {\n'
        '    "billed": "billing",\n'
        '}\n'
    )
    new = (
        '_TOKEN_FOLDS = {\n'
        '    "billed": "billing",\n'
        '    "auth": "login",\n'
        '    "authentication": "login",\n'
        '}\n'
    )
    findings = MOD.detect_python(old, new, "pkg/support_ticket_clustering.py")
    widened = [f for f in findings if f.code == "RECOGNITION_SET_WIDENED"]
    assert len(widened) == 1
    assert widened[0].name == "_TOKEN_FOLDS"
    assert "auth" in widened[0].detail


def test_normalizer_termset_coupling_fires_on_the_1466_replay() -> None:
    old = (
        '_RESOLUTION_ACTION_TERMS = {"enable", "reset", "update"}\n'
        'def _resolution_signal_token(token):\n'
        '    return token\n'
    )
    new = (
        '_RESOLUTION_ACTION_TERMS = {"enable", "reset", "update"}\n'
        'def _resolution_signal_token(token):\n'
        '    if len(token) > 4 and token.endswith("ed"):\n'
        '        return token[:-2]\n'
        '    return token\n'
    )
    findings = MOD.detect_python(old, new, "pkg/ticket_faq_markdown.py")
    coupled = [f for f in findings if f.code == "NORMALIZER_TERMSET_COUPLING"]
    assert len(coupled) == 1
    assert "_resolution_signal_token" in coupled[0].name
    assert "_RESOLUTION_ACTION_TERMS" in coupled[0].detail


def test_defaulted_contract_field_fires_on_the_1453_replay() -> None:
    old = (
        'function projectSnapshot(snapshot) {\n'
        '  const generated = finiteNumber(snapshot.summary.generated);\n'
        '  return { generated };\n'
        '}\n'
    )
    new = (
        'function projectSnapshot(snapshot) {\n'
        '  const generated = finiteNumber(snapshot.summary.generated);\n'
        '  const repeatTicketCount = finiteNumber(snapshot.summary.repeat_ticket_count) ?? 0;\n'
        '  return { generated, repeat_ticket_count: repeatTicketCount };\n'
        '}\n'
    )
    findings = MOD.detect(old, new, "portfolio-ui/api/content-ops/deflection/atlas-report.js")
    defaulted = [f for f in findings if f.code == "DEFAULTED_CONTRACT_FIELD"]
    assert len(defaulted) == 1
    assert "?? 0" in defaulted[0].detail


def test_matcher_changed_fires_on_a_regex_pattern_change() -> None:
    old = '_HTML_SIGNAL_RE = re.compile(r"</?(?:p|div)\\b")\n'
    new = '_HTML_SIGNAL_RE = re.compile(r"</?(?:a|b|p|div)\\b")\n'
    findings = MOD.detect_python("import re\n" + old, "import re\n" + new,
                                 "pkg/support_ticket_clustering.py")
    changed = [f for f in findings if f.code == "MATCHER_CHANGED"]
    assert len(changed) == 1
    assert changed[0].name == "_HTML_SIGNAL_RE"


def test_quiet_on_unrelated_changes() -> None:
    old = (
        'def helper(value):\n'
        '    return value + 1\n'
    )
    new = (
        'def helper(value):\n'
        '    return value + 2\n'
    )
    assert MOD.detect_python(old, new, "pkg/anything.py") == []


def test_quiet_when_set_unchanged_and_defaults_preexisting() -> None:
    src = (
        '_KEYS = ("a", "b", "c")\n'
        'function = None\n'
    )
    assert MOD.detect_python(src, src, "pkg/static.py") == []
    js = (
        'function parsePayload(x) {\n'
        '  const n = x.count ?? 0;\n'
        '  return n;\n'
        '}\n'
    )
    # unchanged file: the defaulting idiom is pre-existing, not added
    assert MOD.detect(js, js, "ui/api/thing.js") == []


def test_test_paths_are_excluded_by_the_path_filter() -> None:
    assert MOD.is_test_path("tests/test_extracted_support_ticket_input_package.py")
    assert MOD.is_test_path("portfolio-ui/scripts/faq-deflection-result-page.test.mjs")
    assert not MOD.is_test_path("extracted_content_pipeline/support_ticket_clustering.py")
