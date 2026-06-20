from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_gitleaks_baseline_rotation.py"


def load_checker():
    name = "check_gitleaks_baseline_rotation"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def _changed(*paths: str) -> set[str]:
    return set(paths)


def test_allows_unchanged_baseline_without_label() -> None:
    checker = load_checker()
    decision = checker.evaluate_baseline_rotation(
        _changed("app/api/foo.py"),
        labels=set(),
        base_has_baseline=True,
    )

    assert decision.allowed is True
    assert "unchanged" in decision.reason


def test_rejects_baseline_change_without_rotation_label() -> None:
    checker = load_checker()
    decision = checker.evaluate_baseline_rotation(
        _changed(checker.BASELINE_PATH),
        labels=set(),
        base_has_baseline=True,
    )

    assert decision.allowed is False
    assert "security-rotation" in decision.reason


def test_allows_labeled_rotation_with_only_baseline_docs_and_plan() -> None:
    checker = load_checker()
    decision = checker.evaluate_baseline_rotation(
        _changed(
            checker.BASELINE_PATH,
            "docs/SECURITY_GUARDRAILS.md",
            "HARDENING.md",
            "plans/PR-Gitleaks-Baseline-Rotation.md",
        ),
        labels={"security-rotation"},
        base_has_baseline=True,
        base_fingerprints={"base-a", "base-b"},
        candidate_fingerprints={"base-a", "base-b", "new-c"},
    )

    assert decision.allowed is True
    assert "accepted" in decision.reason


def test_rejects_labeled_rotation_that_changes_product_code() -> None:
    checker = load_checker()
    decision = checker.evaluate_baseline_rotation(
        _changed(
            checker.BASELINE_PATH,
            "app/api/graphrag/upload/route.ts",
            "plans/PR-Gitleaks-Baseline-Rotation.md",
        ),
        labels={"security-rotation"},
        base_has_baseline=True,
        base_fingerprints={"base-a"},
        candidate_fingerprints={"base-a"},
    )

    assert decision.allowed is False
    assert decision.disallowed_paths == ("app/api/graphrag/upload/route.ts",)


def test_rejects_labeled_rotation_that_drops_existing_fingerprint() -> None:
    checker = load_checker()
    decision = checker.evaluate_baseline_rotation(
        _changed(checker.BASELINE_PATH, "plans/PR-Gitleaks-Baseline-Rotation.md"),
        labels={"security-rotation"},
        base_has_baseline=True,
        base_fingerprints={"kept-fingerprint", "dropped-fingerprint"},
        candidate_fingerprints={"kept-fingerprint"},
    )

    assert decision.allowed is False
    assert decision.missing_fingerprints == ("dropped-fingerprint",)


def test_allows_initial_adoption_when_base_has_no_baseline() -> None:
    checker = load_checker()
    decision = checker.evaluate_baseline_rotation(
        _changed(checker.BASELINE_PATH, "app/api/foo.py"),
        labels=set(),
        base_has_baseline=False,
    )

    assert decision.allowed is True
    assert "initial adoption" in decision.reason


def test_parse_labels_strips_empty_entries() -> None:
    checker = load_checker()
    assert checker.parse_labels(" security-rotation, ,docs-only ") == {
        "security-rotation",
        "docs-only",
    }


def test_parse_labels_json_preserves_comma_inside_label() -> None:
    checker = load_checker()
    assert checker.parse_labels_json('["security-rotation","contains,comma"]') == {
        "security-rotation",
        "contains,comma",
    }


def test_parse_labels_json_rejects_non_string_labels() -> None:
    checker = load_checker()
    try:
        checker.parse_labels_json('["security-rotation", 123]')
    except ValueError as exc:
        assert "array of strings" in str(exc)
    else:
        raise AssertionError("non-string label should fail closed")
