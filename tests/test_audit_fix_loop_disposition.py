from __future__ import annotations

import itertools
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_fix_loop_disposition.py"


def load_auditor():
    spec = importlib.util.spec_from_file_location("audit_fix_loop_disposition", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_no_findings_needs_no_fix_loop_preflight(tmp_path: Path) -> None:
    aud = load_auditor()
    body = "## AI reconciliation\n- no-findings\n"

    assert aud.audit_body(body, repo_root=tmp_path) == []


def test_fixed_in_reconciliation_requires_preflight(tmp_path: Path) -> None:
    aud = load_auditor()
    body = (
        "Plan: plans/PR-Example.md\n"
        "\n"
        "## AI reconciliation\n"
        "- Parser guard -- fixed-in: scripts/parser.py and tests/test_parser.py\n"
    )

    errors = aud.audit_body(body, repo_root=tmp_path)

    assert any("preflight missing" in error for error in errors)


@pytest.mark.parametrize("fence", ["```", "~~~"])
def test_fenced_preflight_example_is_not_a_record(tmp_path: Path, fence: str) -> None:
    aud = load_auditor()
    _write_plan(tmp_path, max_files=1)
    body = "\n".join(
        [
            "Plan: plans/PR-Example.md",
            "",
            "## AI reconciliation",
            "- Parser guard -- fixed-in: scripts/parser.py",
            "",
            "## Fix-loop disposition preflight",
            fence,
            "- Root decision: Parser guard",
            "- Blocking predicate: contract",
            "- Disposition: fixed-in",
            "- Allowed files: scripts/parser.py",
            "- Max files: 1",
            "- Parked hardening: none",
            fence,
        ]
    )

    errors = aud.audit_body(body, repo_root=tmp_path, changed_file_set={"scripts/parser.py"})

    assert any("no root decision records found" in error for error in errors)
    assert any("missing preflight record for AI reconciliation root 'parser guard'" in error for error in errors)


def test_canonical_ai_reconciliation_heading_requires_preflight(tmp_path: Path) -> None:
    aud = load_auditor()
    body = (
        "Plan: plans/PR-Example.md\n"
        "\n"
        "### AI reconciliation\n"
        "* Parser guard -- fixed-in: scripts/parser.py\n"
    )

    errors = aud.audit_body(body, repo_root=tmp_path)

    assert any("preflight missing" in error for error in errors)


def test_valid_fixed_in_preflight_requires_matching_plan_budget(tmp_path: Path) -> None:
    aud = load_auditor()
    _write_plan(tmp_path, max_files=2)
    body = _body(
        ai="- Parser guard -- fixed-in: scripts/parser.py and tests/test_parser.py",
        predicate="contract",
        disposition="fixed-in",
        max_files=2,
    )

    assert aud.audit_body(
        body,
        repo_root=tmp_path,
        changed_file_set={"scripts/parser.py", "tests/test_parser.py"},
    ) == []


def test_fixed_in_requires_source_trace_fields(tmp_path: Path) -> None:
    aud = load_auditor()
    _write_plan(tmp_path, max_files=1)
    body = _body(
        ai="- Parser guard -- fixed-in: scripts/parser.py",
        predicate="contract",
        disposition="fixed-in",
        max_files=1,
        include_trace=False,
    )

    errors = aud.audit_body(body, repo_root=tmp_path, changed_file_set={"scripts/parser.py"})

    assert any("missing '- Source Trace: ...'" in error for error in errors)
    assert any("missing '- Upstream Files: ...'" in error for error in errors)
    assert any("missing '- Fix Strategy: ...'" in error for error in errors)


def test_waiver_requires_source_trace_fields(tmp_path: Path) -> None:
    aud = load_auditor()
    _write_plan(tmp_path, max_files=1)
    body = _body(
        ai="- Logging polish -- waived-nit: skip-worthy",
        root="Logging polish",
        predicate="not-blocking",
        disposition="waived-nit",
        max_files=1,
        include_trace=False,
    )

    errors = aud.audit_body(body, repo_root=tmp_path, changed_file_set={"scripts/parser.py"})

    assert any("missing '- Source Trace: ...'" in error for error in errors)
    assert any("missing '- Upstream Files: ...'" in error for error in errors)
    assert any("missing '- Fix Strategy: ...'" in error for error in errors)


def test_upstream_root_must_touch_declared_upstream_file(tmp_path: Path) -> None:
    aud = load_auditor()
    _write_plan(tmp_path, max_files=2)
    body = _body(
        ai="- Parser guard -- fixed-in: scripts/downstream.py",
        predicate="contract",
        disposition="fixed-in",
        max_files=2,
        allowed_files="scripts/downstream.py, tests/test_parser.py, scripts/parser.py",
        upstream_files="scripts/parser.py",
    )

    errors = aud.audit_body(
        body,
        repo_root=tmp_path,
        changed_file_set={"scripts/downstream.py", "tests/test_parser.py"},
    )

    assert any("fixed-in upstream-root must change at least one declared upstream file" in error for error in errors)


def test_source_trace_must_show_chain_to_source(tmp_path: Path) -> None:
    aud = load_auditor()
    _write_plan(tmp_path, max_files=1)
    body = _body(
        ai="- Parser guard -- fixed-in: scripts/parser.py",
        predicate="contract",
        disposition="fixed-in",
        max_files=1,
        source_trace="Parser guard",
    )

    errors = aud.audit_body(body, repo_root=tmp_path, changed_file_set={"scripts/parser.py"})

    assert any("source trace must name the chain from symptom -> upstream source" in error for error in errors)


def test_source_trace_rejects_placeholder_chain_endpoints(tmp_path: Path) -> None:
    aud = load_auditor()
    _write_plan(tmp_path, max_files=1)
    body = _body(
        ai="- Parser guard -- fixed-in: scripts/parser.py",
        predicate="contract",
        disposition="fixed-in",
        max_files=1,
        source_trace="TBD -> TBD",
    )

    errors = aud.audit_body(body, repo_root=tmp_path, changed_file_set={"scripts/parser.py"})

    assert any("non-placeholder endpoints" in error for error in errors)


def test_source_trace_rejects_decorated_template_chain(tmp_path: Path) -> None:
    aud = load_auditor()
    _write_plan(tmp_path, max_files=1)
    body = _body(
        ai="- Parser guard -- fixed-in: scripts/parser.py",
        predicate="contract",
        disposition="fixed-in",
        max_files=1,
        source_trace="<symptom -> intermediate cause -> upstream source>",
    )

    errors = aud.audit_body(body, repo_root=tmp_path, changed_file_set={"scripts/parser.py"})

    assert any("non-placeholder endpoints" in error for error in errors)


def test_source_trace_accepts_unicode_endpoint_chain(tmp_path: Path) -> None:
    aud = load_auditor()
    _write_plan(tmp_path, max_files=1)
    body = _body(
        ai="- Parser guard -- fixed-in: scripts/parser.py",
        predicate="contract",
        disposition="fixed-in",
        max_files=1,
        source_trace="症状 -> 根因",
    )

    assert aud.audit_body(body, repo_root=tmp_path, changed_file_set={"scripts/parser.py"}) == []


def test_fix_loop_trace_contract_source_trace_endpoint_grammar() -> None:
    aud = load_auditor()
    trace_tokens_by_expected = {
        "review claim": True,
        "parser branch": True,
        "症状": True,
        "admission source": True,
        "TBD": False,
        "TBD symptom": False,
        "TBD upstream source": False,
        "unknown": False,
        "<symptom": False,
        "intermediate cause": False,
        "upstream source>": False,
        "...": False,
    }
    trace_containers = {
        "bare": lambda value: value,
        "padded": lambda value: f"  {value}  ",
    }
    trace_families = {
        "symptom": 0,
        "middle": 1,
        "source": 2,
    }

    for token, container, family in itertools.product(
        trace_tokens_by_expected,
        trace_containers,
        trace_families,
    ):
        endpoints = ["review claim", "parser branch", "admission source"]
        endpoints[trace_families[family]] = trace_containers[container](token)
        trace = " -> ".join(endpoints)
        spec_derived_oracle = trace_tokens_by_expected[token]

        assert aud.source_trace_is_valid(trace) is spec_derived_oracle


def test_upstream_files_are_normalized_before_changed_file_match(tmp_path: Path) -> None:
    aud = load_auditor()
    _write_plan(tmp_path, max_files=1)
    body = _body(
        ai="- Parser guard -- fixed-in: scripts/parser.py",
        predicate="contract",
        disposition="fixed-in",
        max_files=1,
        upstream_files="./scripts\\parser.py",
    )

    assert aud.audit_body(body, repo_root=tmp_path, changed_file_set={"scripts/parser.py"}) == []


def test_upstream_files_reject_placeholder_tokens(tmp_path: Path) -> None:
    aud = load_auditor()
    _write_plan(tmp_path, max_files=1)
    body = _body(
        ai="- Logging polish -- waived-nit: skip-worthy",
        root="Logging polish",
        predicate="not-blocking",
        disposition="waived-nit",
        max_files=1,
        fix_strategy="symptom-only-deferred",
        upstream_files="none",
        symptom_only_reason="skip-worthy nit is not blocking this workflow gate",
        follow_up="waived-nit in AI reconciliation",
    )

    errors = aud.audit_body(body, repo_root=tmp_path, changed_file_set={"scripts/parser.py"})

    assert any("upstream files must contain repo-relative paths" in error for error in errors)


def test_symptom_only_strategy_requires_reason_and_followup(tmp_path: Path) -> None:
    aud = load_auditor()
    _write_plan(tmp_path, max_files=1)
    body = _body(
        ai="- Parser guard -- fixed-in: scripts/parser.py",
        predicate="contract",
        disposition="fixed-in",
        max_files=1,
        fix_strategy="symptom-only-deferred",
    )

    errors = aud.audit_body(body, repo_root=tmp_path, changed_file_set={"scripts/parser.py"})

    assert any("symptom-only-deferred requires '- Symptom-Only Reason: ...'" in error for error in errors)
    assert any("symptom-only-deferred requires '- Follow-Up: ...'" in error for error in errors)


def test_symptom_only_strategy_passes_with_reason_and_followup(tmp_path: Path) -> None:
    aud = load_auditor()
    _write_plan(tmp_path, max_files=1)
    body = _body(
        ai="- Parser guard -- fixed-in: scripts/parser.py",
        predicate="contract",
        disposition="fixed-in",
        max_files=1,
        fix_strategy="symptom-only-deferred",
        symptom_only_reason="true upstream parser is owned by another active lane",
        follow_up="HARDENING.md ROOT-TRACE-1",
    )

    assert aud.audit_body(body, repo_root=tmp_path, changed_file_set={"scripts/parser.py"}) == []


def test_waived_hardening_preflight_uses_not_blocking(tmp_path: Path) -> None:
    aud = load_auditor()
    _write_plan(tmp_path, max_files=1)
    body = _body(
        ai="- Logging polish -- waived-out-of-scope: parked in HARDENING.md",
        root="Logging polish",
        predicate="not-blocking",
        disposition="waived-out-of-scope",
        max_files=1,
    )

    assert aud.audit_body(
        body,
        repo_root=tmp_path,
        changed_file_set={"scripts/parser.py"},
    ) == []


def test_fixed_in_cannot_claim_not_blocking(tmp_path: Path) -> None:
    aud = load_auditor()
    _write_plan(tmp_path, max_files=1)
    body = _body(
        ai="- Parser guard -- fixed-in: scripts/parser.py",
        predicate="not-blocking",
        disposition="fixed-in",
        max_files=1,
    )

    errors = aud.audit_body(body, repo_root=tmp_path)

    assert any("fixed-in findings need a blocking predicate" in error for error in errors)


def test_waiver_cannot_claim_blocking_predicate(tmp_path: Path) -> None:
    aud = load_auditor()
    _write_plan(tmp_path, max_files=1)
    body = _body(
        ai="- Logging polish -- waived-nit: skip-worthy",
        root="Logging polish",
        predicate="contract",
        disposition="waived-nit",
        max_files=1,
    )

    errors = aud.audit_body(body, repo_root=tmp_path)

    assert any("waived findings must use blocking predicate 'not-blocking'" in error for error in errors)


def test_body_budget_must_match_plan_scope_budget(tmp_path: Path) -> None:
    aud = load_auditor()
    _write_plan(tmp_path, max_files=2)
    body = _body(
        ai="- Parser guard -- fixed-in: scripts/parser.py",
        predicate="contract",
        disposition="fixed-in",
        max_files=3,
    )

    errors = aud.audit_body(body, repo_root=tmp_path)

    assert any("does not match plan Scope Max files 2" in error for error in errors)


def test_allowed_files_must_cover_changed_files(tmp_path: Path) -> None:
    aud = load_auditor()
    _write_plan(tmp_path, max_files=2)
    body = _body(
        ai="- Parser guard -- fixed-in: scripts/parser.py",
        predicate="contract",
        disposition="fixed-in",
        max_files=2,
    )

    errors = aud.audit_body(
        body,
        repo_root=tmp_path,
        changed_file_set={"scripts/parser.py", "scripts/unlisted.py"},
    )

    assert any("changed files outside allowed set: scripts/unlisted.py" in error for error in errors)


def test_preflight_disposition_must_match_ai_reconciliation(tmp_path: Path) -> None:
    aud = load_auditor()
    _write_plan(tmp_path, max_files=1)
    body = _body(
        ai="- Parser guard -- waived-nit: skip-worthy",
        predicate="not-blocking",
        disposition="waived-out-of-scope",
        max_files=1,
    )

    errors = aud.audit_body(body, repo_root=tmp_path)

    assert any("does not match AI reconciliation disposition 'waived-nit'" in error for error in errors)


def test_each_ai_root_needs_its_own_preflight_record(tmp_path: Path) -> None:
    aud = load_auditor()
    _write_plan(tmp_path, max_files=1)
    body = _body(
        ai="\n".join(
            [
                "- Parser guard -- fixed-in: scripts/parser.py",
                "- Logging polish -- waived-nit: skip-worthy",
            ]
        ),
        predicate="contract",
        disposition="fixed-in",
        max_files=1,
    )

    errors = aud.audit_body(body, repo_root=tmp_path)

    assert any("missing preflight record for AI reconciliation root 'logging polish'" in error for error in errors)


def test_internal_separators_do_not_truncate_distinct_roots(tmp_path: Path) -> None:
    aud = load_auditor()
    _write_plan(tmp_path, max_files=1)
    body = "\n".join(
        [
            "Plan: plans/PR-Example.md",
            "",
            "## AI reconciliation",
            "- Cache -- parser bug -- fixed-in: scripts/parser.py",
            "- Cache -- eviction bug -- fixed-in: scripts/parser.py",
            "",
            "## Fix-loop disposition preflight",
            "- Root decision: Cache",
            "- Source trace: review claim -> cache parser branch -> cache source",
            "- Upstream files: scripts/parser.py",
            "- Fix strategy: upstream-root",
            "- Blocking predicate: contract",
            "- Disposition: fixed-in",
            "- Allowed files: scripts/parser.py",
            "- Max files: 1",
            "- Parked hardening: none",
        ]
    )

    errors = aud.audit_body(body, repo_root=tmp_path, changed_file_set={"scripts/parser.py"})

    assert any("missing preflight record for AI reconciliation root 'cache parser bug'" in error for error in errors)
    assert any("missing preflight record for AI reconciliation root 'cache eviction bug'" in error for error in errors)


def test_multiple_ai_roots_pass_with_multiple_preflight_records(tmp_path: Path) -> None:
    aud = load_auditor()
    _write_plan(tmp_path, max_files=1)
    body = "\n".join(
        [
            "Plan: plans/PR-Example.md",
            "",
            "## AI reconciliation",
            "- Parser guard -- fixed-in: scripts/parser.py",
            "- Logging polish -- waived-nit: skip-worthy",
            "",
            "## Fix-loop disposition preflight",
            "- Root decision: Parser guard",
            "- Source trace: review claim -> parser branch -> admission source",
            "- Upstream files: scripts/parser.py",
            "- Fix strategy: upstream-root",
            "- Blocking predicate: contract",
            "- Disposition: fixed-in",
            "- Allowed files: scripts/parser.py",
            "- Max files: 1",
            "- Parked hardening: none",
            "- Root decision: Logging polish",
            "- Source trace: review claim -> non-blocking polish -> parked outside this slice",
            "- Upstream files: HARDENING.md",
            "- Fix strategy: symptom-only-deferred",
            "- Symptom-only reason: skip-worthy nit is not blocking this workflow gate",
            "- Follow-up: waived-nit in AI reconciliation",
            "- Blocking predicate: not-blocking",
            "- Disposition: waived-nit",
            "- Allowed files: scripts/parser.py",
            "- Max files: 1",
            "- Parked hardening: none",
        ]
    )

    assert aud.audit_body(body, repo_root=tmp_path, changed_file_set={"scripts/parser.py"}) == []


def test_per_root_allowed_files_are_validated_as_a_union(tmp_path: Path) -> None:
    aud = load_auditor()
    _write_plan(tmp_path, max_files=2)
    body = "\n".join(
        [
            "Plan: plans/PR-Example.md",
            "",
            "## AI reconciliation",
            "- Parser guard -- fixed-in: scripts/parser.py",
            "- Renderer guard -- fixed-in: scripts/renderer.py",
            "",
            "## Fix-loop disposition preflight",
            "- Root decision: Parser guard",
            "- Source trace: review claim -> parser branch -> admission source",
            "- Upstream files: scripts/parser.py",
            "- Fix strategy: upstream-root",
            "- Blocking predicate: contract",
            "- Disposition: fixed-in",
            "- Allowed files: scripts/parser.py",
            "- Max files: 2",
            "- Parked hardening: none",
            "- Root decision: Renderer guard",
            "- Source trace: review claim -> renderer branch -> render source",
            "- Upstream files: scripts/renderer.py",
            "- Fix strategy: upstream-root",
            "- Blocking predicate: contract",
            "- Disposition: fixed-in",
            "- Allowed files: scripts/renderer.py",
            "- Max files: 2",
            "- Parked hardening: none",
        ]
    )

    assert aud.audit_body(
        body,
        repo_root=tmp_path,
        changed_file_set={"scripts/parser.py", "scripts/renderer.py"},
    ) == []


def test_star_and_numbered_reconciliation_bullets_are_validated(tmp_path: Path) -> None:
    aud = load_auditor()
    _write_plan(tmp_path, max_files=1)
    body = "\n".join(
        [
            "Plan: plans/PR-Example.md",
            "",
            "## AI reconciliation",
            "* Parser guard -- fixed-in: scripts/parser.py",
            "1. Logging polish -- waived-nit: skip-worthy",
            "",
            "## Fix-loop disposition preflight",
            "- Root decision: Parser guard",
            "- Source trace: review claim -> parser branch -> admission source",
            "- Upstream files: scripts/parser.py",
            "- Fix strategy: upstream-root",
            "- Blocking predicate: contract",
            "- Disposition: fixed-in",
            "- Allowed files: scripts/parser.py",
            "- Max files: 1",
            "- Parked hardening: none",
            "- Root decision: Logging polish",
            "- Source trace: review claim -> non-blocking polish -> parked outside this slice",
            "- Upstream files: HARDENING.md",
            "- Fix strategy: symptom-only-deferred",
            "- Symptom-only reason: skip-worthy nit is not blocking this workflow gate",
            "- Follow-up: waived-nit in AI reconciliation",
            "- Blocking predicate: not-blocking",
            "- Disposition: waived-nit",
            "- Allowed files: scripts/parser.py",
            "- Max files: 1",
            "- Parked hardening: none",
        ]
    )

    assert aud.audit_body(body, repo_root=tmp_path, changed_file_set={"scripts/parser.py"}) == []


def test_inconsistent_preflight_budgets_fail(tmp_path: Path) -> None:
    aud = load_auditor()
    _write_plan(tmp_path, max_files=1)
    body = "\n".join(
        [
            "Plan: plans/PR-Example.md",
            "",
            "## AI reconciliation",
            "- Parser guard -- fixed-in: scripts/parser.py",
            "- Logging polish -- waived-nit: skip-worthy",
            "",
            "## Fix-loop disposition preflight",
            "- Root decision: Parser guard",
            "- Source trace: review claim -> parser branch -> admission source",
            "- Upstream files: scripts/parser.py",
            "- Fix strategy: upstream-root",
            "- Blocking predicate: contract",
            "- Disposition: fixed-in",
            "- Allowed files: scripts/parser.py",
            "- Max files: 1",
            "- Parked hardening: none",
            "- Root decision: Logging polish",
            "- Source trace: review claim -> non-blocking polish -> parked outside this slice",
            "- Upstream files: HARDENING.md",
            "- Fix strategy: symptom-only-deferred",
            "- Symptom-only reason: skip-worthy nit is not blocking this workflow gate",
            "- Follow-up: waived-nit in AI reconciliation",
            "- Blocking predicate: not-blocking",
            "- Disposition: waived-nit",
            "- Allowed files: scripts/parser.py",
            "- Max files: 2",
            "- Parked hardening: none",
        ]
    )

    errors = aud.audit_body(body, repo_root=tmp_path, changed_file_set={"scripts/parser.py"})

    assert any("all records must declare the same Max files value" in error for error in errors)
    assert any("body Max files 2 does not match plan Scope Max files 1" in error for error in errors)


def test_absolute_or_escaping_plan_path_is_rejected(tmp_path: Path) -> None:
    aud = load_auditor()
    _write_plan(tmp_path, max_files=1)
    for plan_line in ("Plan: /tmp/PR-Example.md", "Plan: plans/../PR-Example.md"):
        body = _body(
            ai="- Parser guard -- fixed-in: scripts/parser.py",
            predicate="contract",
            disposition="fixed-in",
            max_files=1,
        ).replace("Plan: plans/PR-Example.md", plan_line)

        errors = aud.audit_body(body, repo_root=tmp_path)

        assert any("PR body must start with Plan: plans/PR-<Slice>.md" in error for error in errors)


def test_symlinked_plan_path_is_rejected(tmp_path: Path) -> None:
    aud = load_auditor()
    target = tmp_path / "target.md"
    target.write_text("# Target\n\n## Scope (this PR)\nMax files: 1\n", encoding="utf-8")
    plan = tmp_path / "plans" / "PR-Example.md"
    plan.parent.mkdir()
    plan.symlink_to(target)
    body = _body(
        ai="- Parser guard -- fixed-in: scripts/parser.py",
        predicate="contract",
        disposition="fixed-in",
        max_files=1,
    )

    errors = aud.audit_body(body, repo_root=tmp_path)

    assert any("plan path must not be a symlink" in error for error in errors)


def test_plan_scope_budget_is_required(tmp_path: Path) -> None:
    aud = load_auditor()
    plan = tmp_path / "plans" / "PR-Example.md"
    plan.parent.mkdir()
    plan.write_text("# PR-Example\n\n## Scope (this PR)\n\nNo budget.\n", encoding="utf-8")
    body = _body(
        ai="- Parser guard -- fixed-in: scripts/parser.py",
        predicate="contract",
        disposition="fixed-in",
        max_files=1,
    )

    errors = aud.audit_body(body, repo_root=tmp_path)

    assert any("plan Scope must declare Max files: N" in error for error in errors)


def test_plan_scope_budget_rejects_malformed_value(tmp_path: Path) -> None:
    aud = load_auditor()
    plan = tmp_path / "plans" / "PR-Example.md"
    plan.parent.mkdir()
    plan.write_text("# PR-Example\n\n## Scope (this PR)\nMax files: many\n", encoding="utf-8")

    with pytest.raises(ValueError, match="malformed Max files value"):
        aud.plan_max_files(plan.read_text(encoding="utf-8"))


def test_changed_files_raises_when_git_diff_fails(tmp_path: Path) -> None:
    aud = load_auditor()

    with pytest.raises(RuntimeError, match="git diff failed|Not a git repository"):
        aud.changed_files("origin/main", repo_root=tmp_path)


def test_cli_exit_codes(tmp_path: Path) -> None:
    aud = load_auditor()
    _write_plan(tmp_path, max_files=1)
    ok = tmp_path / "ok.md"
    ok.write_text(
        _body(
            ai="- Logging polish -- waived-out-of-scope: parked in HARDENING.md",
            root="Logging polish",
            predicate="not-blocking",
            disposition="waived-out-of-scope",
            max_files=1,
        ),
        encoding="utf-8",
    )
    bad = tmp_path / "bad.md"
    bad.write_text("## AI reconciliation\n- Parser guard -- fixed-in: scripts/parser.py\n", encoding="utf-8")

    assert aud.main(["--repo-root", str(tmp_path), "--current-pr-body-file", str(ok)]) == 0
    assert aud.main(["--repo-root", str(tmp_path), "--current-pr-body-file", str(bad)]) == 1


def _write_plan(root: Path, *, max_files: int) -> None:
    plan = root / "plans" / "PR-Example.md"
    plan.parent.mkdir()
    plan.write_text(
        "\n".join(
            [
                "# PR-Example",
                "",
                "## Scope (this PR)",
                "",
                "Ownership lane: workflow/fix-loop-disposition",
                "Slice phase: Workflow/process",
                f"Max files: {max_files}",
                "",
                "## Mechanism",
                "Details.",
            ]
        ),
        encoding="utf-8",
    )


def _body(
    *,
    ai: str,
    predicate: str,
    disposition: str,
    max_files: int,
    root: str = "Parser guard",
    allowed_files: str = "scripts/parser.py, tests/test_parser.py",
    include_trace: bool = True,
    source_trace: str = "review claim -> parser branch -> admission source",
    upstream_files: str = "scripts/parser.py",
    fix_strategy: str = "upstream-root",
    symptom_only_reason: str | None = None,
    follow_up: str | None = None,
) -> str:
    lines = [
        "Plan: plans/PR-Example.md",
        "",
        "## AI reconciliation",
        ai,
        "",
        "## Fix-loop disposition preflight",
        f"- Root decision: {root}",
    ]
    if include_trace:
        lines.extend(
            [
                f"- Source trace: {source_trace}",
                f"- Upstream files: {upstream_files}",
                f"- Fix strategy: {fix_strategy}",
            ]
        )
        if symptom_only_reason is not None:
            lines.append(f"- Symptom-only reason: {symptom_only_reason}")
        if follow_up is not None:
            lines.append(f"- Follow-up: {follow_up}")
    lines.extend(
        [
            f"- Blocking predicate: {predicate}",
            f"- Disposition: {disposition}",
            f"- Allowed files: {allowed_files}",
            f"- Max files: {max_files}",
            "- Parked hardening: none",
            "",
            "## Verification",
            "- pending",
        ]
    )
    return "\n".join(lines)
