from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_workflow_security_posture.py"


def load_auditor():
    name = "audit_workflow_security_posture"
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


def _write_workflow(tmp_path: Path, name: str, text: str) -> Path:
    workflow = tmp_path / name
    workflow.write_text(text, encoding="utf-8")
    return workflow


def test_unapproved_pull_request_target_is_error(tmp_path: Path) -> None:
    auditor = load_auditor()
    workflow = _write_workflow(
        tmp_path,
        "unsafe.yml",
        """
name: Unsafe
on:
  pull_request_target:
jobs:
  test:
    runs-on: ubuntu-latest
    steps: []
""",
    )

    findings = auditor.audit_workflow(workflow)

    assert any(f.level == "ERROR" and "pull_request_target" in f.detail for f in findings)


def test_approved_gitleaks_baseline_workflow_pull_request_target_is_allowed(tmp_path: Path) -> None:
    auditor = load_auditor()
    workflow = _write_workflow(
        tmp_path,
        "gitleaks_baseline_growth_guard.yml",
        """
name: Security
on:
  pull_request_target:
permissions:
  contents: read
  pull-requests: read
jobs:
  gitleaks-baseline-guard:
    if: github.event_name == 'pull_request_target'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@1234567890abcdef1234567890abcdef12345678
        with:
          ref: ${{ github.event.pull_request.base.sha }}
""",
    )

    findings = auditor.audit_workflow(workflow)

    assert not [f for f in findings if f.level == "ERROR"]
    assert any(f.level == "WARN" and "allowed pull_request_target" in f.detail for f in findings)


def test_old_security_guardrails_pull_request_target_job_is_no_longer_allowed(tmp_path: Path) -> None:
    auditor = load_auditor()
    workflow = _write_workflow(
        tmp_path,
        "security_guardrails.yml",
        """
name: Security
on:
  pull_request_target:
jobs:
  gitleaks-baseline-guard:
    if: github.event_name == 'pull_request_target'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@1234567890abcdef1234567890abcdef12345678
        with:
          ref: ${{ github.event.pull_request.base.sha }}
""",
    )

    findings = auditor.audit_workflow(workflow)

    assert any(f.level == "ERROR" and "pull_request_target" in f.detail for f in findings)


def test_gitleaks_baseline_extra_pull_request_target_job_is_error(tmp_path: Path) -> None:
    auditor = load_auditor()
    workflow = _write_workflow(
        tmp_path,
        "gitleaks_baseline_growth_guard.yml",
        """
name: Security
on:
  pull_request_target:
jobs:
  gitleaks-baseline-guard:
    if: github.event_name == 'pull_request_target'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@1234567890abcdef1234567890abcdef12345678
        with:
          ref: ${{ github.event.pull_request.base.sha }}
  unsafe:
    if: github.event_name == 'pull_request_target'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@1234567890abcdef1234567890abcdef12345678
""",
    )

    findings = auditor.audit_workflow(workflow)

    assert any(f.level == "ERROR" and "unsafe" in f.detail and "pull_request_target" in f.detail for f in findings)


def test_unapproved_oidc_write_is_error(tmp_path: Path) -> None:
    auditor = load_auditor()
    workflow = _write_workflow(
        tmp_path,
        "oidc.yml",
        """
name: OIDC
on: pull_request
jobs:
  cloud:
    runs-on: ubuntu-latest
    permissions:
      id-token: write
    steps: []
""",
    )

    findings = auditor.audit_workflow(workflow)

    assert any(f.level == "ERROR" and "id-token" in f.detail for f in findings)


def test_write_all_is_oidc_error_at_workflow_and_job_scope(tmp_path: Path) -> None:
    auditor = load_auditor()
    workflow = _write_workflow(
        tmp_path,
        "write-all.yml",
        """
name: Write All
on: pull_request
permissions: write-all
jobs:
  cloud:
    runs-on: ubuntu-latest
    permissions: write-all
    steps: []
""",
    )

    findings = auditor.audit_workflow(workflow)

    assert any(f.level == "ERROR" and "workflow-scope" in f.detail and "write-all" in f.detail for f in findings)
    assert any(f.level == "ERROR" and "job cloud" in f.detail and "write-all" in f.detail for f in findings)


def test_claude_oidc_write_is_warn_only(tmp_path: Path) -> None:
    auditor = load_auditor()
    workflow = _write_workflow(
        tmp_path,
        "claude.yml",
        """
name: Claude
on: issue_comment
jobs:
  claude:
    if: github.actor == github.repository_owner && contains(github.event.comment.body, '@claude')
    runs-on: ubuntu-latest
    permissions:
      id-token: write
    steps: []
""",
    )

    findings = auditor.audit_workflow(workflow)

    assert any(f.level == "WARN" and "allowed id-token" in f.detail for f in findings)
    assert not [f for f in findings if f.level == "ERROR"]


def test_claude_extra_oidc_job_without_owner_gate_is_error(tmp_path: Path) -> None:
    auditor = load_auditor()
    workflow = _write_workflow(
        tmp_path,
        "claude.yml",
        """
name: Claude
on: issue_comment
jobs:
  claude:
    if: github.actor == github.repository_owner && contains(github.event.comment.body, '@claude')
    runs-on: ubuntu-latest
    permissions:
      id-token: write
    steps: []
  unsafe:
    runs-on: ubuntu-latest
    permissions:
      id-token: write
    steps: []
""",
    )

    findings = auditor.audit_workflow(workflow)

    assert any(f.level == "ERROR" and "job unsafe" in f.detail and "owner-gated" in f.detail for f in findings)


def test_mutable_action_ref_is_warn_only(tmp_path: Path) -> None:
    auditor = load_auditor()
    workflow = _write_workflow(
        tmp_path,
        "mutable.yml",
        """
name: Mutable
on: pull_request
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: local/action
""",
    )

    findings = auditor.audit_workflow(workflow)

    assert any(f.level == "WARN" and "actions/checkout@v4" in f.detail for f in findings)
    assert any(f.level == "WARN" and "local/action@<missing ref>" in f.detail for f in findings)


def test_setup_python_tag_warns_and_pinned_ref_is_clean(tmp_path: Path) -> None:
    auditor = load_auditor()
    mutable = _write_workflow(
        tmp_path,
        "setup-python-mutable.yml",
        """
name: Mutable setup-python
on: pull_request
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/setup-python@v5
""",
    )
    pinned = _write_workflow(
        tmp_path,
        "setup-python-pinned.yml",
        """
name: Pinned setup-python
on: pull_request
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065
""",
    )

    mutable_findings = auditor.audit_workflow(mutable)
    pinned_findings = auditor.audit_workflow(pinned)

    assert any(f.level == "WARN" and "actions/setup-python@v5" in f.detail for f in mutable_findings)
    assert pinned_findings == []


def test_yaml_workflow_files_are_audited(tmp_path: Path) -> None:
    auditor = load_auditor()
    _write_workflow(
        tmp_path,
        "unsafe.yaml",
        """
name: Unsafe YAML
on:
  pull_request_target:
jobs:
  test:
    runs-on: ubuntu-latest
    steps: []
""",
    )

    findings = auditor.audit_workflows(tmp_path)

    assert any(f.level == "ERROR" and "unsafe.yaml" in f.path for f in findings)


def test_job_level_reusable_workflow_ref_is_warned(tmp_path: Path) -> None:
    auditor = load_auditor()
    workflow = _write_workflow(
        tmp_path,
        "reusable.yml",
        """
name: Reusable
on: pull_request
jobs:
  call:
    uses: owner/repo/.github/workflows/build.yml@main
""",
    )

    findings = auditor.audit_workflow(workflow)

    assert any(f.level == "WARN" and "job call reusable workflow" in f.detail and "@main" in f.detail for f in findings)


def test_container_and_service_images_are_warned_when_not_digest_pinned(tmp_path: Path) -> None:
    auditor = load_auditor()
    workflow = _write_workflow(
        tmp_path,
        "containers.yml",
        """
name: Containers
on: pull_request
jobs:
  test:
    runs-on: ubuntu-latest
    container:
      image: python:3.13
    services:
      postgres:
        image: postgres:16
      redis: redis:7
    steps: []
""",
    )

    findings = auditor.audit_workflow(workflow)

    assert any(f.level == "WARN" and "container image python:3.13" in f.detail for f in findings)
    assert any(f.level == "WARN" and "service postgres image postgres:16" in f.detail for f in findings)
    assert any(f.level == "WARN" and "service redis image redis:7" in f.detail for f in findings)


def test_sha_pinned_action_ref_is_clean(tmp_path: Path) -> None:
    auditor = load_auditor()
    workflow = _write_workflow(
        tmp_path,
        "pinned.yml",
        """
name: Pinned
on: pull_request
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@1234567890abcdef1234567890abcdef12345678
""",
    )

    findings = auditor.audit_workflow(workflow)

    assert findings == []


def _trusted_gate_workflow(job: str) -> str:
    # Declares permissions explicitly because every real enrolled gate does and
    # because an omitted block is no longer admissible: it would inherit a
    # write-capable repo/org default. See
    # test_enrolled_job_omitting_permissions_entirely_is_rejected.
    return f"""
name: Gate
on:
  pull_request_target:
permissions:
  contents: read
  pull-requests: read
jobs:
  {job}:
    if: github.event_name == 'pull_request_target'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0
        with:
          ref: ${{{{ github.event.pull_request.base.sha }}}}
"""


def test_converted_meta_gates_pull_request_target_is_allowed(tmp_path: Path) -> None:
    auditor = load_auditor()
    for name, job in (
        ("diff_budget.yml", "diff-budget"),
        ("ai_reconciliation_live.yml", "live-reconciliation"),
        ("pr_body_contract.yml", "pr-body-contract"),
        ("pre_push_audit.yml", "pre-push-audit"),
        ("session_lane.yml", "session-lane"),
        ("plan_admission.yml", "plan-admission"),
    ):
        workflow = _write_workflow(tmp_path, name, _trusted_gate_workflow(job))
        findings = auditor.audit_workflow(workflow)
        assert not [f for f in findings if f.level == "ERROR"], (name, findings)
        assert any("allowed pull_request_target" in f.detail for f in findings)


def test_review_contract_preadmission_requires_canonical_workflow(tmp_path: Path) -> None:
    auditor = load_auditor()
    workflow = _write_workflow(
        tmp_path,
        "review_contract.yml",
        auditor.REVIEW_CONTRACT_CANONICAL_WORKFLOW,
    )

    findings = auditor.audit_workflow(workflow)

    assert not [f for f in findings if f.level == "ERROR"], findings
    assert any("allowed pull_request_target" in f.detail for f in findings)


def test_review_contract_preadmission_rejects_noncanonical_workflow(tmp_path: Path) -> None:
    auditor = load_auditor()
    workflow_text = auditor.REVIEW_CONTRACT_CANONICAL_WORKFLOW.replace(
        "  contents: read\n  pull-requests: read",
        "  contents: write\n  pull-requests: write",
    ).replace(
        '          cd "$RUNNER_TEMP/pr-tree"',
        '          cd "$RUNNER_TEMP/pr-tree"\n          gh pr merge "$PR_NUMBER" --squash',
    )
    workflow = _write_workflow(tmp_path, "review_contract.yml", workflow_text)

    findings = auditor.audit_workflow(workflow)

    assert any(f.level == "ERROR" and "pull_request_target" in f.detail for f in findings)


def _gate_workflow_missing(part: str) -> str:
    """A trusted-base gate that is correct except for exactly one element.

    Permissions are always present and read-only. Otherwise the permissions
    precondition rejects the fixture first and the guard-shape branch under test
    is never reached, so the test would pass with that branch deleted.
    """
    if_line = (
        "" if part == "if" else "    if: github.event_name == 'pull_request_target'\n"
    )
    if part == "steps":
        steps = "    steps: []\n"
    else:
        ref = "v7" if part == "pin" else "9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0"
        with_ref = (
            "${{ github.event.pull_request.head.sha }}"
            if part == "ref"
            else "${{ github.event.pull_request.base.sha }}"
        )
        steps = (
            "    steps:\n"
            f"      - uses: actions/checkout@{ref}\n"
            "        with:\n"
            f"          ref: {with_ref}\n"
        )
    return (
        "\nname: Gate\non:\n  pull_request_target:\n"
        "permissions:\n  contents: read\n  pull-requests: read\n"
        "jobs:\n  diff-budget:\n" + if_line + "    runs-on: ubuntu-latest\n" + steps
    )


@pytest.mark.parametrize("part", ["if", "pin", "ref", "steps"])
def test_allowlisted_gate_without_guard_shape_is_still_error(
    tmp_path: Path, part: str
) -> None:
    """Allowlisting is necessary but not sufficient.

    One parameter per guard-shape element, each violated on its own with the
    other elements correct, so every branch is independently proven to fire.
    Previously this was a single fixture that omitted `permissions`; once the
    permissions precondition landed it rejected the fixture before any
    guard-shape check ran, and the test stayed green with both checks deleted.
    """
    auditor = load_auditor()
    workflow = _write_workflow(tmp_path, "diff_budget.yml", _gate_workflow_missing(part))

    findings = auditor.audit_workflow(workflow)

    assert any(
        f.level == "ERROR" and "trusted-base guard shape" in f.detail
        for f in findings
    ), part


def test_allowlisted_gate_with_the_full_guard_shape_is_admitted(tmp_path: Path) -> None:
    """The control for the parametrized rejections above.

    Without this, every parameter could be passing because the fixture builder
    produces something universally rejected rather than because the specific
    element under test is missing.
    """
    auditor = load_auditor()
    workflow = _write_workflow(tmp_path, "diff_budget.yml", _gate_workflow_missing("none"))

    findings = auditor.audit_workflow(workflow)

    assert not [f for f in findings if f.level == "ERROR"], findings
    assert any("allowed pull_request_target" in f.detail for f in findings)


def test_gate_job_name_in_wrong_file_is_error(tmp_path: Path) -> None:
    auditor = load_auditor()
    workflow = _write_workflow(
        tmp_path, "impostor.yml", _trusted_gate_workflow("diff-budget")
    )
    findings = auditor.audit_workflow(workflow)
    assert any(
        f.level == "ERROR" and "trusted-base guard shape" in f.detail
        for f in findings
    )


def test_contact_write_boundary_identity_is_allowlisted(tmp_path: Path) -> None:
    """The enrolled (file, job) pair must match the real workflow exactly.

    The entry is two strings. Nothing else in this suite constructs
    `contact_write_boundary.yml` with the `contact-write-boundary` job, so a
    typo in either would merge green and silently leave the workflow
    unenrolled -- the gate it authorises would then fail its own audit for a
    reason nobody would connect to this change.
    """
    auditor = load_auditor()
    workflow = _write_workflow(
        tmp_path,
        "contact_write_boundary.yml",
        _trusted_gate_workflow("contact-write-boundary"),
    )

    findings = auditor.audit_workflow(workflow)
    assert not [f for f in findings if f.level == "ERROR"], findings
    assert any("allowed pull_request_target" in f.detail for f in findings)


def test_contact_write_boundary_enrolment_still_requires_the_guard_shape(
    tmp_path: Path,
) -> None:
    """Enrolment widens eligibility, not permission.

    The enrolled identity without the event-name guard must still be rejected,
    or the allowlist entry would be a bypass rather than an admission record.

    Permissions are read-only here on purpose. The fixture originally omitted
    them, and once the permissions precondition landed it rejected this workflow
    before the guard-shape checks ran -- so the test passed for the wrong reason
    and stayed green with those checks deleted.
    """
    auditor = load_auditor()
    workflow = _write_workflow(
        tmp_path,
        "contact_write_boundary.yml",
        """
name: Contact Write Boundary
on:
  pull_request_target:
permissions:
  contents: read
jobs:
  contact-write-boundary:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0
        with:
          ref: ${{ github.event.pull_request.base.sha }}
""",
    )

    findings = auditor.audit_workflow(workflow)
    assert [f for f in findings if f.level == "ERROR"], (
        "an enrolled job without the event-name guard must still error"
    )


def test_enrolled_job_with_write_permission_is_rejected(tmp_path: Path) -> None:
    """A trusted-base job runs with the BASE repository's token.

    Granting it a write scope hands that token to a job whose purpose is to
    read PR-authored content. Enrolment must not make that shape admissible.
    """
    auditor = load_auditor()
    workflow = _write_workflow(
        tmp_path,
        "contact_write_boundary.yml",
        """
name: Contact Write Boundary
on:
  pull_request_target:
permissions:
  contents: write
jobs:
  contact-write-boundary:
    if: github.event_name == 'pull_request_target'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0
        with:
          ref: ${{ github.event.pull_request.base.sha }}
""",
    )

    findings = auditor.audit_workflow(workflow)
    assert [f for f in findings if f.level == "ERROR"], (
        "a write scope on an enrolled trusted-base job must not be admitted"
    )


def test_enrolled_job_with_job_level_write_permission_is_rejected(
    tmp_path: Path,
) -> None:
    """Job-level permissions override the workflow block, so check both."""
    auditor = load_auditor()
    workflow = _write_workflow(
        tmp_path,
        "contact_write_boundary.yml",
        """
name: Contact Write Boundary
on:
  pull_request_target:
permissions:
  contents: read
jobs:
  contact-write-boundary:
    if: github.event_name == 'pull_request_target'
    permissions:
      pull-requests: write
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0
        with:
          ref: ${{ github.event.pull_request.base.sha }}
""",
    )

    findings = auditor.audit_workflow(workflow)
    assert [f for f in findings if f.level == "ERROR"], (
        "a job-level write scope must not be admitted"
    )


def test_enrolled_job_omitting_permissions_entirely_is_rejected(
    tmp_path: Path,
) -> None:
    """Absence of a permissions block is not evidence of read-only.

    With no block at either scope the job inherits the repository/organization
    default for `GITHUB_TOKEN`, which is write-capable on older repositories and
    is configured outside this repository entirely. The predicate must fail
    closed rather than read the omission as safe.
    """
    auditor = load_auditor()
    workflow = _write_workflow(
        tmp_path,
        "contact_write_boundary.yml",
        """
name: Contact Write Boundary
on:
  pull_request_target:
jobs:
  contact-write-boundary:
    if: github.event_name == 'pull_request_target'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0
        with:
          ref: ${{ github.event.pull_request.base.sha }}
""",
    )

    findings = auditor.audit_workflow(workflow)
    assert [f for f in findings if f.level == "ERROR"], (
        "an enrolled trusted-base job with no permissions block must not be admitted"
    )


@pytest.mark.parametrize(
    "permissions, expected",
    [
        ({"contents": "read"}, True),
        ({"contents": "read", "pull-requests": "read"}, True),
        ({"contents": "none"}, True),
        ({}, True),
        ("read-all", True),
        ({"contents": "read", "id-token": "write"}, True),
        (None, False),
        ("write-all", False),
        ({"contents": "write"}, False),
        ({"contents": "read", "pull-requests": "write"}, False),
        ("read", False),
        ({"contents": "${{ inputs.scope }}"}, False),
    ],
)
def test_permissions_read_only_predicate_boundaries(
    permissions: object, expected: bool
) -> None:
    """Both sides of the guard, including the shapes it cannot evaluate.

    The `${{ }}` and bare-scalar rows matter most: an unresolved expression is
    not statically provable as read-only, so it must land on the reject side
    rather than falling through an `isinstance` check into admission.
    """
    auditor = load_auditor()
    assert auditor._permissions_are_explicitly_read_only(permissions) is expected


def test_every_currently_enrolled_job_declares_read_only_permissions() -> None:
    """The new rule rejects nothing that exists today.

    Pinned so that adding a write scope to any enrolled gate -- or dropping its
    permissions block so it silently inherits the repo default -- is a
    deliberate, visibly failing change rather than a silent widening.
    """
    auditor = load_auditor()
    workflows = Path(__file__).resolve().parents[1] / ".github" / "workflows"
    checked = 0
    for filename, job_name in sorted(auditor.ALLOWED_PULL_REQUEST_TARGET_JOBS):
        path = workflows / filename
        if not path.exists():
            continue  # arrives in a later PR
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        job = (document.get("jobs") or {}).get(job_name) or {}
        effective = auditor._effective_permissions(job, document)
        assert auditor._permissions_are_explicitly_read_only(effective), (
            filename,
            job_name,
            effective,
        )
        checked += 1
    assert checked, "no enrolled workflow was actually checked"


# --- Malformed permission-value shapes (post-merge finding on ATLAS #2305) ---
#
# The first version skipped `id-token` by KEY before validating its VALUE, and
# compared other values against a frozenset. Two consequences, both real:
#   `{id-token: [write]}` -> admitted as read-only AND missed by the OIDC check,
#                            because that check compared only with the scalar.
#   `{contents: [read]}`  -> TypeError: unhashable type, so the auditor crashed
#                            instead of returning a verdict.


@pytest.mark.parametrize(
    "permissions",
    [
        {"id-token": ["write"]},
        {"id-token": {"nested": "write"}},
        {"contents": ["read"]},
        {"contents": {"nested": "read"}},
        {"contents": True},
        {"contents": 1},
        {"contents": None},
        {5: "read"},
        {"contents": "read", "pull-requests": ["read"]},
    ],
)
def test_malformed_permission_shapes_are_rejected_not_crashed(
    permissions: object,
) -> None:
    """Every unevaluable shape must reach the reject verdict.

    Reject, specifically -- not raise. A crash inside the predicate is not a
    safe failure just because the process exits non-zero: it stops the audit
    before later workflows are examined, and it contradicts the documented
    contract that unrecognized shapes fall on the reject side.
    """
    auditor = load_auditor()
    assert auditor._permissions_are_explicitly_read_only(permissions) is False


def test_non_scalar_id_token_is_treated_as_a_write_request() -> None:
    """The OIDC check governs every workflow, not just trusted-base ones.

    `{id-token: [write]}` previously escaped it entirely, because the check
    compared against the scalar `"write"` and a list is not that scalar. An
    unevaluable value must count as a write request rather than as absence.
    """
    auditor = load_auditor()
    assert auditor._permissions_write_oidc({"id-token": ["write"]}) is True
    assert auditor._permissions_write_oidc({"id-token": {"a": "b"}}) is True
    # The legitimate shapes must be unchanged.
    assert auditor._permissions_write_oidc({"id-token": "write"}) is True
    assert auditor._permissions_write_oidc({"id-token": "none"}) is False
    assert auditor._permissions_write_oidc({"contents": "read"}) is False


def test_enrolled_job_with_list_valued_id_token_is_rejected(tmp_path: Path) -> None:
    """End-to-end: the exact shape that passed both guards before.

    A predicate-level assertion alone would not prove the audit rejects the
    workflow, since admission and the OIDC finding are separate paths.
    """
    auditor = load_auditor()
    workflow = _write_workflow(
        tmp_path,
        "contact_write_boundary.yml",
        """
name: Contact Write Boundary
on:
  pull_request_target:
permissions:
  contents: read
  id-token:
    - write
jobs:
  contact-write-boundary:
    if: github.event_name == 'pull_request_target'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0
        with:
          ref: ${{ github.event.pull_request.base.sha }}
""",
    )

    findings = auditor.audit_workflow(workflow)
    errors = [f for f in findings if f.level == "ERROR"]

    # This fixture trips TWO independent branches: the workflow-scope
    # id-token invalid finding, and the trusted-base admission rejection.
    # Asserting only "some error exists" let either one carry the test, so
    # deleting the OIDC branch kept it green -- verified by deleting it.
    # Each branch is therefore asserted by its own message.
    assert any('id-token must be the string "write" or "none"' in f.detail for f in errors), errors
    assert any("trusted-base guard shape" in f.detail for f in errors), errors


# --- OIDC tri-state: "unevaluable" must not be allowlistable -----------------
#
# Collapsing "requests write" and "cannot be evaluated" into one boolean let a
# malformed value inherit the Claude job's allowlist and downgrade to a WARN,
# indistinguishable from the reviewed `id-token: write`. The allowlist permits a
# KNOWN value on a KNOWN job; a shape nobody can evaluate is not that value.


def _claude_workflow(id_token_literal: str) -> str:
    return f"""
name: Claude
on:
  workflow_dispatch:
permissions:
  contents: read
jobs:
  claude:
    if: github.actor == github.repository_owner
    runs-on: ubuntu-latest
    permissions:
      id-token: {id_token_literal}
    steps:
      - uses: actions/checkout@9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0
"""


@pytest.mark.parametrize(
    "id_token_literal", ["\n        - write", "\n        a: b", "\n        - read"]
)
def test_malformed_id_token_errors_even_on_the_allowlisted_job(
    tmp_path: Path, id_token_literal: str
) -> None:
    """The allowlisted identity must not launder an unevaluable value."""
    auditor = load_auditor()
    workflow = _write_workflow(tmp_path, "claude.yml", _claude_workflow(id_token_literal))

    findings = auditor.audit_workflow(workflow)

    assert [f for f in findings if f.level == "ERROR"], findings
    assert not [
        f for f in findings if f.level == "WARN" and "allowed id-token" in f.detail
    ], "an unevaluable value must not be reported as an allowed one"


def test_valid_id_token_write_on_the_allowlisted_job_is_still_only_a_warning(
    tmp_path: Path,
) -> None:
    """The control. Without it, the parametrization above would also pass if the
    fix simply errored on every id-token value, which would break the Claude
    Code action outright."""
    auditor = load_auditor()
    workflow = _write_workflow(tmp_path, "claude.yml", _claude_workflow("write"))

    findings = auditor.audit_workflow(workflow)

    assert not [f for f in findings if f.level == "ERROR"], findings
    assert any(f.level == "WARN" and "allowed id-token" in f.detail for f in findings)


@pytest.mark.parametrize(
    "permissions, expected",
    [
        ({"id-token": "write"}, "write"),
        ({"id-token": "none"}, "none"),
        ({"id-token": ["write"]}, "invalid"),
        ({"id-token": {"a": "b"}}, "invalid"),
        ({"id-token": True}, "invalid"),
        ({"contents": "read"}, "none"),
        ("write-all", "write"),
        ("read-all", "none"),
        (None, "none"),
        ({}, "none"),
    ],
)
def test_oidc_state_tri_state_boundaries(permissions: object, expected: str) -> None:
    auditor = load_auditor()
    assert auditor._permissions_oidc_state(permissions) == expected



def test_workflow_scope_invalid_id_token_errors_on_its_own(tmp_path: Path) -> None:
    """The workflow-scope branch in isolation, with nothing else wrong.

    The combined fixture above can only prove both branches fire together. This
    one removes every other reason to fail -- ordinary event, no allowlisted
    identity, correct job shape -- so the invalid-value finding is the only
    thing that can produce an error.
    """
    auditor = load_auditor()
    workflow = _write_workflow(
        tmp_path,
        "some_ordinary_workflow.yml",
        """
name: Ordinary
on:
  workflow_dispatch:
permissions:
  contents: read
  id-token:
    - write
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0
""",
    )

    findings = auditor.audit_workflow(workflow)
    errors = [f for f in findings if f.level == "ERROR"]

    assert len(errors) == 1, errors
    assert 'workflow-scope id-token must be the string "write" or "none"' in errors[0].detail


@pytest.mark.parametrize(
    "value, expected",
    [
        ("write", "write"),
        ("none", "none"),
        ("read", "invalid"),
        ("", "invalid"),
        ("${{ inputs.scope }}", "invalid"),
        ("WRITE", "invalid"),
        ("Write", "invalid"),
        ("write ", "invalid"),
    ],
)
def test_id_token_vocabulary_is_closed(value: str, expected: str) -> None:
    """Only `write` and `none` are recognized; every other scalar is invalid.

    The catch-all previously returned `none`, so an unresolved `${{ }}`
    expression, an empty string, or a case variant read as "no OIDC request"
    and skipped the allowlist. Absence and unevaluable are different answers.
    """
    auditor = load_auditor()
    assert auditor._permissions_oidc_state({"id-token": value}) == expected



@pytest.mark.parametrize("value", ["read", "WRITE", "${{ inputs.scope }}", ""])
def test_unsupported_scalar_id_token_reports_the_vocabulary_not_the_shape(
    tmp_path: Path, value: str
) -> None:
    """An unsupported scalar must not be described as a non-scalar.

    `read` and `WRITE` are perfectly well-formed scalars. Reporting them as
    "not a scalar" sent the operator looking for malformed YAML when the actual
    problem was the vocabulary, which is the more expensive kind of wrong
    message: it is confidently specific and points the wrong way.
    """
    auditor = load_auditor()
    workflow = _write_workflow(
        tmp_path,
        "ordinary.yml",
        f"""
name: Ordinary
on:
  workflow_dispatch:
permissions:
  contents: read
  id-token: "{value}"
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0
""",
    )

    errors = [f for f in auditor.audit_workflow(workflow) if f.level == "ERROR"]

    assert len(errors) == 1, errors
    assert 'must be the string "write" or "none"' in errors[0].detail
    assert "not a scalar" not in errors[0].detail



@pytest.mark.parametrize("value", ["read", "WRITE", "${{ inputs.scope }}", ""])
def test_job_scope_unsupported_scalar_reports_the_vocabulary(
    tmp_path: Path, value: str
) -> None:
    """The job-scope twin of the workflow-scope diagnostic test.

    The workflow-scope case was covered; the job-scope one was not. Existing
    job-scope tests use list/mapping values and assert only that some error
    exists, so reverting the job message to the false "is not a scalar" wording
    left every test green while the plan claimed both diagnostics were pinned.
    """
    auditor = load_auditor()
    workflow = _write_workflow(
        tmp_path,
        "ordinary.yml",
        f"""
name: Ordinary
on:
  workflow_dispatch:
permissions:
  contents: read
jobs:
  build:
    runs-on: ubuntu-latest
    permissions:
      id-token: "{value}"
    steps:
      - uses: actions/checkout@9c091bb21b7c1c1d1991bb908d89e4e9dddfe3e0
""",
    )

    errors = [f for f in auditor.audit_workflow(workflow) if f.level == "ERROR"]

    assert len(errors) == 1, errors
    assert 'job build id-token must be the string "write" or "none"' in errors[0].detail
    assert "not a scalar" not in errors[0].detail
