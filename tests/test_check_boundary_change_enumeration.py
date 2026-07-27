from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "check_boundary_change_enumeration",
    Path(__file__).resolve().parent.parent / "scripts" / "check_boundary_change_enumeration.py",
)
mod = importlib.util.module_from_spec(_SPEC)
assert _SPEC and _SPEC.loader
sys.modules[_SPEC.name] = mod
_SPEC.loader.exec_module(mod)

BOUNDARY_CHANGE = (
    "+def resolve_contact_identity(payload):\n"
    "+    return payload.get('email') or payload.get('phone')\n"
)
TS_BOUNDARY_CHANGE = (
    "+export function validateClaimGate(input) {\n"
    "+  return Boolean(input.accountId)\n"
    "+}\n"
)
TS_ALLOWED_BOUNDARY_CHANGE = (
    "@@ -25,6 +25,7 @@ function isProductClaimAllowed(product, account) {\n"
    "+  return Boolean(account?.id && product.claimable)\n"
)
PY_CLASSIFY_ROUTE_CHANGE = (
    "@@ -199,6 +199,7 @@ def classify_and_route(state):\n"
    "+    return 'booking' if state.get('intent') == 'book' else 'fallback'\n"
)
PY_SHOULD_SCRAPE_CHANGE = (
    "@@ -723,6 +723,7 @@ async def should_scrape_now(candidate):\n"
    "+    return candidate.manual or candidate.window_open\n"
)
TWO_SEAMS_CHANGE = (
    "+def resolve_contact_identity(payload):\n"
    "+    return payload.get('email')\n"
    "+def resolve_tenant_identity(payload):\n"
    "+    return payload.get('tenant_id')\n"
)
PATH_ONLY_BOUNDARY_CHANGE = "+VALUE = 1\n"
CONST_BOUNDARY_CHANGE = "+const resolveClaimGate = (input) => Boolean(input.accountId)\n"
CLASS_BOUNDARY_CHANGE = "+class TenantResolver {\n+  resolve(input) { return input.tenantId }\n+}\n"
SHELL_BOUNDARY_CHANGE = "+validate_scope() {\n+  test -n \"$ATLAS_SCOPE\"\n+}\n"
TS_METHOD_BOUNDARY_CHANGE = "+  validateFileType(buffer, expectedType) {\n+    return true\n+  }\n"
TS_PRIVATE_METHOD_BOUNDARY_CHANGE = (
    "@@ -671,6 +671,7 @@ export class DocumentService {\n"
    "+  private validateFile(file) {\n"
    "+    return file.size < 1000\n"
    "+  }\n"
)
TS_RETURN_ANNOTATED_METHOD_BOUNDARY_CHANGE = (
    "@@ -671,6 +671,7 @@ export class DocumentService {\n"
    "+  private validateFile(file): void {\n"
    "+    return file.size < 1000\n"
    "+  }\n"
)
TS_RETURN_ANNOTATED_FUNCTION_BOUNDARY_CHANGE = (
    "+export function validateFileType(buffer, expectedType): boolean {\n"
    "+  return Boolean(buffer && expectedType)\n"
    "+}\n"
)
TWO_CLASS_SAME_SEAM_CHANGE = (
    "+class CustomerValidator {\n"
    "+  validate(input): boolean {\n"
    "+    return Boolean(input.customerId)\n"
    "+  }\n"
    "+}\n"
    "+class TenantValidator {\n"
    "+  validate(input): boolean {\n"
    "+    return Boolean(input.tenantId)\n"
    "+  }\n"
    "+}\n"
)
TS_NORMALIZER_CHANGE = "+function normalizeLandingPageRepairAttemptValue(value) {\n+  return Number(value)\n+}\n"
ROUTING_TABLE_CHANGE = "+ROUTE_TO_ACTION = {\n+    'book': 'schedule_visit',\n+}\n"
REMOVED_BOUNDARY_CHANGE = "-def resolve_contact(row):\n-    return row.email\n"
CLAIM_SERIALIZER_CHANGE = "+def _serialize_claim(row):\n+    return {'claim': row.claim}\n"
AUTH_ERROR_CHANGE = "+class RedditAuthError(RuntimeError):\n+    pass\n"
SEO_RESOLVED_IMAGE_ASSIGNMENT = "+const resolvedImage = image ?? defaultImage\n"
PLAN_WITH_ENUMERATION = """
### Boundary-change enumeration

- Boundary path: atlas_brain/services/crm_provider.py.
- Replaced-path behaviors: preserved existing email-first lookup.
- Guard-relevant fields: email, phone, source_ref.
- Caller x input shape: intake x email-only -> preserved.
"""
SCAFFOLD_PLACEHOLDER = """
### Boundary-change enumeration

- Replaced-path behaviors: TODO/N/A.
- Guard-relevant fields: TODO/N/A.
- Caller x input shape: TODO/N/A.
"""


def test_boundary_change_without_plan_enumeration_is_flagged() -> None:
    findings = mod.scan_diff({"atlas_brain/services/crm_provider.py": BOUNDARY_CHANGE}, [])
    assert len(findings) == 1
    assert findings[0].path == "atlas_brain/services/crm_provider.py"
    assert mod.RULE in findings[0].reason


def test_typescript_boundary_change_without_plan_enumeration_is_flagged() -> None:
    findings = mod.scan_diff({"atlas-churn-ui/src/components/ProductClaimGate.tsx": TS_BOUNDARY_CHANGE}, [])
    assert len(findings) == 1
    assert findings[0].path == "atlas-churn-ui/src/components/ProductClaimGate.tsx"


def test_allowed_product_gate_without_plan_enumeration_is_flagged() -> None:
    findings = mod.scan_diff(
        {"atlas-churn-ui/src/components/ProductClaimGate.tsx": TS_ALLOWED_BOUNDARY_CHANGE},
        [],
    )
    assert len(findings) == 1


def test_classify_and_route_without_plan_enumeration_is_flagged() -> None:
    findings = mod.scan_diff(
        {"atlas_brain/agents/graphs/atlas.py": PY_CLASSIFY_ROUTE_CHANGE},
        [],
    )
    assert len(findings) == 1


def test_should_scrape_eligibility_without_plan_enumeration_is_flagged() -> None:
    findings = mod.scan_diff(
        {"atlas_brain/services/scraping/eligibility.py": PY_SHOULD_SCRAPE_CHANGE},
        [],
    )
    assert len(findings) == 1


def test_boundary_path_signal_without_plan_enumeration_is_flagged() -> None:
    findings = mod.scan_diff({"atlas_brain/services/admission_gate.py": PATH_ONLY_BOUNDARY_CHANGE}, [])
    assert len(findings) == 1


def test_const_boundary_signal_without_plan_enumeration_is_flagged() -> None:
    findings = mod.scan_diff({"atlas-churn-ui/src/components/ClaimPanel.tsx": CONST_BOUNDARY_CHANGE}, [])
    assert len(findings) == 1


def test_class_boundary_signal_without_plan_enumeration_is_flagged() -> None:
    findings = mod.scan_diff({"portfolio-ui/src/auth.ts": CLASS_BOUNDARY_CHANGE}, [])
    assert len(findings) == 1


def test_shell_boundary_signal_without_plan_enumeration_is_flagged() -> None:
    findings = mod.scan_diff({"scripts/validate_extracted_content_pipeline.sh": SHELL_BOUNDARY_CHANGE}, [])
    assert len(findings) == 1


def test_parameterized_typescript_method_without_plan_enumeration_is_flagged() -> None:
    findings = mod.scan_diff({"lib/graphrag/parsers/index.ts": TS_METHOD_BOUNDARY_CHANGE}, [])
    assert len(findings) == 1


def test_access_modified_typescript_method_without_plan_enumeration_is_flagged() -> None:
    findings = mod.scan_diff(
        {"lib/graphrag/service/document-service.ts": TS_PRIVATE_METHOD_BOUNDARY_CHANGE},
        [],
    )
    assert len(findings) == 1


def test_return_annotated_typescript_methods_without_plan_enumeration_are_flagged() -> None:
    findings = mod.scan_diff(
        {
            "lib/graphrag/service/document-service.ts": TS_RETURN_ANNOTATED_METHOD_BOUNDARY_CHANGE,
            "lib/graphrag/parsers/index.ts": TS_RETURN_ANNOTATED_FUNCTION_BOUNDARY_CHANGE,
        },
        [],
    )
    assert [finding.path for finding in findings] == [
        "lib/graphrag/parsers/index.ts",
        "lib/graphrag/service/document-service.ts",
    ]


def test_normalizing_decision_seam_without_plan_enumeration_is_flagged() -> None:
    findings = mod.scan_diff({"atlas-intel-ui/src/pages/ContentOpsNewRun.tsx": TS_NORMALIZER_CHANGE}, [])
    assert len(findings) == 1


def test_routing_decision_seam_without_plan_enumeration_is_flagged() -> None:
    findings = mod.scan_diff({"atlas_brain/services/intent_router.py": ROUTING_TABLE_CHANGE}, [])
    assert len(findings) == 1


def test_removed_boundary_declaration_without_plan_enumeration_is_flagged() -> None:
    findings = mod.scan_diff({"scripts/sync_eom_portal_customers.py": REMOVED_BOUNDARY_CHANGE}, [])
    assert len(findings) == 1


def test_boundary_change_with_plan_enumeration_is_clean() -> None:
    findings = mod.scan_diff(
        {"atlas_brain/services/crm_provider.py": BOUNDARY_CHANGE},
        [PLAN_WITH_ENUMERATION],
    )
    assert findings == []


def test_boundary_change_with_exact_seam_enumeration_is_clean() -> None:
    seam_plan = """
### Boundary-change enumeration

- Boundary path/seam: resolve_contact_identity.
- Replaced-path behaviors: preserved existing email-first lookup.
- Guard-relevant fields: preserved email, phone, source_ref.
- Caller x input shape: preserved intake x email-only.
"""
    findings = mod.scan_diff(
        {"atlas_brain/services/crm_provider.py": BOUNDARY_CHANGE},
        [seam_plan],
    )
    assert findings == []


def test_non_boundary_change_is_clean() -> None:
    findings = mod.scan_diff({"atlas_brain/api/health.py": "+def ping():\n+    return {'ok': True}\n"}, [])
    assert findings == []


def test_claim_serializer_lexical_lookalike_is_clean() -> None:
    findings = mod.scan_diff({"atlas_brain/api/b2b_vendor_claims.py": CLAIM_SERIALIZER_CHANGE}, [])
    assert findings == []


def test_auth_exception_lexical_lookalike_is_clean() -> None:
    findings = mod.scan_diff({"atlas_reddit/reddit_client.py": AUTH_ERROR_CHANGE}, [])
    assert findings == []


def test_token_bearing_local_variable_assignment_is_clean() -> None:
    findings = mod.scan_diff(
        {"atlas-intel-ui/src/components/SeoHead.tsx": SEO_RESOLVED_IMAGE_ASSIGNMENT},
        [],
    )
    assert findings == []


def test_process_detector_change_is_clean_with_na_plan() -> None:
    findings = mod.scan_diff({"scripts/check_boundary_change_enumeration.py": BOUNDARY_CHANGE}, [])
    assert findings == []


def test_other_checker_boundary_change_is_not_exempt() -> None:
    findings = mod.scan_diff({"scripts/check_guard_class_closure.py": BOUNDARY_CHANGE}, [])
    assert len(findings) == 1


def test_test_file_boundary_words_are_ignored() -> None:
    findings = mod.scan_diff({"tests/test_contact_resolver.py": BOUNDARY_CHANGE}, [])
    assert findings == []


def test_mjs_test_file_boundary_words_are_ignored() -> None:
    findings = mod.scan_diff({
        "atlas-intel-ui/scripts/content-ops-ingestion-routing.test.mjs": PATH_ONLY_BOUNDARY_CHANGE,
        "atlas-intel-ui/scripts/content-ops-ingestion-routing.spec.cjs": PATH_ONLY_BOUNDARY_CHANGE,
    }, [])
    assert findings == []


def test_plan_requires_all_enumeration_markers() -> None:
    incomplete = "### Boundary-change enumeration\n- Replaced-path behaviors: preserved."
    narrative = (
        "## Why this slice exists\n"
        "replaced-path behaviors documented; guard-relevant fields documented; "
        "caller x input shape documented.\n"
        + SCAFFOLD_PLACEHOLDER
    )
    assert not mod.plan_has_boundary_enumeration(incomplete)
    assert not mod.plan_has_boundary_enumeration(SCAFFOLD_PLACEHOLDER)
    assert not mod.plan_has_boundary_enumeration(narrative)
    assert mod.plan_has_boundary_enumeration(PLAN_WITH_ENUMERATION)


def test_reasoned_not_applicable_dispositions_are_clean() -> None:
    not_applicable = """
### Boundary-change enumeration

- Replaced-path behaviors: N/A - no boundary behavior is replaced.
- Guard-relevant fields: not applicable - no guard verdict fields.
- Caller x input shape: N/A - no caller can reach a boundary.
    """
    assert mod.plan_has_boundary_enumeration(not_applicable)


def test_section_level_not_applicable_reason_covers_bare_na_rows() -> None:
    not_applicable = """
### Boundary-change enumeration

N/A - no boundary change; detector path matched a process-only helper.

- Boundary path/seam: atlas_brain/services/admission_gate.py.
- Replaced-path behaviors: N/A.
- Guard-relevant fields: N/A.
- Caller x input shape: N/A.
"""
    assert mod.plan_has_boundary_enumeration(not_applicable)
    assert (
        mod.scan_diff(
            {"atlas_brain/services/admission_gate.py": PATH_ONLY_BOUNDARY_CHANGE},
            [not_applicable],
        )
        == []
    )


def test_section_level_not_applicable_does_not_cover_unlisted_boundary() -> None:
    not_applicable = """
### Boundary-change enumeration

N/A - no boundary change; detector path matched a process-only helper.

- Boundary path/seam: atlas_brain/services/other_gate.py.
- Replaced-path behaviors: N/A.
- Guard-relevant fields: N/A.
- Caller x input shape: N/A.
"""
    findings = mod.scan_diff(
        {"atlas_brain/services/admission_gate.py": PATH_ONLY_BOUNDARY_CHANGE},
        [not_applicable],
    )
    assert [finding.path for finding in findings] == ["atlas_brain/services/admission_gate.py"]


def test_duplicate_enumeration_rows_must_all_be_dispositioned() -> None:
    duplicate_todo = """
### Boundary-change enumeration

- Replaced-path behaviors: preserved legacy lookup.
- Replaced-path behaviors: TODO disposition replacement fallback.
- Guard-relevant fields: email, phone.
- Caller x input shape: intake x email-only -> preserved.
    """
    assert not mod.plan_has_boundary_enumeration(duplicate_todo)


def test_unresolved_disposition_words_are_not_enumeration() -> None:
    unresolved = """
### Boundary-change enumeration

- Boundary path: atlas_brain/services/crm_resolver.py.
- Replaced-path behaviors: TBD after caller audit.
- Guard-relevant fields: unknown.
- Caller x input shape: pending review.
"""
    assert not mod.plan_has_boundary_enumeration(unresolved)


def test_each_changed_boundary_requires_its_own_path_enumeration() -> None:
    crm_only_plan = """
### Boundary-change enumeration

- Boundary path: atlas_brain/services/crm_resolver.py.
- Replaced-path behaviors: crm_resolver.py preserves existing CRM lookup.
- Guard-relevant fields: crm_resolver.py email.
- Caller x input shape: crm_resolver.py intake x email-only -> preserved.
"""
    findings = mod.scan_diff(
        {
            "atlas_brain/services/crm_resolver.py": BOUNDARY_CHANGE,
            "atlas_brain/services/tenant_resolver.py": BOUNDARY_CHANGE,
        },
        [crm_only_plan],
    )
    assert [finding.path for finding in findings] == ["atlas_brain/services/tenant_resolver.py"]


def test_each_boundary_path_owns_its_own_disposition_group() -> None:
    shared_dispositions_plan = """
### Boundary-change enumeration

- Boundary path: atlas_brain/services/crm_resolver.py.
- Replaced-path behaviors: CRM preserves existing lookup.
- Guard-relevant fields: CRM email.
- Caller x input shape: CRM intake x email-only -> preserved.
- Boundary path: atlas_brain/services/tenant_resolver.py.
"""
    findings = mod.scan_diff(
        {
            "atlas_brain/services/crm_resolver.py": BOUNDARY_CHANGE,
            "atlas_brain/services/tenant_resolver.py": BOUNDARY_CHANGE,
        },
        [shared_dispositions_plan],
    )
    assert [finding.path for finding in findings] == ["atlas_brain/services/tenant_resolver.py"]


def test_each_changed_seam_in_one_file_requires_coverage() -> None:
    one_seam_plan = """
### Boundary-change enumeration

- Boundary path/seam: resolve_contact_identity.
- Replaced-path behaviors: preserved contact lookup.
- Guard-relevant fields: preserved email.
- Caller x input shape: preserved intake contact payload.
"""
    findings = mod.scan_diff(
        {"atlas_brain/services/crm_resolver.py": TWO_SEAMS_CHANGE},
        [one_seam_plan],
    )
    assert [finding.path for finding in findings] == ["atlas_brain/services/crm_resolver.py"]


def test_each_changed_seam_in_one_file_can_be_dispositioned() -> None:
    two_seam_plan = """
### Boundary-change enumeration

- Boundary path/seam: resolve_contact_identity.
- Replaced-path behaviors: preserved contact lookup.
- Guard-relevant fields: preserved email.
- Caller x input shape: preserved intake contact payload.
- Boundary path/seam: resolve_tenant_identity.
- Replaced-path behaviors: preserved tenant lookup.
- Guard-relevant fields: preserved tenant_id.
- Caller x input shape: preserved intake tenant payload.
"""
    findings = mod.scan_diff(
        {"atlas_brain/services/crm_resolver.py": TWO_SEAMS_CHANGE},
        [two_seam_plan],
    )
    assert findings == []


def test_same_named_methods_in_distinct_classes_need_distinct_boundary_entries() -> None:
    bare_validate_plan = """
### Boundary-change enumeration

- Boundary path/seam: validate.
- Replaced-path behaviors: preserved validation behavior.
- Guard-relevant fields: preserved id fields.
- Caller x input shape: preserved callers.
"""
    findings = mod.scan_diff(
        {"atlas-churn-ui/src/validators/customer.ts": TWO_CLASS_SAME_SEAM_CHANGE},
        [bare_validate_plan],
    )
    assert [finding.path for finding in findings] == ["atlas-churn-ui/src/validators/customer.ts"]


def test_same_named_methods_in_distinct_classes_can_be_dispositioned_by_qualified_entry() -> None:
    qualified_plan = """
### Boundary-change enumeration

- Boundary path/seam: CustomerValidator.validate.
- Replaced-path behaviors: preserved customer validation behavior.
- Guard-relevant fields: preserved customerId.
- Caller x input shape: preserved customer callers.
- Boundary path/seam: TenantValidator.validate.
- Replaced-path behaviors: preserved tenant validation behavior.
- Guard-relevant fields: preserved tenantId.
- Caller x input shape: preserved tenant callers.
"""
    findings = mod.scan_diff(
        {"atlas-churn-ui/src/validators/customer.ts": TWO_CLASS_SAME_SEAM_CHANGE},
        [qualified_plan],
    )
    assert findings == []


def test_boundary_path_match_requires_exact_changed_path() -> None:
    basename_only_plan = """
### Boundary-change enumeration

- Boundary path: index.ts.
- Replaced-path behaviors: preserves existing lookup.
- Guard-relevant fields: account id.
- Caller x input shape: panel x account-only -> preserved.
"""
    findings = mod.scan_diff(
        {"atlas-churn-ui/src/components/index.ts": TS_BOUNDARY_CHANGE},
        [basename_only_plan],
    )
    assert [finding.path for finding in findings] == ["atlas-churn-ui/src/components/index.ts"]


def test_cli_entrypoint_warns_advisory_and_fails_strict(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(
        mod,
        "changed_added_lines",
        lambda base: {"atlas_brain/services/crm_provider.py": BOUNDARY_CHANGE},
    )
    monkeypatch.setattr(mod, "changed_plan_texts", lambda base: [])

    assert mod.main(["--base", "ignored"]) == 0
    out = capsys.readouterr().out
    assert "::warning file=atlas_brain/services/crm_provider.py::" in out
    assert mod.RULE in out

    assert mod.main(["--base", "ignored", "--strict"]) == 1


def test_git_failure_raises_system_exit() -> None:
    with pytest.raises(SystemExit, match="git .* failed"):
        mod._git(["rev-parse", "--verify", "definitely-not-a-ref-xyz"])


def test_changed_plan_texts_reads_from_repo_root_when_called_from_subdir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_path = tmp_path / "plans" / "PR-Cwd.md"
    plan_path.parent.mkdir()
    plan_path.write_text(PLAN_WITH_ENUMERATION, encoding="utf-8")
    caller_cwd = tmp_path / "subdir"
    caller_cwd.mkdir()

    monkeypatch.setattr(mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(mod, "_git", lambda args: "plans/PR-Cwd.md\n")
    monkeypatch.chdir(caller_cwd)

    assert mod.changed_plan_texts("ignored") == [PLAN_WITH_ENUMERATION]
