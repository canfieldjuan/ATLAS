# PR-Deflection-Scrub-Regression

## Why this slice exists

Issue #1740 tracks a live regression in the paid deflection report privacy
boundary after #1738: the deterministic scrubber is invoked before storing the
paid artifact and free Snapshot, but several regex branches can still leak or
mis-shape PII in answer/steps prose.

Root cause: the scrubber's text patterns do not encode the intended precedence
and contextual grammar. The person-name pattern greedily admits 2-3 capitalized
tokens after a cue, so trailing support words can be swallowed or turn a valid
name into a rejected 3-token candidate. The contextual opaque-id pattern lists
the compact `{8,}` form before the segmented UUID/token form, so a dashed token
can be partially matched. The contextual identifier separator made `is`
optional but did not cover the all-digit `id is 12345678` shape safely. This
change fixes the root in the shared scrub helper rather than patching one
downstream report field.

## Scope (this PR)

Ownership lane: content-ops/deflection-privacy
Slice phase: Production hardening

1. Fix the shared deflection text scrubber so cued names, contextual dashed
   UUID/opaque identifiers, and contextual all-digit identifiers redact
   deterministically before persisted artifact/Snapshot projection.
2. Preserve safe technical references such as CVE/SKU/ISO-style tokens and
   support nouns that follow a redacted name.
3. Add regression tests for the direct scrub helper and the stored
   artifact/Snapshot gate.

### Review Contract

- Acceptance criteria:
  - [ ] Cued two-token names redact while trailing support words remain visible.
  - [ ] Cued three-token names redact without consuming the next support noun.
  - [ ] Contextual dashed UUID/opaque identifiers redact as a whole, not as a
        partial compact prefix.
  - [ ] Contextual `account/customer/id is <digits>` phrases redact the digit
        payload.
  - [ ] CVE/SKU/ISO-like technical references and ordinary non-contextual
        counts remain visible.
  - [ ] The stored artifact/Snapshot gate exercises at least one new regression
        shape so downstream persistence uses the fixed scrubber.
- Affected surfaces: extracted_content_pipeline deflection report scrubber;
  content-ops deflection report artifact/Snapshot persistence.
- Risk areas: security/privacy under-redaction; customer-visible report
  over-redaction; backcompat for source-link preservation.
- Reviewer rules triggered: R1, R2, R3, R10, R13, R14.

### Files touched

- `extracted_content_pipeline/faq_deflection_report.py`
- `plans/PR-Deflection-Scrub-Regression.md`
- `tests/test_content_ops_deflection_report.py`
- `tests/test_extracted_content_deflection_submit.py`

## Mechanism

The scrubber remains deterministic and local. This slice keeps the invocation
points unchanged:

1. `scrub_deflection_report_payload(...)` cleans the completed paid artifact.
2. `build_deflection_snapshot(...)` projects the free Snapshot from that
   scrubbed artifact.
3. The Snapshot is scrubbed again before persistence as defense in depth.

Inside `_scrub_deflection_text`, the regexes are tightened rather than
replaced. The contextual opaque identifier pattern admits the dashed/segmented
form before the compact form, so UUID-like values are consumed as one token.
The contextual branch admits all-digit IDs only when there is an explicit
identifier cue. The name branch redacts the shortest valid name candidate after
a cue and leaves subsequent support words in place. Strong name cues (`name is`
or `<role> is`) can override the reject-token list so `Jane Client` and
`John Member` do not leak just because the surname is also a support noun.
Source-link preservation uses the same supported-PII detector family as the
text scrubber, with an explicit safe-source-link exception for internal
`ticket-*`/`source-*` IDs, so source links cannot bypass labeled numeric IDs
that the text path redacts. Tests lock both the redaction cases and near-miss
preservation cases.

## Intentional

- No external NER dependency. The regression is in deterministic post-processing
  and should stay dependency-light for extracted-checks CI.
- Bare all-digit strings remain preserved unless they are already covered by an
  identifier/key context. The product still reports ordinary counts and years.
- Source-link preservation remains unchanged; source IDs are preserved unless
  they are PII-shaped by existing policy.

## Deferred

- Broader final-report action-section restructuring remains in #1612 after the
  scrub boundary is green.
- Bare UUID-without-prefix policy remains explicit but conservative here:
  contextual UUIDs redact, while unrelated bare technical tokens stay governed
  by the existing opaque-token near-miss policy.

Parked hardening: none.

## Verification

- Command passed: `python -m pytest tests/test_content_ops_deflection_report.py::test_deflection_report_payload_scrubs_local_entity_shapes tests/test_content_ops_deflection_report.py::test_deflection_report_payload_scrubs_pii_regression_shapes tests/test_content_ops_deflection_report.py::test_deflection_report_payload_preserves_local_entity_near_misses tests/test_extracted_content_deflection_submit.py::test_deflection_report_storage_gate_scrubs_supported_pii -q` -- 4 passed.
- Command passed: `python -m pytest tests/test_content_ops_deflection_report.py -q` -- 83 passed after rebasing over #1739.
- Command passed: `python -m pytest tests/test_extracted_content_deflection_submit.py -q` -- 70 passed.
- Command passed: `bash` `scripts/validate_extracted_content_pipeline.sh` -- validation passed; `forbid_hard_atlas_imports` clean.
- Command passed: `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline` -- clean.
- Command passed: `python scripts/audit_extracted_standalone.py --fail-on-debt` -- Atlas runtime import findings: 0.
- Command passed: `bash` `scripts/check_ascii_python.sh` -- ASCII check passed for extracted_content_pipeline Python files.
- Command passed: `bash extracted/_shared/scripts/sync_extracted.sh extracted_content_pipeline` -- refreshed mapped files; post-sync focused regression tests still passed.
- Command passed: `python -m pytest tests/test_content_ops_deflection_report.py::test_deflection_report_payload_scrubs_pii_regression_shapes tests/test_extracted_content_deflection_submit.py::test_deflection_report_storage_gate_scrubs_supported_pii -q` -- 2 passed.
- Command passed after local reviewer BLOCKER fix: `python - <<'PY' ... scrub_deflection_report_payload({'source_id': 'customer is Jane Smith ticket'}) ... PY` -- returned `customer is [redacted-name] ticket`.
- Command passed after local reviewer BLOCKER fix: `python -m pytest tests/test_content_ops_deflection_report.py::test_deflection_report_payload_scrubs_pii_regression_shapes tests/test_content_ops_deflection_report.py::test_deflection_report_payload_preserves_local_entity_near_misses tests/test_extracted_content_deflection_submit.py::test_deflection_report_storage_gate_scrubs_supported_pii -q` -- 3 passed.
- Command passed after local reviewer BLOCKER fix: `python -m pytest tests/test_content_ops_deflection_report.py -q` -- 83 passed after rebasing over #1739.
- Command passed after local reviewer BLOCKER fix: `python -m pytest tests/test_extracted_content_deflection_submit.py -q` -- 70 passed.
- Command passed after GitHub BLOCKER fix: direct review repros show `customer id is ISO9X4Q7ABCD`, `session token is SKU90X4Q7AB`, and `account id is HIPAA9X4Q7AB` redact, while `CVE-2021-44228`, `SKU-12345678`, `ISO-27001`, and `HIPAA2026` remain readable.
- Command passed after GitHub BLOCKER fix: `python -m pytest tests/test_content_ops_deflection_report.py::test_deflection_report_payload_scrubs_pii_regression_shapes -q` -- 1 passed.
- Command passed after GitHub BLOCKER fix: `python -m pytest tests/test_content_ops_deflection_report.py -q` -- 83 passed.
- Command passed after GitHub BLOCKER fix: `python -m pytest tests/test_extracted_content_deflection_submit.py -q` -- 70 passed.
- Command passed after GitHub BLOCKER fix: `bash` `scripts/validate_extracted_content_pipeline.sh` -- validation passed; `forbid_hard_atlas_imports` clean.
- Command passed after GitHub BLOCKER fix: `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline` -- clean.
- Command passed after GitHub BLOCKER fix: `python scripts/audit_extracted_standalone.py --fail-on-debt` -- Atlas runtime import findings: 0.
- Command passed after GitHub BLOCKER fix: `bash` `scripts/check_ascii_python.sh` -- ASCII check passed for extracted_content_pipeline Python files.
- Command passed after GitHub BLOCKER fix: `bash extracted/_shared/scripts/sync_extracted.sh extracted_content_pipeline` -- refreshed mapped files.
- Command passed after GitHub BLOCKER fix: `python -m pytest tests/test_content_ops_deflection_report.py::test_deflection_report_payload_scrubs_pii_regression_shapes tests/test_extracted_content_deflection_submit.py::test_deflection_report_storage_gate_scrubs_supported_pii -q` -- 2 passed after sync.
- Command passed after GitHub BLOCKER round 2 fix: direct repros show `source_id: "account is 1234567"` now redacts, `Requester name is Jane Client` and `customer is John Member` redact, `Customer: Reset Password` stays readable, and `ticket-4829103` stays preserved as a source link.
- Command passed after GitHub BLOCKER round 2 fix: `python -m pytest tests/test_content_ops_deflection_report.py::test_deflection_report_payload_scrubs_identifier_fields_markdown_and_keys tests/test_content_ops_deflection_report.py::test_deflection_report_payload_scrubs_pii_regression_shapes tests/test_content_ops_deflection_report.py::test_deflection_report_payload_preserves_local_entity_near_misses -q` -- 3 passed.
- Command passed after GitHub BLOCKER round 2 fix: `python -m pytest tests/test_content_ops_deflection_report.py -q` -- 83 passed.
- Command passed after GitHub BLOCKER round 2 fix: `python -m pytest tests/test_extracted_content_deflection_submit.py -q` -- 70 passed.
- Command passed after GitHub BLOCKER round 2 fix: `bash` `scripts/validate_extracted_content_pipeline.sh` -- validation passed; `forbid_hard_atlas_imports` clean.
- Command passed after GitHub BLOCKER round 2 fix: `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline` -- clean.
- Command passed after GitHub BLOCKER round 2 fix: `python scripts/audit_extracted_standalone.py --fail-on-debt` -- Atlas runtime import findings: 0.
- Command passed after GitHub BLOCKER round 2 fix: `bash` `scripts/check_ascii_python.sh` -- ASCII check passed for extracted_content_pipeline Python files.
- Command passed after GitHub BLOCKER round 2 fix: `bash extracted/_shared/scripts/sync_extracted.sh extracted_content_pipeline` -- refreshed mapped files.
- Command passed after GitHub BLOCKER round 2 fix: `python -m pytest tests/test_content_ops_deflection_report.py::test_deflection_report_payload_scrubs_identifier_fields_markdown_and_keys tests/test_content_ops_deflection_report.py::test_deflection_report_payload_scrubs_pii_regression_shapes tests/test_extracted_content_deflection_submit.py::test_deflection_report_storage_gate_scrubs_supported_pii -q` -- 3 passed after sync.

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/faq_deflection_report.py` | 120 |
| `plans/PR-Deflection-Scrub-Regression.md` | 146 |
| `tests/test_content_ops_deflection_report.py` | 66 |
| `tests/test_extracted_content_deflection_submit.py` | 15 |
| **Total** | **347** |
