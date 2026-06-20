# PR-Deflection-HTTP-Error-Body-Redaction

## Why this slice exists

#1727 centralized the deflection operator-script HTTP boundary, and its review
flagged a pre-existing sanitizer gap: HTTP error-response bodies can be stored
into `text`, `raw_text`, and parsed `payload` without passing through any
redaction hook. `smoke_content_ops_deflection_submit_handoff.py` is the risky
persisting caller because it writes result JSON and currently depends on the
response layer returning already-safe details.

Root cause: the shared HTTP helper only applies the caller-provided redactor to
transport exceptions. HTTP status errors travel a different branch, so their
response body is read, truncated, parsed, and returned before any shared
redaction boundary can run. That leaves each caller responsible for remembering
a separate whole-payload sanitizer.

This PR fixes the root within the report-delivery lane by adding a shared
HTTP-error body redaction hook at the helper seam and wiring submit-handoff to
use it. The fix is intentionally limited to HTTP error bodies so successful
submit responses still expose the request id internally for snapshot/artifact
polling.

## Scope (this PR)

Ownership lane: content-ops/report-delivery-live-funnel
Slice phase: Production hardening

1. Add an optional shared redaction hook for HTTP error-response bodies in the
   deflection HTTP helper.
2. Wire submit-handoff's JSON and multipart response parsing through that
   redaction hook before error bodies can reach `raw_text` or parsed `payload`.
3. Add focused failure-branch tests proving sensitive HTTP error body values
   are redacted while successful submit responses still preserve request ids.

### Files touched

- `plans/PR-Deflection-HTTP-Error-Body-Redaction.md`
- `scripts/_deflection_http.py`
- `scripts/smoke_content_ops_deflection_submit_handoff.py`
- `tests/test_deflection_http_helpers.py`
- `tests/test_smoke_content_ops_deflection_submit_handoff.py`

### Review Contract

- Acceptance criteria:
  - [ ] HTTP error bodies can be redacted before they are stored in `text` or
        `raw_text`.
  - [ ] JSON HTTP error bodies are redacted structurally, so string values are
        sanitized without corrupting numeric/count fields before the payload is
        returned.
  - [ ] Successful submit responses are not redacted and still preserve the
        request id needed for same-request snapshot/artifact probes.
  - [ ] Existing raw Stripe webhook byte handling and Mapping request-body
        encoding remain unchanged.
  - [ ] New failure-branch tests run in extracted checks.
- Affected surfaces: deflection operator HTTP helper, submit-handoff proof
  output, extracted pipeline CI enrollment.
- Risk areas: over-redacting successful submit payloads, malformed JSON after
  redaction, sanitizer false negatives, payment/report proof output safety.
- Reviewer rules triggered: R1, R2, R3, R8, R10, R12, R14.

## Mechanism

Extend `scripts/_deflection_http.py` with an optional error-body redactor
callback. When an HTTP error is caught, the helper first tries to parse JSON
error bodies, recursively redacts only string values, then serializes the
redacted payload for the existing response path. Non-JSON bodies still fall back
to whole-text redaction. Existing callers that do not pass the hook keep their
current behavior.

Add a submit-handoff redactor for persisted HTTP error bodies. It redacts the
same identifier classes that are unsafe in committed proof artifacts: bearer
tokens, content-ops request ids, source-id shaped values, emails, and signed
URL query strings. The submit-handoff wrapper passes that hook to both the JSON
request path and the multipart response parser.

Tests mock the transport boundary below the helper. Helper tests prove the
shared hook redacts HTTP error bodies and keeps numeric JSON fields intact.
Submit tests prove escaped signed URLs and source-id shaped values are redacted
from persisted failure responses while the success response still keeps the
request id.

## Intentional

- The hook applies only to HTTP error bodies, not all successful response
  bodies. Submit-handoff needs the successful request id for follow-up probes,
  and globally redacting success payloads would break the smoke's real flow.
- This does not add a generic artifact sanitizer framework. The narrow root is
  the shared HTTP error branch introduced by #1727, and a helper-level hook
  fixes that class without widening the slice into every proof artifact.
- Existing callers keep their behavior unless they opt into the new hook. That
  keeps this PR focused on the known persisting caller rather than silently
  changing every operator script's diagnostics.

## Deferred

- A broader proof-output sanitizer framework remains deferred until another
  caller demonstrates the same persistence risk. This slice fixes the current
  HTTP error branch and the known persisting caller.

Parked hardening: none.

## Verification

- Command passed: python -m py_compile scripts/_deflection_http.py scripts/smoke_content_ops_deflection_submit_handoff.py tests/test_deflection_http_helpers.py tests/test_smoke_content_ops_deflection_submit_handoff.py.
- Command passed: pytest tests/test_deflection_http_helpers.py tests/test_smoke_content_ops_deflection_submit_handoff.py -q -- 57 passed.
- Command passed: python scripts/audit_extracted_pipeline_ci_enrollment.py -- OK: 185 matching tests are enrolled.
- Command passed: deflection maturity ratchet command from .github/workflows/maturity_sweep_deflection_content_ops.yml -- ratchet gate passed; no new brittleness above baseline.
- Command passed: bash scripts/run_extracted_pipeline_checks.sh -- 4700 passed, 10 skipped.
- Pending before push: scripts/push_pr.sh.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Deflection-HTTP-Error-Body-Redaction.md` | 122 |
| `scripts/_deflection_http.py` | 24 |
| `scripts/smoke_content_ops_deflection_submit_handoff.py` | 26 |
| `tests/test_deflection_http_helpers.py` | 76 |
| `tests/test_smoke_content_ops_deflection_submit_handoff.py` | 88 |
| **Total** | **336** |
