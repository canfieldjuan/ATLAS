# PR-Deflection-HTTP-Error-Source-ID-Values

## Why this slice exists

#1728 closed the shared HTTP error-body redaction gap, but its LGTM left a
non-blocking NIT: submit-handoff can still persist an opaque
`source_id` / `source_ids` JSON value when the value has no vendor prefix, row
prefix, ticket prefix, or long numeric run. That means an error response like
`{"source_id": "opaque7key"}` is sanitized for tokens and URLs but can still
store the opaque source id in proof output.

Root cause: the shared HTTP helper's structural JSON redaction is value-only
and context-free. The submit caller can redact string values by pattern, but it
cannot know that a short opaque string is sensitive when it appears under the
`source_id` or `source_ids` keys. Adding more denylist regexes would either
miss the next opaque shape or over-redact unrelated short strings.

This PR fixes the root for operator proof output by adding an optional
key-path-aware JSON value redaction hook at the shared helper seam, then using
it only in submit-handoff's HTTP error-body path. It intentionally does not
change backend paid artifact storage or buyer report source-id policy; #1730 is
already carrying that separate backend PII/source-id tradeoff.

## Scope (this PR)

Ownership lane: content-ops/report-delivery-live-funnel
Slice phase: Production hardening

1. Add an optional key-path-aware JSON value redaction hook for HTTP error
   bodies in the deflection HTTP helper.
2. Wire submit-handoff to redact scalar values under `source_id` and
   `source_ids` in HTTP error JSON payloads, including short opaque values and
   numeric source IDs.
3. Add focused negative and near-miss tests proving keyed source-id values are
   redacted without broad regex overreach.

### Files touched

- `plans/PR-Deflection-HTTP-Error-Source-ID-Values.md`
- `scripts/_deflection_http.py`
- `scripts/smoke_content_ops_deflection_submit_handoff.py`
- `tests/test_deflection_http_helpers.py`
- `tests/test_smoke_content_ops_deflection_submit_handoff.py`

### Review Contract

- Acceptance criteria:
  - [ ] Submit-handoff HTTP error JSON redacts scalar values under `source_id`
        and `source_ids`, even when they are opaque short strings or numeric
        IDs.
  - [ ] Non-source-id short strings in the same payload are not redacted solely
        because they are short or opaque.
  - [ ] Existing token, content-ops id, email, signed URL, and numeric JSON
        preservation behavior from #1728 remains intact.
  - [ ] The change is limited to HTTP error-body proof output and does not
        change backend paid artifact persistence or buyer-facing source-id
        semantics.
- Affected surfaces: deflection operator HTTP helper, submit-handoff proof
  output, extracted pipeline CI enrollment.
- Risk areas: over-redacting unrelated diagnostics, changing successful
  response payloads, colliding with #1730 backend source-id policy, JSON type
  corruption.
- Reviewer rules triggered: R1, R2, R3, R8, R10, R12, R14.

## Mechanism

Extend the helper's HTTP error-body JSON redaction path with an optional
JSON-value callback that receives the current key path and value. The default
is absent, so existing callers keep the #1728 behavior: recurse through JSON,
redact string values with the plain text redactor, and preserve non-string
values.

Submit-handoff passes a small key-aware callback that replaces scalar values
under `source_id` and `source_ids` with `[source-id-redacted]`. The normal text
redactor still handles bearer tokens, content-ops request ids, emails,
source-id shaped strings, signed URL query strings, and long numeric string
values.

Tests mock the HTTP boundary and assert both sides of the guard: opaque keyed
source-id values are removed from `payload` and `raw_text`, while a same-shaped
short diagnostic value under an unrelated key survives.

## Intentional

- This is operator-output-only. It does not touch `content_ops_deflection_reports`
  persistence, report artifacts, snapshots, or buyer report traceability.
- The fix is key-aware instead of another broad regex. Opaque source ids are
  sensitive because of their field context; unrelated short diagnostics are not.
- Successful submit responses remain unchanged so request ids still support the
  same-request snapshot/artifact probes.

## Deferred

- Backend paid-artifact source-id policy remains with #1730. This slice does
  not decide whether buyer-visible paid artifacts should redact, pseudonymize,
  or preserve source IDs.

Parked hardening: none.

## Verification

- Command passed: python -m py_compile scripts/_deflection_http.py scripts/smoke_content_ops_deflection_submit_handoff.py tests/test_deflection_http_helpers.py tests/test_smoke_content_ops_deflection_submit_handoff.py.
- Command passed: pytest tests/test_deflection_http_helpers.py tests/test_smoke_content_ops_deflection_submit_handoff.py -q -- 59 passed.
- Command passed: python scripts/audit_extracted_pipeline_ci_enrollment.py -- OK: 185 matching tests are enrolled.
- Command passed: exact maturity-sweep-deflection-content-ops workflow command -- 14 + 8 maturity tests passed; product surface manifest ok; deflection and AI content ops ratchet gates passed with no new brittleness above baseline.
- Command passed: bash scripts/run_extracted_pipeline_checks.sh -- 4702 passed, 10 skipped.
- Pending before push: scripts/push_pr.sh.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Deflection-HTTP-Error-Source-ID-Values.md` | 118 |
| `scripts/_deflection_http.py` | 41 |
| `scripts/smoke_content_ops_deflection_submit_handoff.py` | 11 |
| `tests/test_deflection_http_helpers.py` | 42 |
| `tests/test_smoke_content_ops_deflection_submit_handoff.py` | 34 |
| **Total** | **246** |
