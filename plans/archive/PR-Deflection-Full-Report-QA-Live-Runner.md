# PR-Deflection-Full-Report-QA-Live-Runner

## Why this slice exists

#1612's ATLAS-side deterministic harness and PDF/export validator are now
merged. The remaining backend proof gap is an operator-run live runner that can
take a real paid request, fetch the live paid JSON surfaces, validate the
operator's downloaded PDF artifact, and write only a sanitized scorecard. This
is the next slice named after #1622: the buyer hosted-result smoke belongs in
`atlas-portfolio/web`, but ATLAS still owns proving the paid artifact JSON,
evidence export, and PDF attachment agree with the persisted `deflection.v1`
model.

Root cause: the QA framework can validate local JSON/PDF inputs, but the live
operator path still has to assemble those inputs by hand. That leaves the
highest-risk proof step outside the tested harness: a live run can fetch the
wrong request, accept a locked or malformed artifact, silently use empty PDF
text, or commit raw paths/request IDs while still saying the validator passed.
This PR fixes the root at the live-runner seam by making transport, artifact
projection, PDF-text non-emptiness, scorecard execution, and redacted output one
tested script. It does not add a PDF download route because the production PDF
is delivered as an email attachment, not exposed by the ATLAS API.

The diff exceeds the soft cap because the runner and its failure-branch proof
are not separable: the script is a live transport wrapper around a leak-sensitive
validator, and the test matrix has to prove each new fail-closed branch before
the operator uses it on live paid artifacts.

Review fix root cause: the first runner pass treated `--pdf-bytes` and
`--pdf-text` as independent trusted operator inputs. The downstream validator
checked PDF bytes only for a `%PDF-` header and size, while every leak/count
assertion ran against the separate text file. A clean text file could therefore
green-light bytes containing a live result ID or customer email. The same pass
also left the final output sanitizer untested even though it is the last guard
before a committed proof. This update fixes the reachable root in this slice:
raw PDF bytes are decoded and leak-scanned before network calls, the result
marks PDF text as operator-asserted rather than byte-derived, and tests now
exercise both the byte-leak failure path and the sanitizer redaction/exit path.

Second review fix root cause: the artifact/report-model consistency guard only
checked mismatches when `artifact.report_model` was already a mapping. If the
artifact carried `report_model` as a string, null, or list, the runner silently
skipped the drift check. This update fixes that validation root directly:
`artifact.report_model` may be absent, but when present it must be a JSON
object before it can be compared to the report-model route.

## Scope (this PR)

Ownership lane: content-ops/deflection-full-report-qa
Slice phase: Functional validation

1. Add an operator-run live QA runner for paid deflection report artifacts.
2. Fetch the live `/report-model` and `/artifact` JSON endpoints for a supplied
   request, requiring HTTP 200 and object payloads.
3. Extract `evidence_export` from the paid artifact JSON and verify any
   artifact-embedded `report_model` is an object that matches the report-model
   route.
4. Read operator-supplied PDF bytes and already-extracted PDF text, failing
   closed when text extraction is empty or the raw PDF bytes contain a
   sensitive leak pattern.
5. Reuse `build_pdf_export_scorecard` from #1622 so model/export/PDF checks stay
   centralized.
6. Write only a sanitized runner result: statuses, primitive artifact counts,
   scorecard assertions, and redacted errors. No raw request ID, token, URLs,
   local paths, PDF text, source IDs, or evidence rows are written.
7. Add tests for successful transport/scoring, preflight redaction, malformed
   live payloads, report-model drift, empty PDF text extraction, PDF-byte leak
   detection, sanitizer redaction, and HTTP/JSON failure handling.
8. Enroll the new test in `scripts/run_extracted_pipeline_checks.sh`.

### Files touched

- `.github/workflows/extracted_pipeline_checks.yml`
- `plans/PR-Deflection-Full-Report-QA-Live-Runner.md`
- `scripts/run_deflection_full_report_qa_live_runner.py`
- `scripts/run_extracted_pipeline_checks.sh`
- `tests/test_run_deflection_full_report_qa_live_runner.py`

### Review Contract

Acceptance criteria:

- The runner validates inputs before network calls and supports a preflight mode
  that never calls the network.
- The runner uses the live `/report-model` route as the canonical model input
  and the paid `/artifact` route as the canonical evidence-export input.
- The artifact route must return an object with an object `evidence_export`;
  malformed or missing export data produces a structured runner failure, not a
  traceback.
- If the artifact includes `report_model`, it must match the report-model route
  exactly, and non-object `report_model` values must fail closed, preventing a
  scorecard built from one model and export from another.
- PDF bytes and extracted PDF text are read from operator-supplied files because
  ATLAS does not expose a PDF download route; empty PDF text and raw PDF-byte
  leaks fail before the scorecard can treat extraction as successful.
- The runner output marks PDF text as operator-asserted and not verified from
  PDF bytes until a real extraction dependency lands.
- The runner delegates artifact assertions to the #1622 PDF/export validator and
  does not duplicate count/leak logic.
- The output JSON is safe to commit: it omits raw request IDs, bearer tokens,
  endpoint URLs, local file paths, source IDs, evidence rows, PDF text, customer
  emails, Stripe IDs, and private-note text.
- Tests mock the HTTP transport boundary, not the runner's checker logic, and
  include negative fixtures for each failure branch added in this slice,
  including byte-level PDF leaks and final sanitizer rewrites.
- The new test is enrolled in the extracted checks suite.

Affected surfaces: full-report QA operator scripts, paid deflection artifact
API consumption, PDF/export validation, and #1612 live-proof documentation.
The buyer hosted-result page remains an `atlas-portfolio/web` follow-up.

Risk areas: leaking live capability identifiers into committed proof output,
certifying an empty PDF text extraction, mixing model/export payloads from
different requests, and creating another one-off harness instead of reusing the
shared validator.

- Reviewer rules triggered: R1, R2, R3, R6, R9, R10, R12, R14.

## Mechanism

The script builds two authenticated GET requests from a hosted base URL, a
request ID, and path templates matching the existing public deflection report
routes. The `/report-model` response supplies the canonical `deflection.v1`
model. The `/artifact` response supplies the paid artifact and its
`evidence_export`. If the artifact also carries `report_model`, the runner
requires that value to be an object, compares it to the route model, and fails
on malformed or drifted artifact models.

The runner reads local PDF bytes and extracted PDF text supplied by the
operator. This matches current production delivery: the PDF is an email
attachment rendered by the delivery worker, not a backend download endpoint.
Before any network request, the runner rejects empty extracted text and scans
the decoded PDF bytes for the same sensitive leak classes used by the PDF/export
validator. Once inputs are assembled, it calls `build_pdf_export_scorecard` and
wraps the scorecard with fetch status, input-presence metadata, and an explicit
`operator_asserted` PDF text marker. Before writing or printing, the runner
walks the output and rejects any concrete forbidden values or sensitive token
patterns.

## Intentional

- No ATLAS hosted-result-page smoke. The buyer route lives in
  `atlas-portfolio/web`, and this PR stays in the ATLAS artifact lane.
- No PDF parser or browser automation. This slice proves live artifact assembly
  around the existing validator; PDF text extraction tooling remains an
  operator input until a dedicated extraction dependency is chosen. The runner
  now scans raw PDF bytes for leak patterns, but it does not claim counts are
  byte-derived.
- No API changes. The runner consumes the existing paid report routes and the
  PDF attachment produced by delivery.
- No committed live proof artifact. The script can write a sanitized summary
  locally; the actual live run and any sanitized committed proof are deferred.

## Deferred

- `atlas-portfolio/web` buyer hosted-result smoke: validate the actual
  `juancanfield.com/systems/support-ticket-deflection/results/{requestId}`
  route, not ATLAS `portfolio-ui`.
- Live execution artifact: run this script against a controlled paid
  Zendesk-shaped request and commit only the sanitized scorecard if it passes.
- PDF text extraction automation: choose and gate the extraction mechanism
  before replacing the explicit `--pdf-text` input.
- Redirect hardening for the live runner: build a no-redirect opener or
  revalidate the final URL after redirects. Deferred because this review-fix
  patch is scoped to artifact proof integrity and sanitizer coverage, while the
  current runner still requires an operator-supplied hosted HTTPS base URL and
  keeps URL/token values out of output.

Parked hardening: none.

## Verification

- Focused live-runner pytest for `tests/test_run_deflection_full_report_qa_live_runner.py` - 12 passed.
- Python compile check for the live-runner script and its test - passed.
- Extracted pipeline CI enrollment audit - OK, 185 matching tests enrolled.
- Whitespace diff check - passed.
- Full extracted pipeline bundle via `scripts/run_extracted_pipeline_checks.sh` - package audits passed; reasoning-core suite 295 passed; extracted-content suite 4572 passed, 10 skipped.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/extracted_pipeline_checks.yml` | 4 |
| `plans/PR-Deflection-Full-Report-QA-Live-Runner.md` | 188 |
| `scripts/run_deflection_full_report_qa_live_runner.py` | 439 |
| `scripts/run_extracted_pipeline_checks.sh` | 1 |
| `tests/test_run_deflection_full_report_qa_live_runner.py` | 529 |
| **Total** | **1161** |
