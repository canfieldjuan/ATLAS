# PR-Resolution-Audit-S8a-Runtime-QA-Gate

Ownership lane: resolution-audit-csv

## Why this slice exists

S8a of the #1993 remediation arc (issue #2051, launch blocker #7 first
half). The deterministic QA scorecard
(`build_deflection_full_report_qa_scorecard`) exists and detects report
drift, but it has ZERO runtime callers -- it is reachable only from
scripts/ and tests. Both runtime boundary crossings accept any artifact
unchecked: persist-at-generation (`_gate_deflection_report_artifacts` ->
`store.save_report`) and paid-PDF delivery
(`send_pending_deflection_report_deliveries` -> `_pdf_attachments`). A
drifted or wrong paid artifact can be generated, persisted, paid for,
and delivered with nothing blocking it.

### Problem-derived contract

- Root cause: the scorecard is a standalone checker, not a gate -- no
  runtime path consults it before persisting or delivering an artifact.
- Correct fix must touch/change:
  1. `extracted_content_pipeline/faq_deflection_report.py` (owned): one
     runtime gate predicate that derives the evidence export from the
     artifact itself, runs the EXISTING scorecard (no surface
     observations at these boundaries; default caps), and raises a typed
     `DeflectionReportQAGateError` naming the failing assertion ids when
     `ok` is false. No new assertion logic -- the scorecard stays the
     single source of truth.
  2. `extracted_content_pipeline/api/control_surfaces.py` (owned): in
     `_gate_deflection_report_artifacts`, after the PII scrub and before
     `save_report`, gate the scrubbed artifact; failure -> HTTP 502
     naming the failing assertion ids. Fail-closed at persist: a
     QA-failing artifact never enters the store, so checkout can never
     take money for an artifact delivery would later refuse.
  3. Tests, both directions, on the real fixture artifact and the real
     store harness already used by the suites: healthy artifact passes
     the gate and persists unchanged; drifted artifact (bogus
     schema_version / missing required section) 502s at the route naming
     the failing assertion ids and never persists; the gate never
     modifies the artifact; an unreadable artifact fails closed.

Build-time contract amendment (discovered, not planned): a delivery-side
gate was initially planned and BUILT, then reverted before commit. The
existing delivery suite pins an explicit tolerance contract
(`test_delivery_worker_falls_back_for_future_report_model_schema`):
persisted artifacts may straddle schema versions across deploys and must
still deliver. A full-scorecard delivery gate contradicts that pinned
product contract -- the scorecard asserts the CURRENT schema, which is
only guaranteed at generation time. Fail-closed for paid delivery holds
TRANSITIVELY: delivery only sends persisted+paid artifacts, and nothing
QA-failing can be persisted anymore. Pre-existing persisted rows are an
ops-sweep follow-up (Deferred).
- Must not change:
  - Report content/shape -- the gate gates, it never edits.
  - Scrub (S6A) and snapshot behavior; scorecard assertions themselves.
  - scripts/ callers -- diagnostics must still be able to build and
    score BAD artifacts, so `build_deflection_report_artifact` itself
    stays ungated.
  - Delta delivery path (separate lane); checkout/payment semantics.
  - The delivery worker's schema-tolerance contract (see amendment
    above): `atlas_brain/content_ops_deflection_delivery.py` is
    deliberately untouched.

Empirical pre-checks (run before coding): the healthy fixture artifact
passes the scorecard post-scrub (ok=true, 73 assertions); a bogus
schema_version and a truncated evidence export fail with named
assertion ids (`model.schema_version`,
`evidence_export.evidence_rows.length`); the delivery loop already
fail-closes typed exceptions via incident + `_mark_failed`.

Round-1 review refinement (verified failing first): the gate scored a
FRESHLY DERIVED evidence export while `DeflectionReportArtifact.as_dict`
embeds one in the persisted payload (and the process contract requires
it for paid artifacts) -- so a drifted or missing embedded
evidence_export rode through the gate unvalidated. The gate now scores
the export that will actually be stored: for a mapping artifact it reads
the embedded `evidence_export` (missing/non-mapping fails closed); it
derives only for a DeflectionReportArtifact object, whose as_dict embeds
exactly that derivation. The PII-scrub test stops replacing the real
embedded export (its rows already carry the planted PII).

Round-2 review refinements (both verified failing first): (1) a
same-count export whose ELEMENTS drifted (junk questions/rows with
matching lengths) passed the count-level scorecard -- the gate now
requires the stored export to equal the canonical derivation of the
stored artifact, excluding only the content-hash linkage keys
(cluster_id, repeat_key) that legitimately change when re-derived from
scrubbed text; (2) paid read surfaces project the stored model through
stored_deflection_report_model(), and an artifact that projection
rejects (e.g. non-integer section priority) persisted, took payment,
then 404'd the report-model route -- the gate now runs that SAME
projection (lazy import; deflection_report_access imports this module).
Both checks reuse canonical readers; no hand-written field lists.

Round-3 refinements (cap round; both verified failing first): (1) the
stored projection silently DROPS invalid sections rather than returning
None, so breaking one required section passed the None-check while the
paid report-model route lost that section -- the gate now runs the SAME
scorecard on the projected model paid readers see (failing ids prefixed
`stored_projection:`); (2) stripping the volatile linkage keys from the
export-integrity comparison accepted exports that OMIT them entirely --
the keys are now normalized to a presence/type marker instead of
removed, so omission and non-string corruption fail while the
scrub-volatile hash content stays ignored. Residual (accepted by
construction): a tampered-but-well-typed hash value is indistinguishable
from a legitimate scrub re-derivation.

Review-loop guard: HARD cap of 3 Codex rounds counted by ROUNDS; at
cap, fix what is written, resolve/waive remaining threads, merge on
required-green.

## Scope (this PR)

Slice phase: Vertical slice

Max files: 5

1. `extracted_content_pipeline/faq_deflection_report.py` -- gate
   predicate + typed error.
2. `extracted_content_pipeline/api/control_surfaces.py` -- persist-side
   wiring.
3. `tests/test_content_ops_deflection_report.py` -- gate predicate both
   directions.
4. `tests/test_extracted_content_deflection_submit.py` -- persist
   boundary both directions; the existing PII-scrub proof rebuilt on a
   REAL-builder artifact (the storage boundary now refuses the
   hand-rolled one-section shape it used, which is the intended
   contract change).
5. This plan doc.

### Files touched

- `extracted_content_pipeline/api/control_surfaces.py`
- `extracted_content_pipeline/faq_deflection_report.py`
- `plans/PR-Resolution-Audit-S8a-Runtime-QA-Gate.md`
- `tests/test_content_ops_deflection_report.py`
- `tests/test_extracted_content_deflection_submit.py`

### Review Contract

Acceptance criteria:
1. A healthy artifact flows through generate -> scrub -> persist ->
   paid delivery unchanged (no content edits, no new failures).
2. A drifted artifact (bogus schema_version or missing required
   section) is refused at persist with HTTP 502 naming the failing
   assertion ids, and nothing is written to the store.
3. scripts/ diagnostics still build and score bad artifacts (the raw
   builder is not gated).
4. The gate never modifies the artifact, snapshot, or scrub output.
5. The delivery worker's tolerance contract is untouched (its suite
   passes unchanged).

Reachability proof: wired on the live `_gate_deflection_report_artifacts`
persist path; tests assert store contents through the real
InMemoryDeflectionReportArtifactStore, not just the predicate.

Affected surfaces: deflection report persist route.

Risk areas: false-positive gate bricking healthy reports (bounded by
the empirical pre-check: real fixture passes post-scrub, and the
healthy-path e2e tests pin it); artifacts persisted pre-gate (delivery
tolerance keeps them deliverable; ops sweep deferred).

Reviewer rules triggered: R1, R2, R10, R14 (extracted package change; test evidence; new fail-closed gate on the paid path probed both directions; admission-adjacent boundary).

## Mechanism

`check_deflection_report_artifact_qa(artifact)` builds
`build_deflection_evidence_export(artifact)` and runs
`build_deflection_full_report_qa_scorecard(report_model,
evidence_export=...)`; on `ok == false` it raises
`DeflectionReportQAGateError` (a `ValueError`) carrying the failing
assertion ids and the scorecard. `_gate_deflection_report_artifacts`
calls it on the scrubbed artifact before `save_report`; the route's
existing `ValueError` handler maps it to HTTP 502 with the assertion ids
in the detail.

## Intentional

- Fail-closed at the persist crossing: a QA-failing artifact never
  enters the store, so checkout can never take money for a report that
  delivery would refuse. Flag-only was rejected for the same reason.
- Delivery deliberately NOT gated (contract amendment above): the
  scorecard asserts the current schema, which only generation
  guarantees; delivery's pinned tolerance contract keeps already-paid
  artifacts deliverable across deploys.
- The gate lives beside the scorecard in the extracted package so the
  API layer and the host brain import one predicate -- no forked
  assertion logic (the same-predicate discipline S8b will follow for
  the billed-repeat rule).
- Surface observations are deliberately not asserted at these
  boundaries: they describe rendered surfaces (hosted page), which do
  not exist yet at persist/delivery time.

## Deferred

- S8b money reconciliation guard (#2052) -- next slice, same lane.
- Ops sweep script over already-persisted artifacts (they predate the
  gate and stay deliverable under the delivery tolerance contract).
- Surface-observation assertions for the hosted result page.

## Verification

- `python -m pytest tests/test_content_ops_deflection_report.py -q`
- `python -m pytest tests/test_extracted_content_deflection_submit.py -q`
- `python -m pytest tests/test_atlas_content_ops_deflection_delivery.py -q`
  (unchanged suite proving the delivery tolerance contract holds)
- Full CI mirror bash `scripts/run_extracted_pipeline_checks.sh`
- Empirical both-direction probes above reproduced as pinned tests.

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/api/control_surfaces.py` | 8 |
| `extracted_content_pipeline/faq_deflection_report.py` | 129 |
| `plans/PR-Resolution-Audit-S8a-Runtime-QA-Gate.md` | 221 |
| `tests/test_content_ops_deflection_report.py` | 174 |
| `tests/test_extracted_content_deflection_submit.py` | 220 |
| **Total** | **752** |
