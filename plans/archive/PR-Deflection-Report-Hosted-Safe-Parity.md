# PR-Deflection-Report-Hosted-Safe-Parity

## Why this slice exists

#1828 made the public `/api/content-ops/deflection/report` response project the
paid report model through generated hosted-safe field allowlists, but the
reviewer flagged one remaining weak spot: the projection has to hand-classify a
few non-scalar leaf shapes (`source_date_window`, `status_counts`,
`status_mix`, `reason_counts`, `term_mappings`).

Root cause: generated field-name parity is covered, but generated field-shape
parity is not. A future backend contract could add a hosted-safe record or
object-array field, the public projection could silently drop it, and the
current tests would still pass if they only exercised the already-known fields.
During review, that exact class turned up in current runtime behavior:
`source_date_window` is produced by `_support_tax_data` as a scalar record, but
the hosted proxy only preserved `status_counts` and `status_mix` records.
After rebasing onto #1830, the same parity guard caught the new
`suppressed_repeat_review_queue` section: `reason_counts` is a scalar record and
`suppression_reason` / `suppression_reason_label` are scalar leaves that must be
classified for the hosted projection to preserve them.

This change fixes the test-gap root for the hosted-safe public boundary by
building one report model fixture from every generated hosted-safe section
field, classifying each current leaf shape, and asserting the public projection
preserves every classified field. It also fixes the current upstream-proven
runtime miss by preserving `source_date_window` as a hosted-safe scalar record.
Unclassified future leaf fields become object fixtures by default, which the
current projection intentionally drops; that makes a new shape fail test-time
until the projection and classifier are updated together.

The slice is slightly over the 400 LOC soft cap after review fixes because the
`source_date_window` runtime repair, two-entry nested parity fixture, and
section-metadata assertions are one coupled boundary proof. The extra #1830
field classifications are also same-class parity coverage; splitting them would
leave the hosted-safe projection green while part of the reviewed defect class
remains unproven.

## Scope (this PR)

Ownership lane: deflection/report-contract-1805
Slice phase: Production hardening

1. Add a focused parity test to the existing `portfolio-ui` ATLAS proxy suite.
2. Prove every generated hosted-safe report-model field survives
   `publicHostedReportModel`, including nested rows/items, scalar arrays,
   scalar records, and known object arrays.
3. Preserve the real backend `source_date_window` record at the public hosted
   paid report boundary.
4. Strengthen the parity assertion to compare section metadata and every nested
   row/item, not only the first item.
5. Classify the #1830 suppressed-repeat review fields discovered by the
   merge-commit CI run.

### Review Contract

Acceptance criteria:

1. The test imports the generated report-model contract rather than duplicating
   section IDs by hand.
2. Every current hosted-safe leaf field is classified as scalar, scalar array,
   scalar record, or object array.
3. The fixture covers every generated hosted-safe section and nested item shape.
4. The assertion checks preservation, not just absence of private fields,
   including section metadata and multiple nested rows/items.
5. Summary remains fail-closed (`summary: {}`), so the previous public payload
   boundary is not widened.
6. A real `source_date_window` scalar-record shape survives
   `publicHostedReportModel`.

Affected surfaces:

- Public paid report JSON proxy projection.
- Public paid report JSON proxy test coverage.

Risk areas:

- False-green projection coverage for future generated report-model shape
  changes.
- Privacy boundary regressions if a new scalar-record projection admits
  non-scalar values or raw evidence fields.

Reviewer rules triggered:

- R2 Test evidence.
- R5 Contract/API compatibility.
- R13 Class-fix proof.
- R14 Codebase verification.

### Files touched

- `plans/PR-Deflection-Report-Hosted-Safe-Parity.md`
- `portfolio-ui/api/content-ops/deflection/report.js`
- `portfolio-ui/scripts/faq-deflection-atlas-proxy.test.mjs`

## Mechanism

The test walks `DEFLECTION_REPORT_SECTION_IDS` and each section's generated
`*_HOSTED_CONSUMER_SAFE_FIELDS` constants. For nested generated field constants
such as `*_ROWS_HOSTED_CONSUMER_SAFE_FIELDS` and
`*_ITEMS_CSAT_SIGNAL_HOSTED_CONSUMER_SAFE_FIELDS`, it builds a matching nested
object or array fixture.

Leaf fields are shape-classified in the test. Current scalar fields receive
realistic scalar values, scalar arrays receive representative arrays, record
fields receive scalar records, `source_date_window` receives the real
`{source_date_start, source_date_end, source_window_days}` shape, and
`term_mappings` receives an object-array fixture with a deliberately private
extra key. The #1830 `suppression_reason` fields use scalar values, and
`reason_counts` uses the scalar-record path. Unknown future leaves are
intentionally emitted as object records, so the current projection drops them
and the parity assertion fails until the new shape is explicitly handled.

The assertion then runs the fixture through `publicHostedReportModel` and checks
that each generated hosted-safe field path survives while the fail-closed
summary remains empty and private marker values remain absent. Nested
`rows`/`items` use two distinct entries and the assertion recurses over every
entry, so later-row projection drift is covered.

The runtime projection adds `source_date_window` and `reason_counts` to the
hosted-safe scalar-record field set, reusing the existing `cloneScalarRecord`
path so only scalar record members can cross the public boundary.

## Intentional

- Runtime behavior changes only for upstream-proven scalar records
  (`source_date_window` and #1830 `reason_counts`); this does not add a broad
  object passthrough.
- The shape classifier is test-local because it is a guardrail for the
  hand-coded projection bridge; a generated shape artifact would be better, but
  that requires changing the contract generator and consumer runtime together.
- The fixture uses realistic synthetic values instead of real ticket prose so
  the test can safely live in the public repo without introducing customer data.

## Deferred

- Generate report-model field-shape metadata from the ATLAS projection contract
  so `report.js` can project records/object arrays without test-local shape
  classification.

Parked hardening: none.

## Verification

- Passed: `npm --prefix portfolio-ui run test:deflection-atlas-proxy` (28 tests).
- Pending before push: `bash scripts/push_pr.sh <body-file> -u origin HEAD`
  (runs the local PR review bundle through the managed push path).

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Deflection-Report-Hosted-Safe-Parity.md` | 156 |
| `portfolio-ui/api/content-ops/deflection/report.js` | 7 |
| `portfolio-ui/scripts/faq-deflection-atlas-proxy.test.mjs` | 281 |
| **Total** | **444** |
