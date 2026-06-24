# PR-Deflection-Report-Hosted-Safe-Proxy

## Why this slice exists

#1805's hosted-safe artifact is now published, and the #1826 review carried the
consumer requirement forward: `hosted_consumer_safe_fields` is a paid web-render
allowlist, not a free-surface allowlist. The public report API currently returns the full unlocked
ATLAS artifact, which can carry raw paid/export-only fields such as markdown,
`faq_result`, `evidence_export`, `source_ids`, `evidence_quotes`, and
`top_evidence`.

Root cause: the public browser JSON boundary is still validate-and-pass for
unlocked artifacts instead of allowlist-constructing the hosted report payload
from the generated report-model contract. This fixes the root at that boundary:
locked/free responses stay snapshot-only, and unlocked responses expose only a
hosted-safe report model constructed from generated allowlists.

Diff-size note: this is just over the 400 LOC soft cap because the boundary
test needs a representative unlocked report model with nested safe and raw
fields. The production change is limited to the public JSON proxy.

## Scope (this PR)

Ownership lane: deflection/report-contract-1805
Slice phase: Production hardening

1. Add a public-report payload constructor that returns no artifact unless the
   artifact status is unlocked.
2. For unlocked reports, project `artifact.report_model` through the generated
   hosted-consumer-safe field constants and return that as the only public
   artifact payload.
3. Add tests proving locked/free JSON has no `answer`/`steps`, while unlocked
   JSON can include paid web-render `answer`/`steps` but strips export-only raw
   fields.

### Review Contract

- Acceptance criteria: public report API responses are snapshot-only when
  locked, hosted-safe report-model-only when unlocked, and never expose raw
  `faq_result`, `evidence_export`, markdown, source ids, evidence quotes,
  `top_evidence`, `recommended_title`, or `representative_phrasing`.
- Affected surfaces: `portfolio-ui/api/content-ops/deflection/report.js` and
  `portfolio-ui/scripts/faq-deflection-atlas-proxy.test.mjs`.
- Risk areas: do not break server-side result-page rendering or evidence export
  download, both of which need the raw server-only artifact after unlock; do not
  treat hosted-safe fields as free-safe.
- Reviewer rules triggered: R1 requirements match, R2 test evidence, R3
  security/privacy boundary, R6 generated contract consumption, R9 checker
  failure branches, R13 class fix, R14 codebase verification.

### Files touched

- `plans/PR-Deflection-Report-Hosted-Safe-Proxy.md`
- `portfolio-ui/api/content-ops/deflection/report.js`
- `portfolio-ui/scripts/faq-deflection-atlas-proxy.test.mjs`

## Mechanism

The public `report.js` handler continues to call `loadDeflectionReport`, but it
serializes a new public payload instead of dumping the returned object. The
constructor keeps the snapshot envelope for all successful responses, includes
no artifact unless ATLAS returned `artifact_status: "unlocked"`, and projects
`artifact.report_model.sections[].data` through generated hosted-consumer-safe
field constants when unlocked.

The projection is recursive for object fields and object-array fields when a
matching nested generated allowlist exists. Fields without a generated
allowlist are copied only as scalar or scalar-array values; object payloads
without explicit nested allowlists are omitted.

## Intentional

- `loadDeflectionReport` remains raw/server-only. The result-page HTML renderer
  and evidence-export download still need the full unlocked artifact on the
  server side; this PR constrains only the browser-facing JSON API boundary.
- `answer` and `steps` remain allowed only in unlocked hosted-safe report-model
  rows. The locked/free path still returns no artifact and no paid answer body.

## Deferred

- atlas-portfolio cross-repo consumption should mirror this pattern once it
  consumes the generated ATLAS report-model artifact.

Parked hardening: none.

## Verification

- Portfolio proxy test: package `portfolio-ui`, script
  `test:deflection-atlas-proxy` -- 27 tests passed.
- Local PR review bundle -- passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Deflection-Report-Hosted-Safe-Proxy.md` | 99 |
| `portfolio-ui/api/content-ops/deflection/report.js` | 127 |
| `portfolio-ui/scripts/faq-deflection-atlas-proxy.test.mjs` | 181 |
| **Total** | **407** |
