# PR-Deflection-Landing-Demo-High-Volume-Example

## Why this slice exists

atlas-portfolio PR #389 correctly switched the landing demo to consume the
ATLAS-generated paired report/snapshot example from #1857, but that ATLAS
example was only a compact four-ticket proof fixture. The downstream landing
demo needs a producer-truthful example that still carries marketing-scale
support volume; otherwise the page weakens the Support Tax story and the
portfolio cost-projection guard fails when the old hand-authored high-volume
literals disappear.

Root cause: the upstream generated demo artifact used a tiny hand-listed
synthetic input as both the contract example and the public landing demo seed.
That made the artifact structurally correct, but not representative of the
volume semantics the landing page is supposed to demonstrate.

This change fixes the root for the downstream #389 blocker by generating a
moderate-volume real synthetic input programmatically inside the ATLAS
producer, then regenerating both the report-model example and the snapshot
projection from that one producer output.

Diff budget note: this PR exceeds the 400 LOC soft cap because the committed
demo report artifact carries a full producer-derived report model. The review
fix uses 360 real synthetic rows instead of either four rows or 1,140 rows,
keeping the Support Tax story substantial while making `ticket_count`,
`source_count`, source proof, source ids, and evidence rows agree by
construction.

## Scope (this PR)

Ownership lane: deflection/landing-demo-contract-derived
Slice phase: Functional validation

1. Replace the four-row demo input in
   `scripts/generate_deflection_snapshot_example.py` with a deterministic,
   programmatic, moderate-volume synthetic support-ticket sample.
2. Regenerate the committed report and snapshot examples from the producer so
   the downstream landing demo can consume a truthful high-volume artifact.
3. Add focused generator tests proving the committed example remains paired,
   synthetic, high-volume, and rich enough to include outcome diagnostics.
4. Update the Atlas Intel UI fixture test that reads the same generated JSON so
   CI asserts the new high-volume/property-based contract.

### Review Contract

- Reviewer rules triggered: R9, R12.
- The canonical demo input is synthetic/curated only; no customer export,
  secrets, email addresses, phone numbers, or real account identifiers are
  introduced.
- The report example remains produced by `build_deflection_report_artifact`;
  the snapshot example remains derived by `build_deflection_snapshot`.
- The generated snapshot summary exposes at least 300 repeat tickets and the
  top question exposes at least 90 tickets so portfolio can guard a volume
  property instead of old literals.
- The generated paid-report evidence surfaces agree with the count surfaces:
  per-question `ticket_count` matches source proof and evidence rows, and
  report-wide repeat count matches source/evidence totals.
- The report model contains buyer-visible action sections plus
  `outcome_diagnostics`, proving the landing demo can render a richer preview
  from the generated artifact.
- The generator `--check` path fails closed when either committed artifact
  drifts.

### Files touched

- `atlas-intel-ui/scripts/content-ops-deflection-report-ui.test.mjs`
- `docs/frontend/content_ops_faq_deflection_report_example.json`
- `docs/frontend/content_ops_faq_deflection_snapshot_example.json`
- `plans/PR-Deflection-Landing-Demo-High-Volume-Example.md`
- `scripts/generate_deflection_snapshot_example.py`
- `tests/test_content_ops_faq_deflection_snapshot_example_generator.py`
- `tests/test_content_ops_faq_report_contract_docs.py`

## Mechanism

The generator defines a small set of synthetic repeat-question cohorts with
ticket counts, safe customer wording, routing metadata, status/CSAT patterns,
and optional publishable resolution text. A helper expands those cohorts into
360 deterministic support-ticket rows with stable synthetic `source_id`s and
May 2026 dates. The existing producer path then runs `build_ticket_faq_markdown`
over that expanded input and passes the result to
`build_deflection_report_artifact`; the snapshot file is still the
`build_deflection_snapshot(report_payload)` projection.

The tests assert the generated files match the committed examples, the
snapshot is a projection of the report, the sample clears the intended volume
thresholds, count/evidence surfaces agree, and no obviously real PII-shaped
values appear in the synthetic source ids or text fixtures.

## Intentional

- The sample uses 360 real synthetic rows as the middle ground between the
  tiny four-row fixture, the bloated 1,140-row fixture, and the rejected
  weighted-row version whose proof surfaces contradicted the count surfaces.
- The wording stays curated and short. This is public marketing/demo data, so
  it should demonstrate the contract and product story without imitating real
  customer prose too closely.

## Deferred

- atlas-portfolio #389 still needs to rerun
  `npm --prefix web run generate:deflection-contracts` after this ATLAS PR
  lands, then update its landing guards to check volume properties instead of
  the old hand-authored literals.

Parked hardening: none.

## Verification

- `python scripts/generate_deflection_snapshot_example.py --check` -- passed.
- `python -m pytest tests/test_content_ops_faq_deflection_snapshot_example_generator.py -q` -- 12 passed.
- `python -m pytest tests/test_content_ops_faq_report_contract_docs.py -q` -- 5 passed.
- `npm run test:content-ops-deflection-report-ui` from `atlas-intel-ui/` -- 3 passed.
- `scripts/local_pr_review.sh --current-pr-body-file /tmp/pr_body_deflection_landing_demo_high_volume_example.md` -- passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas-intel-ui/scripts/content-ops-deflection-report-ui.test.mjs` | 7 |
| `docs/frontend/content_ops_faq_deflection_report_example.json` | 9718 |
| `docs/frontend/content_ops_faq_deflection_snapshot_example.json` | 72 |
| `plans/PR-Deflection-Landing-Demo-High-Volume-Example.md` | 128 |
| `scripts/generate_deflection_snapshot_example.py` | 157 |
| `tests/test_content_ops_faq_deflection_snapshot_example_generator.py` | 56 |
| `tests/test_content_ops_faq_report_contract_docs.py` | 5 |
| **Total** | **10143** |
