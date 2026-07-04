# PR-Deflection-Landing-Demo-Contract-Example

## Why this slice exists

#1853 is the ATLAS-side prerequisite for contract-deriving the landing demo.
The Product Gaps report-shape arc (#1843 -> #1856) is now settled, so the
paused demo-derive work can resume without racing active `deflection.v1` field
changes.

Root cause: the repo has a producer-backed snapshot example generator, but the
paid deflection report example is only compared to producer output in tests. The
generator's `--check` drift gate writes/checks the snapshot file only, so the
public demo can still depend on a hand-maintained paid report fixture that
silently drifts from the producer and from the snapshot projection relationship.

This PR fixes that root for the ATLAS artifact: one frozen synthetic input
builds a real `build_deflection_report_artifact(...)` payload, the committed
paid report example is generated from that payload, and the committed snapshot
example is derived from the same report through `build_deflection_snapshot(...)`.
The downstream atlas-portfolio slice can then vendor/consume this generated
example instead of copying hand-authored demo constants.

This is slightly over the 400 LOC soft cap because the review fix added the
held-out `--output` alias class coverage for write and check modes; splitting
those negative fixtures out would leave the fail-closed CLI behavior
under-proven.

## Scope (this PR)

Ownership lane: deflection/landing-demo-contract-derived
Slice phase: Functional validation

1. Extend `scripts/generate_deflection_snapshot_example.py` so `--check`
   verifies both `content_ops_faq_deflection_report_example.json` and
   `content_ops_faq_deflection_snapshot_example.json`.
2. Keep the canonical input synthetic and curated; no real customer export or
   real ticket text enters the public demo artifacts.
3. Add tests proving the report example and snapshot example are generated from
   the same producer payload, and that stale/missing report examples fail the
   checker.
4. Enroll the paid report example in the extracted pipeline drift trigger/gate.
5. Do not touch atlas-portfolio, the locked landing preview constants, or #1836
   in this PR.
6. Preserve isolated scratch generation: when callers use deprecated
   `--output /tmp/snapshot.json` without `--report-output`, write/check the
   paired report next to the requested snapshot instead of touching the
   committed report example.

### Files touched

- `.github/workflows/extracted_pipeline_checks.yml`
- `plans/PR-Deflection-Landing-Demo-Contract-Example.md`
- `scripts/generate_deflection_snapshot_example.py`
- `tests/test_content_ops_faq_deflection_snapshot_example_generator.py`

### Review Contract

- Acceptance criteria:
  - [ ] The committed paid deflection report example is generated from
        `build_deflection_report_artifact(...)` on a frozen synthetic sample.
  - [ ] The committed snapshot example is derived from that same paid report
        payload via the production snapshot projection.
  - [ ] `--check` fails closed when either committed example is missing or
        stale.
  - [ ] The synthetic input is fabricated and contains no real customer export
        or private ticket data.
  - [ ] No atlas-portfolio consumer or landing-page constants are changed in
        this PR.
- Affected surfaces: docs/frontend deflection examples, extracted pipeline
  drift gate, report-contract docs tests.
- Risk areas: public-demo PII boundary, generated artifact drift, cross-repo
  sequencing.
- Reviewer rules triggered: R1, R2, R3, R9, R10, R12, R14.

## Mechanism

Reuse the existing synthetic support-ticket fixture in
`generate_deflection_snapshot_example.py` as the single canonical input. The
script already builds a real FAQ result and report artifact; this slice makes
that artifact the generator-owned paid example as well as the source for the
snapshot projection.

The generator keeps backward-compatible snapshot output behavior while adding
explicit report/snapshot outputs for normal generation and `--check`. In check
mode it renders both JSON payloads with sorted keys and compares them to the
committed docs files. A stale/missing paid report example fails the same way the
snapshot example fails today.

Tests exercise the CLI at the transport boundary: write both outputs to a temp
directory, check both current files, and assert missing/stale report examples
return non-zero without rewriting the files. The extracted pipeline check then
becomes the CI drift gate because it already runs the generator in `--check`
mode.

Review follow-up: the deprecated `--output` alias remains a snapshot-output
alias. If a caller uses it for scratch output and does not provide an explicit
`--report-output`, the CLI derives a sibling report JSON path next to the
requested snapshot. No scratch command writes/checks the committed report example
unless the caller uses the default outputs or passes that path explicitly.

## Intentional

- Keep the existing docs example paths so current docs/tests/consumers do not
  need a rename migration.
- Keep atlas-portfolio consumption deferred; this PR publishes the ATLAS-owned
  generated artifact that the downstream repo needs.
- Keep the locked preview's curated 8-section subset deferred to PR-B; a
  consumer may render a subset of a generated example, but the source fixture
  should no longer be hand-authored.
- Do not introduce real SaaS/customer data for demo appeal; public demo examples
  must remain fabricated even when generated through the real producer.

## Deferred

- #1853 PR-B / atlas-portfolio: replace the two hand-authored landing demo
  constants with the generated ATLAS example and its projection, then add a
  `DEMO_SNAPSHOT === projectSnapshot(DEMO_REPORT)` consumer test.
- Marketing-quality tuning of the synthetic sample is deferred unless the
  generated artifact is degenerate; this PR is the contract/drift foundation.

Parked hardening: none.

## Verification

- `python scripts/generate_deflection_snapshot_example.py --check` - passed.
- `python -m pytest tests/test_content_ops_faq_deflection_snapshot_example_generator.py -q` - passed, 10 tests.
- `python -m pytest tests/test_content_ops_faq_report_contract_docs.py -q` - passed, 5 tests.
- Python compile check for `scripts/generate_deflection_snapshot_example.py` and `tests/test_content_ops_faq_deflection_snapshot_example_generator.py` - passed.
- `scripts/run_extracted_pipeline_checks.sh` - passed (`5004 passed, 15 skipped`).
- `bash scripts/local_pr_review.sh --current-pr-body-file /tmp/deflection-landing-demo-contract-example-pr-body.md` - passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/extracted_pipeline_checks.yml` | 2 |
| `plans/PR-Deflection-Landing-Demo-Contract-Example.md` | 140 |
| `scripts/generate_deflection_snapshot_example.py` | 94 |
| `tests/test_content_ops_faq_deflection_snapshot_example_generator.py` | 203 |
| **Total** | **439** |
