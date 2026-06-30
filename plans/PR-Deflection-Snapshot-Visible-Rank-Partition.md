# PR-Deflection-Snapshot-Visible-Rank-Partition

## Why this slice exists

atlas-portfolio #389 exposed a source-projection bug after ATLAS #1859 enriched
the generated landing demo. The free Snapshot now emits `locked_questions` for
ranks that are already visible elsewhere: rank 6 is visible in
`top_blind_spots`, and rank 7 is visible in `teaser.previews`. The portfolio
landing can hide or filter that locally, but the real contract problem is
upstream: ATLAS is claiming those ranks are "withheld" even though the Snapshot
already surfaced their question text in another public section.

Root cause: `build_deflection_snapshot` partitions only the teaser full-answer
rank out of `locked_questions`. It does not exclude teaser preview ranks or
blind-spot ranks, so the free Snapshot surfaces can overlap. That violates the
buyer-facing contract: proven teaser, blind spots, previews, and locked backlog
should be a partition of ranked questions, with each rank appearing in the
right public surface exactly once.

This PR fixes the root in the ATLAS projection instead of patching one portfolio
component. `locked_questions` becomes the genuinely withheld set: ranks already
visible in the top questions, teaser full answer, teaser previews, or blind
spots are excluded before the Snapshot is emitted. The generated demo Snapshot
is then refreshed from that corrected projection.

## Scope (this PR)

Ownership lane: deflection/landing-demo-contract-derived
Slice phase: Production hardening

1. Change Snapshot `locked_questions` construction to exclude all visible
   Snapshot ranks, not only the teaser full-answer rank.
2. Refresh the generated public demo Snapshot after the projection fix.
3. Add tests proving visible ranks are partitioned and the generated example
   no longer requires a locked backlog when no rank is genuinely withheld.

### Review Contract

Acceptance criteria:

- `locked_questions` contains only ranks not visible through top questions,
  teaser full answer, teaser previews, or top blind spots.
- Both legacy artifact projection and report-model projection follow the same
  partition rule.
- The generated docs/frontend Snapshot example is current and may have an empty
  `locked_questions` list if every post-top-N rank is visible elsewhere.
- No private fields are added to `locked_questions`; it still exposes only rank
  and ticket count.

Affected surfaces:

- Free deflection Snapshot JSON.
- Generated public demo Snapshot consumed by atlas-portfolio #389.
- Snapshot/report drift tests.

Risk areas:

- Snapshot coverage accounting: ranks removed from `locked_questions` must still
  be covered by another public surface.
- Public copy/demo assumptions: downstream consumers must not assume locked rows
  are always present.

Reviewer rules triggered: R1, R2, R7, R9, R10, R12, R13, R14.

### Files touched

- `docs/frontend/content_ops_faq_deflection_snapshot_example.json`
- `extracted_content_pipeline/faq_deflection_report.py`
- `plans/PR-Deflection-Snapshot-Visible-Rank-Partition.md`
- `tests/test_content_ops_deflection_report.py`
- `tests/test_content_ops_faq_deflection_snapshot_example_generator.py`
- `tests/test_deflection_snapshot_report_drift.py`

## Mechanism

Add small projection helpers that collect visible Snapshot ranks from:

- top questions,
- teaser full answer,
- teaser previews,
- top blind spots.

`locked_questions` then filters out those visible ranks. The legacy artifact
path has no `top_blind_spots`, but it still excludes teaser preview ranks. The
report-model path computes `top_blind_spots` before locked rows so blind-spot
ranks are excluded at the same source boundary.

The drift test's rank-coverage check is updated to count teaser previews and
blind spots as visible coverage, not just locked rows and the teaser full
answer. The generated demo example test asserts the public surfaces are
disjoint instead of requiring a non-empty locked backlog.

## Intentional

- This does not add a portfolio-side locked-row filter. The partition belongs
  in the ATLAS projection so PDF/email/result-page consumers receive the same
  corrected Snapshot.
- The generated demo may have zero `locked_questions` after this fix. That is
  correct when all post-top-N ranks are already visible as blind spots or
  teaser previews.

## Deferred

- atlas-portfolio #389 must regenerate after this ATLAS PR lands and can then
  remove any downstream workaround that only existed for overlapping locked
  ranks.

Parked hardening: none.

## Verification

- Pass: `python scripts/generate_deflection_snapshot_example.py --check`.
- Pass: `python -m pytest tests/test_content_ops_deflection_report.py tests/test_content_ops_faq_deflection_snapshot_example_generator.py tests/test_deflection_snapshot_report_drift.py -q` (197 passed).
- Pending before push: `bash scripts/local_pr_review.sh --current-pr-body-file /tmp/pr_body_deflection_snapshot_visible_rank_partition.md`.

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/frontend/content_ops_faq_deflection_snapshot_example.json` | 11 |
| `extracted_content_pipeline/faq_deflection_report.py` | 54 |
| `plans/PR-Deflection-Snapshot-Visible-Rank-Partition.md` | 126 |
| `tests/test_content_ops_deflection_report.py` | 10 |
| `tests/test_content_ops_faq_deflection_snapshot_example_generator.py` | 17 |
| `tests/test_deflection_snapshot_report_drift.py` | 17 |
| **Total** | **235** |
