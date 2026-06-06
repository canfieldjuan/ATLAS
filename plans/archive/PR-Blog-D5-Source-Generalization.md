# PR-Blog-D5-Source-Generalization

Ownership lane: `content-ops/blog-d5-source-generalization`

## Why this slice exists

Defect **D5** (`reports/blog-audit-findings.md`): blog prose names a PARTIAL
source set ("verified platforms including G2, Gartner Peer Insights, and
PeerSpot, alongside community discussions on Reddit") that omits sources the post
actually quotes (e.g. Slashdot, Software Advice). A corpus grep found the at-risk
phrasing in **52 of 78 posts** -- so D5 is corpus-wide, not the single instance
the catalog implied.

Per maintainer direction: **thin-slice the generator-generalization NOW** (so new
posts stop naming a partial set); the 52 published posts are cleaned in the
Phase-2 deep pass, not here.

Root cause: the generator hands the LLM the per-platform `source_distribution.sources`
list in the payload `data_context`; the model narrates a partial subset of it as
"the data comes from X, Y, Z".

## Scope (this PR)

Generator-only. Make the LLM's source guidance generalized + deterministic.

### Files touched

- `plans/PR-Blog-D5-Source-Generalization.md`
- `ATLAS-HARDENING.md`
- `atlas_brain/autonomous/tasks/b2b_blog_post_generation.py`
- `extracted_content_pipeline/autonomous/tasks/b2b_blog_post_generation.py`
- `tests/test_b2b_blog_post_generation.py`

## Mechanism

1. `_gather_data` adds a generalized `data_context["source_description"]` =
   "{verified} reviews from verified review platforms and {community} reviews
   from community forums".
2. `_build_blog_generation_payload` hands the LLM a GENERALIZED
   `source_distribution` (verified/community counts only) -- the per-platform
   `sources` list is dropped from the LLM-facing payload copy, so the model has
   no partial list to enumerate. The STORED `blueprint.data_context` is
   untouched, so the source-mix chart still has the per-platform sources.
3. `_build_coverage_snapshot_note` now uses `source_description` instead of the
   (buggy, partial-by-design) `_top_source_summary` -- a deterministic note.

## Intentional

- **Deterministic lever, not a prompt plea.** Dropping the per-platform list
  from the LLM payload (vs only instructing the model) is what actually prevents
  enumeration -- the model can't name what it doesn't see. The stored context
  keeps `sources` for the chart, so nothing visual is lost.
- **Counts kept.** verified/community counts stay in the payload so the
  "N verified + M community" sentence still has its numbers.
- **No published-post edits.** The 52 corpus posts are Phase-2 (maintainer's
  call); this slice only changes generation going forward.

## Deferred

**Parked hardening** in `ATLAS-HARDENING.md` (separate file per maintainer; root
`HARDENING.md` has the pointer):
- LLM may still mention a platform inferred from a quote `source_name` / chart
  label, or ignore `source_description` -- LLM-fidelity residual, Phase-2 corpus
  check + a candidate detector. M / correctness.
- `_top_source_summary` is now dead code (and had a latent dict-shape bug) --
  remove next pass. S / tech-debt.

Also remaining: the 52-post source-list cleanup (Phase-2 deep pass); D2-followup;
D3-followup.

## Verification

- New `test_coverage_snapshot_note_uses_generalized_source_description`: the note
  contains the generalized "verified review platforms / community forums"
  phrasing and NO platform name (G2/Reddit/Gartner/PeerSpot). Verified it FAILS
  on revert to `_top_source_summary`.
- New `test_blog_payload_generalizes_source_distribution_for_llm`: the LLM
  payload's `source_distribution` has no per-platform `sources` (counts +
  `source_description` only); the STORED `data_context` still has `sources` for
  the chart.
- `pytest` generation + quote-gate suites -> 201 passed; both copies
  byte-identical.

## Estimated diff size

| Area | LOC |
|---|---:|
| 2 generator copies (source_description + payload generalize + coverage note) | ~60 |
| Tests (2 deterministic + a shared fixture) | ~55 |
| ATLAS-HARDENING.md (2 parked entries) | ~16 |
| Plan doc | ~95 |
| **Total** | **~225** |
