# Extracted Evidence-to-Story Status

## Current State

- Product boundary exists at `extracted_evidence_to_story/`.
- The parked Evidence-to-Story design docs and golden fixture live inside this
  product boundary, not inside AI Content Ops.
- `sources.py` implements the deterministic Stage-1 source loader from the v0
  build contract.
- `scripts/build_evidence_to_story_sources.py` writes `sources.json` for a
  story package or prints the same payload to stdout.

## Not Implemented

- Claim extraction
- Timeline generation
- Entity extraction
- Angle proposal
- Outline generation
- Script drafting
- Citation validation
- Voice direction
- Audio rendering

Those stages should land as separate slices on this package, not in
`extracted_content_pipeline`.
