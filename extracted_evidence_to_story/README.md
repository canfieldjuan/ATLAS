# Extracted Evidence-to-Story

Standalone product boundary for source-traceable nonfiction storytelling.

This package is separate from AI Content Ops. AI Content Ops owns campaign,
podcast, and content repurposing workflows. Evidence-to-Story owns the
true-crime / mystery narrative nonfiction pipeline described in
`docs/evidence_to_story_v0_build_contract.md`.

## Current Surface

Stage 1 only:

- Load a v0 manifest with exactly one YouTube transcript and one news article.
- Resolve manifest-relative text paths.
- Normalize records into `sources.json` with stable source IDs.
- Write the first file in the `story_package/` directory.

No claim extraction, timeline generation, angle selection, script drafting,
voice direction, LLM calls, database calls, or Atlas imports are implemented in
this slice.

## Example

```bash
python scripts/build_evidence_to_story_sources.py \
  extracted_evidence_to_story/fixtures/evidence_to_story_v0_golden/inputs/manifest.json \
  --output-dir story_package
```

Run checks:

```bash
bash scripts/run_extracted_evidence_to_story_checks.sh
```
