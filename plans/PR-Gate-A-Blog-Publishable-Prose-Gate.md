# PR-Gate-A-Blog-Publishable-Prose-Gate

## Why this slice exists

Gate A live output-quality proof exported an approved blog draft whose opening
read like internal harness narration: "The uploaded CSV contains 36
support-ticket rows...". That can pass structural blog checks while still being
unshippable customer-facing prose.

This slice prevents that class in the initial support-ticket prompt, promotes
debug/source-upload narration into a blog quality blocker, then uses the
existing blog repair loop to rewrite misses into publishable prose before
persistence.

## Scope (this PR)

Ownership lane: content-ops/gate-a-output-quality
Slice phase: Production hardening

1. Detect internal upload/source narration in blog title, description, content,
   FAQ answers, and metadata text that would be visible to readers.
2. Add publishable-prose instructions to the initial support-ticket descriptive
   prompt so the model does not open by narrating uploads, rows, exports, or
   source mechanics.
3. Feed the blocker into the existing blog quality check/repair path.
4. Add focused tests for block-without-repair, repair-with-guidance, varied
   source-mechanics phrasings, and near-misses that can mention support-ticket
   evidence without debug narration.

### Review Contract

- Acceptance criteria:
  - [ ] A blog draft that opens with "The uploaded CSV contains..." is
        `quality_blocked` and not saved when repair is disabled.
  - [ ] With repair enabled, the retry prompt names the debug-prose blocker and
        asks for publishable customer-facing prose while preserving evidence.
  - [ ] The initial support-ticket prompt asks for publishable customer-facing
        prose and forbids opening with upload/CSV/export/row/source mechanics.
  - [ ] Realistic variants such as "The uploaded CSV reveals...", "Analysis of
        the 36 support tickets reveals...", and "This export of 36 support
        tickets surfaces..." are blocked.
  - [ ] A repaired draft that describes support-ticket evidence in publishable
        language persists.
  - [ ] A near-miss sentence such as "support tickets show account and
        reporting questions" remains allowed.
- Affected surfaces: blog generation support-ticket quality blockers, repair
  guidance, focused extracted blog tests.
- Risk areas: over-blocking grounded support-ticket evidence, adding prompt-only
  guidance without deterministic enforcement, reintroducing false-green output.
- Reviewer rules triggered: R1, R2, R10.

### Files touched

- `extracted_content_pipeline/blog_generation.py`
- `plans/PR-Gate-A-Blog-Publishable-Prose-Gate.md`
- `tests/test_extracted_blog_generation.py`
- `tests/test_smoke_content_ops_live_generation.py`

## Mechanism

Add the publishable-prose rule to the existing support-ticket descriptive
prompt addendum and remove "uploaded support tickets" from the generated H2
contract so prevention happens before the LLM writes the first draft.

Add a deterministic helper near the existing support-ticket blog blockers. It
scans public blog text for source-debug lead-ins such as:

```text
The uploaded CSV contains ...
The uploaded CSV reveals ...
Analysis of the 36 support tickets reveals ...
This export of 36 support tickets surfaces ...
```

The helper returns a namespaced blocker
`support_ticket_generated_content:debug_source_narration`. Blog quality checks
append that blocker to the existing support-ticket generated-content blockers.
The repair guidance tells the LLM to rewrite the prose for a customer-facing
article without narrating the upload, CSV, rows, or internal source mechanics.

## Intentional

- This is still not a broad style classifier. Prevention is in the initial
  prompt; the deterministic backstop blocks the realistic source-mechanics
  openings from review while keeping normal support-ticket evidence allowed.
- This does not change support-ticket evidence grounding. The repaired blog may
  still cite ticket counts and clusters; it just must not read like a harness
  report about uploaded files.
- This does not rerun Gate A live validation. The messy-ticket rerun remains
  deferred until the structural output-quality fixes land.

## Deferred

- Gate A landing-page variant distinctness remains the next structural quality
  fix.
- Gate A messy-ticket rerun remains deferred until blog and landing-page output
  quality blockers are in place.
- Issue #1357's report/email_campaign full-factory proof remains deferred until
  the known 3-generator quality misses are structurally addressed.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_extracted_blog_generation.py -q` - PASS (`94 passed`).
- `python -m pytest tests/test_smoke_content_ops_live_generation.py::test_support_ticket_blog_blueprint_payload_uses_csv_counts -q` - PASS (`1 passed`).
- `bash scripts/validate_extracted_content_pipeline.sh` - PASS.
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline` - PASS.
- `python scripts/audit_extracted_standalone.py --fail-on-debt` - PASS (`Atlas runtime import findings: 0`).
- `bash scripts/check_ascii_python.sh` - PASS.
- `bash scripts/run_extracted_pipeline_checks.sh` - PASS (`extracted_reasoning_core`: `295 passed`; `extracted_content_pipeline`: `3303 passed, 10 skipped`; one `pynvml` warning).
- `bash scripts/local_pr_review.sh --current-pr-body-file tmp/gate-a-blog-publishable-prose-gate-pr-body.md` - PASS (no non-diff caller references).

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/blog_generation.py` | 71 |
| `plans/PR-Gate-A-Blog-Publishable-Prose-Gate.md` | 122 |
| `tests/test_extracted_blog_generation.py` | 159 |
| `tests/test_smoke_content_ops_live_generation.py` | 4 |
| **Total** | **356** |
