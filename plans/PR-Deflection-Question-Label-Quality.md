# PR-Deflection-Question-Label-Quality

## Why this slice exists

#1566 proved the Zendesk product-shaped path but deliberately preserved two
buyer-facing output defects in the committed proof artifact: seeded subject
prefixes such as `[Atlas seed 10] Login and MFA access issue` leaked into FAQ
headings, and the weak source-policy label `What should I do about atla?`
appeared repeatedly. The user asked for the next vertical slice after that
review, and this is the narrow upstream fix before treating the Zendesk proof
as buyer-ready question-label evidence.

## Scope (this PR)

Ownership lane: content-ops/deflection-product-proof
Slice phase: Vertical slice

1. Clean provenance-like ticket subject prefixes before FAQ customer-question
   extraction so the label keeps the real customer question and drops seeded or
   tracker metadata.
2. Fix the upstream support-ticket tokenizer/cluster preview path that turned
   non-plural final-`s` words such as `Atlas` and `status` into `atla` and
   `statu`, and stop bracketed seed metadata from dominating cluster anchors.
3. Add focused regression coverage for the cited Zendesk proof examples plus
   held-out same-class cases the reviewer did not provide.

### Files touched

- `extracted_content_pipeline/support_ticket_clustering.py`
- `extracted_content_pipeline/ticket_faq_markdown.py`
- `plans/PR-Deflection-Question-Label-Quality.md`
- `tests/test_extracted_support_ticket_input_package.py`
- `tests/test_extracted_ticket_faq_markdown.py`

### Review Contract

- Acceptance criteria:
  - [ ] FAQ headings extracted from seeded Zendesk-style title plus question
        text no longer include the bracketed seed or theme prefix.
  - [ ] Customer wording with legitimate unbracketed context before a question
        word stays intact instead of being truncated to a generic tail.
  - [ ] The committed Zendesk product corpus no longer clusters or renders FAQ
        labels from the synthetic `Atlas seed` metadata token.
  - [ ] Existing safe representative labels, including short acronyms such as
        `VPN` and `GDPR`, still publish when they are specific enough.
  - [ ] The committed Zendesk proof docs still honestly describe the existing
        artifact boundary; this slice fixes the generator path, not historical
        proof files.
- Affected surfaces: deterministic support-ticket FAQ Markdown generation and
  extracted pipeline tests.
- Risk areas: buyer-facing output quality, false-positive question rejection,
  and source-policy fallback specificity.
- Reviewer rules triggered: R1, R2, R10, R13, R14.

## Mechanism

The FAQ label path already prefers customer wording before source-policy
fallbacks. This slice adds a small normalization step inside that upstream
question-candidate path: strip leading bracketed seed/tracker metadata and, only
when that metadata was actually stripped, offer the later natural question as a
candidate before the metadata-bearing text. Ordinary unbracketed customer
phrasing is not split on embedded question words.

The `atla` label root is upstream of FAQ Markdown formatting. The shared support
ticket tokenizer used to strip a trailing `s` from any token longer than three
characters, so `Atlas` became `atla` and `status` became `statu`. This slice
tightens that plural stripping and removes leading bracketed seed metadata from
cluster-preview token input, so source-policy labels are built from actual
ticket topics instead of the synthetic corpus marker.

## Intentional

- This does not rewrite the committed #1566 proof artifact. The proof doc
  intentionally records historical defects from that run; rewriting it would
  hide the evidence that motivated this slice.
- This does not trust arbitrary ticket subjects or structured fields as safe
  representative vocabulary. Labels still need documentation or injected
  taxonomy terms, preserving the prior root-cause fix for unlisted structured
  vocabulary.
- This does not add a short-acronym denylist. The review showed that such a
  gate collapses real topics like `VPN`, `SSL`, and `GDPR`; this PR fixes the
  tokenizer/metadata root instead.
- This does not resume embedding-booster work. The booster remains outside this
  lane until the enablement decision is made separately.

## Deferred

- Re-run the full Zendesk product proof artifact after this generator fix in a
  follow-up validation slice, so the historical proof remains intact and the new
  artifact can show buyer-ready labels from a fresh run.

Parked hardening: none.

## Verification

- `PATH=/home/juan-canfield/Desktop/Atlas/.venv/bin:$PATH pytest tests/test_extracted_ticket_faq_markdown.py tests/test_extracted_support_ticket_input_package.py tests/test_extracted_content_deflection_submit.py::test_deflection_submit_surfaces_cluster_preview_for_messy_untagged_csv -q` -- 351 passed.
- `PATH=/home/juan-canfield/Desktop/Atlas/.venv/bin:$PATH bash scripts/run_extracted_pipeline_checks.sh` -- 4236 passed, 10 skipped.

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/support_ticket_clustering.py` | 20 |
| `extracted_content_pipeline/ticket_faq_markdown.py` | 42 |
| `plans/PR-Deflection-Question-Label-Quality.md` | 108 |
| `tests/test_extracted_support_ticket_input_package.py` | 20 |
| `tests/test_extracted_ticket_faq_markdown.py` | 189 |
| **Total** | **379** |
