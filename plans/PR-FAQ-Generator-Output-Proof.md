# PR-FAQ-Generator-Output-Proof

## Why this slice exists

The FAQ lane has proven ingestion, scale, search, vocabulary-gap wiring, and
output checks across separate slices. The next useful step is an end-to-end
output proof that a reviewer can inspect without reconstructing the CLI command
or reading a megabyte-scale Markdown artifact.

This slice proves the deterministic FAQ generator can take representative
support-ticket, search-log, and sales-objection shaped rows and produce usable
FAQ Markdown with intent ranking, source IDs, action steps, support guidance,
and vocabulary-gap suggestions.

The diff is expected to exceed the usual 400 LOC target because splitting the
runner from its negative checker fixtures and CI enrollment would recreate the
metadata/enrollment drift this workflow is meant to avoid.

## Scope (this PR)

Ownership lane: content-ops/faq-generator

Slice phase: Functional validation.

Slice size: medium because this is one vertical FAQ proof path with known
boundaries and local blast radius. It exercises the real CLI/generator flow and
adds artifacts/tests, but it does not change shared contracts, DB schema,
auth/tenant boundaries, CI gates, or blog/landing prompts.

1. Add a focused FAQ output-proof smoke script that writes a representative
   source CSV, runs `scripts/build_extracted_ticket_faq_markdown.py`, and emits
   Markdown, CLI result JSON, and compact proof summary artifacts.
2. Assert the proof output includes ranked FAQ items, customer-worded questions,
   source IDs, action-step counts, support-contact guidance, and vocabulary-gap
   diagnostics.
3. Add fixture tests for the proof runner's success path and failure reporting.
4. Park the non-blocking observation that FAQ action steps are intent-template
   based, not source-resolution gated.

### Files touched

- `HARDENING.md`
- `.github/workflows/extracted_pipeline_checks.yml`
- `plans/PR-FAQ-Generator-Output-Proof.md`
- `scripts/run_extracted_pipeline_checks.sh`
- `scripts/smoke_content_ops_faq_output_proof.py`
- `tests/test_smoke_content_ops_faq_output_proof.py`

## Mechanism

The new proof runner creates a small representative CSV inside an artifact
directory and invokes the canonical FAQ CLI with:

```bash
python scripts/build_extracted_ticket_faq_markdown.py <source.csv> \
  --source-format csv \
  --output faq_output.md \
  --result-output faq_result.json \
  --require-output-checks \
  --support-contact support@example.com \
  --documentation-term "Download report" \
  --documentation-term "Single sign-on setup" \
  --vocabulary-gap-rule "SSO,single sign-on"
```

The proof summary reads the CLI result JSON and generated Markdown, then records
only compact product evidence: status, generated count, output checks, topics,
source-id coverage, step counts, support-contact presence, vocabulary-gap
counts, top customer terms, and artifact paths. The script exits non-zero when
the CLI fails or when required proof predicates are missing.

## Intentional

- No generator behavior changes in this slice. This is proof of the current FAQ
  output path; behavior changes should be follow-up slices.
- No blog/landing prompt or support-ticket evidence-contract edits. Those are
  owned by the adjacent generation-evidence lane.
- No DB lifecycle or search route work. This proof targets Markdown output shape,
  not persistence/retrieval.
- The generated Markdown artifact is local to the run artifact directory, not
  checked into the repo.

## Deferred

- Parked hardening: FAQ answer/action steps are currently deterministic
  intent-template guidance rather than resolution-evidence-gated support
  answers. That is not required for this output-proof runner to function, but it
  matters before positioning generated FAQ answers as publish-ready.
- Future PR: if product wants publish-ready FAQ answers from support tickets,
  gate or label FAQ action steps using the resolution-evidence contract now
  available in the support-ticket package.

## Verification

- `python -m pytest tests/test_smoke_content_ops_faq_output_proof.py -q` - 3 passed.
- `python scripts/smoke_content_ops_faq_output_proof.py --artifact-dir /tmp/atlas-faq-output-proof` - passed.
- `python scripts/audit_extracted_pipeline_ci_enrollment.py` - passed, 118 matching tests enrolled.
- `python -m py_compile scripts/smoke_content_ops_faq_output_proof.py tests/test_smoke_content_ops_faq_output_proof.py` - passed.
- `git diff --check` - passed.
- `python scripts/audit_plan_code_consistency.py plans/PR-FAQ-Generator-Output-Proof.md` - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Proof smoke runner | ~352 |
| Fixture tests | ~116 |
| Hardening note | ~12 |
| CI enrollment | ~3 |
| Plan doc | ~115 |
| **Total** | **~598** |

The estimate is above the usual target because the slice intentionally ships the
proof runner, negative checker fixtures, CI enrollment, and plan together. The
runtime behavior surface remains narrow and additive.
