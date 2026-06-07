# PR-Gate-A-Sales-Brief-Type-Lock

## Why this slice exists

Gate A live validation found that sales brief generation accepted
`inputs.brief_type=renewal`, but all exported sales briefs stored
`brief_type=pre_call` because the model-returned JSON field won over the
requested/default brief type. That breaks the output contract: the operator can
ask for a renewal brief and still persist a pre-call brief if the model drifts.

This slice closes the parked `HARDENING.md` finding
`Sales brief live generation drifts from requested renewal brief type` by making
the requested/default brief type authoritative whenever it is provided.

## Scope (this PR)

Ownership lane: content-ops/gate-a-output-quality
Slice phase: Production hardening

1. Change sales brief draft construction so a non-empty per-call
   `default_brief_type` wins over the LLM's parsed `brief_type`.
2. Preserve the construction-time default fallback when the per-call override is
   absent.
3. Keep the LLM's parsed `brief_type` only as a fallback when no requested/default
   type exists.
4. Update focused sales-brief generation tests to prove the requested brief type
   overrides a conflicting model field.
5. Remove the resolved Gate A hardening entry.

### Review Contract

- Acceptance criteria:
  - [ ] A generation request with `default_brief_type="renewal"` persists
        `brief_type="renewal"` even when the LLM returns `"pre_call"`.
  - [ ] The existing config default fallback still fills `brief_type` when the
        per-call default and model field are absent.
  - [ ] Variant-angle metadata, quality checks, source material routing, and
        persistence calls are unchanged.
  - [ ] The resolved `HARDENING.md` entry is removed without changing unrelated
        parked items.
- Affected surfaces: extracted content pipeline sales brief generation; Gate A
  validation quality; persisted sales brief metadata.
- Risk areas: backcompat for callers that intentionally trusted model-provided
  brief types; test/CI enrollment for extracted package checks.
- Reviewer rules triggered: R1, R2, R10.

### Files touched

- `HARDENING.md`
- `extracted_content_pipeline/sales_brief_generation.py`
- `plans/PR-Gate-A-Sales-Brief-Type-Lock.md`
- `tests/test_extracted_sales_brief_generation.py`

## Mechanism

`SalesBriefGenerationService.generate(...)` already resolves the requested
brief type from `generation_plan.py` into the `default_brief_type` argument and
passes it to `_build_draft(...)`. The bug is only the precedence inside
`_build_draft(...)`.

The draft builder now resolves:

```text
per-call requested brief type -> model brief_type -> service config default
```

That keeps the requested run shape load-bearing. If no caller provides a
per-call requested type, the model field still works as before; if neither
exists, the existing `SalesBriefGenerationConfig.default_brief_type` fallback
remains.

## Intentional

- This does not add a new `brief_type` validator or enum. Existing callers and
  stored rows already use string values; this slice only fixes precedence.
- This does not change the prompt. The model may still emit a conflicting
  `brief_type`; the persisted contract ignores that conflict when the caller
  supplied an explicit requested/default type.
- No live Gate A rerun is included. The focused regression test proves the
  precise bug; a future Gate A rerun can validate output quality end to end.
- Cross-layer caller hints were inspected. Host wiring references are
  unaffected because the `SalesBriefGenerationService` constructor and
  `generate(...)` port are unchanged; the full extracted check covered
  content-ops execution and sales-brief export callers. Same-name
  `_build_draft` hints in other generators are regex false positives, not
  shared call sites.

## Deferred

- Gate A rerun on messy tickets remains parked separately in `HARDENING.md`.
- Broader brand-voice POV failures and samey variant quality are separate Gate A
  hardening slices.

Parked hardening: none.

## Verification

- `python -m pytest tests/test_extracted_sales_brief_generation.py -q` -
  PASS; `30 passed in 0.10s` after rebasing onto current `origin/main`.
- `bash scripts/validate_extracted_content_pipeline.sh` - PASS; mapped files
  matched `atlas_brain` sources and hard Atlas imports were clean.
- `python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline`
  - PASS; `forbid_atlas_reasoning_imports: clean`.
- `python scripts/audit_extracted_standalone.py --fail-on-debt` - PASS;
  `Atlas runtime import findings: 0`.
- `bash scripts/check_ascii_python.sh` - PASS; ASCII check passed for
  `extracted_content_pipeline` Python files.
- `bash scripts/run_extracted_pipeline_checks.sh` - PASS; package wrapper
  completed with `3250 passed, 10 skipped, 1 warning`.
- Pending before push: `bash scripts/local_pr_review.sh --current-pr-body-file <body-file>`.

## Estimated diff size

| File | LOC |
|---|---:|
| `HARDENING.md` | 9 |
| `extracted_content_pipeline/sales_brief_generation.py` | 13 |
| `plans/PR-Gate-A-Sales-Brief-Type-Lock.md` | 120 |
| `tests/test_extracted_sales_brief_generation.py` | 24 |
| **Total** | **166** |
