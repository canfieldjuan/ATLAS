Plan: PR-Content-Ops-FAQ-Output-Contract

## Why this slice exists

The support-ticket FAQ path now runs end to end, but the packaged demo fixture
does not prove the output contract the product describes: customer vocabulary,
intent condensation, and actionable next steps. The default CSV has two
unrelated tickets and no customer-worded questions, so the deterministic FAQ
builder can prove lifecycle wiring while still reporting weak output checks.

This slice tightens the source fixture and host CLI so the quick FAQ demo can
be used as a real product confidence check, not only a persistence smoke.

## Scope (this PR)

1. Replace the packaged support-ticket CSV with a small repeated-intent
   fixture that exercises customer wording, volume ranking, and condensation.
2. Update FAQ Markdown tests to assert the packaged fixture passes the
   three-point output checks.
3. Add a host-facing `--require-output-checks` CLI flag to fail the Markdown
   build when any output check is false.
4. Update docs for the stricter FAQ demo command.
5. Remove the merged FAQ lifecycle smoke row from the coordination ledger and
   claim this slice.

### Files touched

- `plans/PR-Content-Ops-FAQ-Output-Contract.md`
- `docs/extraction/coordination/inflight.md`
- `extracted_content_pipeline/examples/support_ticket_sources.csv`
- `extracted_content_pipeline/README.md`
- `extracted_content_pipeline/STATUS.md`
- `scripts/build_extracted_ticket_faq_markdown.py`
- `tests/test_extracted_campaign_source_adapters.py`
- `tests/test_extracted_ticket_faq_markdown.py`
- `tests/test_smoke_content_ops_faq_lifecycle.py`
- `tests/test_smoke_content_ops_source_file_postgres.py`

## Mechanism

The FAQ builder already emits `output_checks` with the three product checks:
`uses_user_vocabulary`, `condensed`, and `has_action_items`. This PR does not
add a second quality layer. It makes the packaged support-ticket fixture
exercise those checks and lets operators opt into failing the CLI if the checks
do not all pass:

```bash
python scripts/build_extracted_ticket_faq_markdown.py ... --require-output-checks
```

The CLI still writes the same Markdown artifact. The new flag only changes the
exit contract when checks fail.

## Intentional

- No LLM call is introduced. FAQ Markdown remains deterministic and extractive.
- The default `build_extracted_ticket_faq_markdown.py` behavior remains
  backwards-compatible unless `--require-output-checks` is supplied.
- The fixture stays small. It is a confidence demo, not a large benchmark.
- The stale PR #666 coordination row is removed here because this PR touches
  the same ledger before opening a new PR.

## Deferred

- A hosted UI review/publish button remains separate from this command-line
  confidence path.
- Broader semantic clustering remains separate from the current deterministic
  intent-rule grouping.
- Real customer help desk exports should still drive future source alias work.

## Verification

- `pytest tests/test_extracted_campaign_source_adapters.py::test_packaged_support_ticket_csv_example_loads_provider_labels tests/test_extracted_campaign_source_adapters.py::test_build_sources_cli_converts_packaged_support_ticket_csv tests/test_smoke_content_ops_faq_lifecycle.py::test_faq_lifecycle_smoke_generates_exports_reviews_and_reexports tests/test_smoke_content_ops_source_file_postgres.py::test_source_file_postgres_smoke_imports_and_persists tests/test_extracted_ticket_faq_markdown.py` - 45 passed
- `python -m py_compile scripts/build_extracted_ticket_faq_markdown.py tests/test_extracted_ticket_faq_markdown.py`
- `python scripts/build_extracted_ticket_faq_markdown.py extracted_content_pipeline/examples/support_ticket_sources.csv --source-format csv --require-output-checks --output /tmp/support_ticket_faq.md`
- `git diff --check`
- `bash scripts/run_extracted_pipeline_checks.sh` - 1526 passed, 1 existing torch/pynvml warning
- `bash scripts/local_pr_review.sh`

## Estimated diff size

| File | Estimated LOC |
|---|---:|
| `plans/PR-Content-Ops-FAQ-Output-Contract.md` | +82 |
| `docs/extraction/coordination/inflight.md` | +2 / -2 |
| `extracted_content_pipeline/examples/support_ticket_sources.csv` | +4 / -2 |
| `extracted_content_pipeline/README.md` | +7 / -3 |
| `extracted_content_pipeline/STATUS.md` | +5 / -4 |
| `scripts/build_extracted_ticket_faq_markdown.py` | +12 |
| `tests/test_extracted_campaign_source_adapters.py` | +9 / -6 |
| `tests/test_extracted_ticket_faq_markdown.py` | +55 / -16 |
| `tests/test_smoke_content_ops_faq_lifecycle.py` | +6 / -1 |
| `tests/test_smoke_content_ops_source_file_postgres.py` | +21 / -7 |
| **Total** | **258** |

This is below the 400 LOC review budget.
