# PR: Support Ticket Provider Live Smoke Source

## Why this slice exists

The support-ticket provider lane has deterministic execute coverage for FAQ and
landing/blog generation. The remaining operator gap is practical live testing:
the existing live Content Ops smoke script can hit the DB-backed landing/blog
services, but it only uses hand-built default inputs. We need a thin real-path
smoke option that feeds the packaged support-ticket CSV through the Atlas
support-ticket provider before executing landing-page or blog-post generation.

This slice is over the normal 400 LOC target because live export inspection
found a data-truthfulness issue in the first version: the support-ticket blog
blueprint inherited generic benchmark counts that were not present in the CSV.
Fixing that at the source requires CSV-derived blueprint facts plus regression
coverage in the same PR that introduced the live smoke path.

## Scope (this PR)

Ownership lane: content-ops/support-ticket-input-provider

Slice phase: Functional validation

1. Add a live smoke CLI option for support-ticket CSV source rows.
2. Package those rows through the Atlas support-ticket input provider before
   calling the existing Content Ops executor.
3. Keep blog blueprint seeding aligned with provider-produced support-ticket
   filters when the smoke runs `--output blog_post`.
4. Derive the default support-ticket blog blueprint counts and clusters from
   the uploaded CSV rows so the live smoke cannot invent evidence.
5. Fail clearly when `--support-ticket-csv` points at a non-ticket-shaped CSV
   instead of running with an unexpanded provider noop.
6. Document the live smoke command and the Haiku OpenRouter model override for
   lower-cost testing.

### Files touched

- `scripts/smoke_content_ops_live_generation.py`
- `tests/test_smoke_content_ops_live_generation.py`
- `extracted_content_pipeline/README.md`
- `plans/PR-Support-Ticket-Provider-Live-Smoke-Source.md`

## Mechanism

`--support-ticket-csv` loads CSV rows as request `source_material`. Once a
tenant scope exists, the script calls `build_content_ops_input_provider()` and
merges the provider package into the execute payload using the extracted
input-provider merge helper. This mirrors the hosted control-surface path while
still using the script's direct executor call.

For blog smoke runs, the default seeded blueprint switches to
`content_ops_support_ticket_faq` when `--support-ticket-csv` is present, so the
provider-produced filter and seeded row agree.

The support-ticket blog blueprint derives source row count, question-like row
count, and top pain/category clusters from the CSV rows. This prevents the
smoke harness from reusing generic benchmark numbers that are not present in
the uploaded ticket source.

If the Atlas support-ticket provider classifies the CSV request as a noop, the
script raises before execution. That keeps operator mistakes visible and avoids
a confusing run that has raw `source_material` but none of the provider's FAQ
Report defaults.

## Intentional

- No production route changes. This only improves the operator smoke harness.
- No live test is added to CI; tests use fake services and the existing executor.
- No FAQ generator changes. FAQ output remains owned by the FAQ session.
- No file-ingestion import path changes. The source is an already-loaded CSV,
  matching the support-ticket provider contract.

## Deferred

- Uploaded file import -> provider -> landing/blog generation remains blocked
  on the file-ingestion/import lookup lane.
- Full live validation beyond the packaged CSV can follow after this
  support-ticket source mode lands.
- Parked hardening: none. `HARDENING.md` was scanned; existing FAQ scale and
  file-ingestion concurrency entries are outside this live smoke source slice.

## Verification

- Focused smoke-script tests for `tests/test_smoke_content_ops_live_generation.py`
  - 14 passed.
- Py compile for the smoke script and test file - passed.
- Git whitespace check - passed.
- Local PR review wrapper - passed.
- Manual landing-page live smoke with the packaged support-ticket CSV and
  Haiku OpenRouter model - passed; saved landing-page draft
  `06e680c5-7ab4-4bff-8c94-24bd2cd1c969`.
- Manual blog-post live smoke with the packaged support-ticket CSV and Haiku
  OpenRouter model - passed; seeded blueprint
  `4bfe5289-8aaf-42a0-9baf-800f5f2b23b2` and saved blog draft
  `5a4b5ec9-ffcb-470b-955f-49e1e3302aaf`.
- Follow-up blog-post live smoke after CSV-derived blueprint fix - passed;
  seeded blueprint `8d4abd7b-4609-4f23-9664-812659e6c2fe` and saved blog draft
  `57b5d81c-987a-4e38-a808-37c8a4ccfee8`. Export confirmed the draft carries
  `source_row_count=4`, `question_like_ticket_count=2`, the two real CSV
  clusters, and no old `186` / `78` / `42%` benchmark numbers.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan | ~105 |
| Script | ~270 |
| Tests | ~170 |
| Docs | ~30 |
| **Total** | **~575** |
