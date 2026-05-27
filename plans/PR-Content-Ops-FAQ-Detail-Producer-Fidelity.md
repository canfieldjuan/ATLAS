# PR-Content-Ops-FAQ-Detail-Producer-Fidelity

## Why this slice exists

PR #1027 tightened the hosted FAQ detail checker so it validates generated FAQ
item rows, but the reviewer noted one remaining drift risk: the checker uses
hardcoded field lists while its tests used hand-built item fixtures. If the
generator changes its item shape later, the remote checker contract could lag
until someone manually notices.

This slice adds the missing producer-fidelity guard. It proves the checker item
contract still matches `build_ticket_faq_markdown(...)`, without importing the
package from the checker script itself.

## Scope (this PR)

Ownership lane: content-ops/faq-search
Slice phase: Robust testing

1. Add a focused test that builds a generated FAQ item with
   `build_ticket_faq_markdown(...)`, JSON-roundtrips it into hosted-route shape,
   and validates it with the checker item validator.
2. Assert the checker’s hardcoded item field set equals the producer’s current
   item keys.
3. Assert the checker’s term-mapping field set equals a producer-generated term
   mapping when vocabulary-gap evidence is present.
4. Keep runtime code unchanged; this is a drift guard for the existing checker.

### Files touched

| File | Purpose |
|---|---|
| `plans/PR-Content-Ops-FAQ-Detail-Producer-Fidelity.md` | Plan contract for this producer-fidelity test slice. |
| `tests/test_check_content_ops_faq_search_route_contract.py` | Add the producer-backed fidelity test for the hosted detail item contract. |

## Mechanism

The test imports `build_ticket_faq_markdown(...)` from
`extracted_content_pipeline.ticket_faq_markdown`, builds a small support/search
fixture that produces both generated FAQ item fields and a vocabulary mapping,
then runs `json.dumps`/`json.loads` to mimic the hosted route JSON boundary.

It derives the expected field sets from the checker module constants and
compares them to the producer output before calling `_validate_detail_item`.
That catches both removed/renamed required fields and newly added producer
fields that the remote checker has not learned to enforce.

## Intentional

- No runtime checker imports from `extracted_content_pipeline`; only the test
  imports the producer.
- No generated FAQ behavior changes.
- No broader schema/export work; this is the smallest drift guard for the
  item-level contract.

## Deferred

- Parked hardening: `Reject Contradictory FAQ Route Detail Concurrency Flags`
  remains parked in `HARDENING.md`; it is unrelated concurrency runner polish.
- Formal generated schema export remains deferred until the hosted API schema
  is generated from the FastAPI app.

## Verification

- python -m pytest tests/test_check_content_ops_faq_search_route_contract.py -q — 81 passed.
- python -m py_compile tests/test_check_content_ops_faq_search_route_contract.py — passed.
- python scripts/audit_plan_code_consistency.py plans/PR-Content-Ops-FAQ-Detail-Producer-Fidelity.md — passed.
- git diff --check — passed.
- python scripts/audit_extracted_pipeline_ci_enrollment.py . — 122 matching tests enrolled.
- bash scripts/validate_extracted_content_pipeline.sh — passed.
- python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline — passed.
- python scripts/audit_extracted_standalone.py --fail-on-debt — passed.
- bash scripts/check_ascii_python.sh — passed.
- bash scripts/run_extracted_pipeline_checks.sh — 2532 passed, 7 skipped, 1 warning.

## Estimated diff size

| Area | LOC |
|---|---:|
| Plan doc | 82 |
| Producer-fidelity test | 62 |
| **Total** | **144** |
