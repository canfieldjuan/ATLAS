# PR-FAQ-SaaS-Demo-Malformed-Artifact-Gate

## Why this slice exists

#1072 made SaaS demo e2e child artifacts load-bearing and pinned two detector
paths: a missing result artifact and a present artifact with `ok: false`. The
same fail-closed branch also handles malformed JSON and non-object JSON, but
those parser inputs are not pinned by a focused fixture. Atlas' checker/gate
contract says parser/type checks should reject malformed JSON and unrelated
envelopes explicitly, so this slice locks that behavior before more hosted demo
work builds on the wrapper.

## Scope (this PR)

Ownership lane: content-ops/faq-search
Slice phase: Functional validation.

1. Add a focused negative fixture for a route command that exits 0 while writing
   malformed JSON to its result artifact.
2. Prove the wrapper exits 1 from artifact-read status, not from route process
   status.
3. Keep production wrapper behavior unchanged.

### Files touched

- `plans/PR-FAQ-SaaS-Demo-Malformed-Artifact-Gate.md`
- `tests/test_smoke_content_ops_faq_saas_demo_route_e2e.py`

## Mechanism

The existing `_read_json_object(...)` helper returns a compact artifact with
`available: False`, `ok: False`, and a JSON parse diagnostic when a child result
artifact cannot be decoded. `_artifact_status_errors(...)` already folds that
diagnostic into the top-level `errors` list and makes `summary["ok"]` false.

This slice drives that path through the wrapper-level test harness by making
the fake route subprocess exit successfully while writing invalid JSON to
the route result artifact.

## Intentional

- No production code changes; #1072 already implemented the behavior.
- No live hosted route behavior changes.
- No broad malformed-artifact matrix is added. This pins the parser-specific
  branch with the smallest representative fixture.

## Deferred

- Parked hardening: none.
- Non-object JSON rejection is the same `_read_json_object(...)` unavailable
  artifact branch and remains covered by the helper's branch shape rather than
  a second near-duplicate wrapper fixture.

## Verification

- python -m py_compile tests/test_smoke_content_ops_faq_saas_demo_route_e2e.py
  - passed.
- python -m pytest tests/test_smoke_content_ops_faq_saas_demo_route_e2e.py -q
  - 9 passed.
- git diff --check
  - passed.
- python scripts/audit_plan_doc.py plans/PR-FAQ-SaaS-Demo-Malformed-Artifact-Gate.md
  - passed.
- python scripts/audit_plan_code_consistency.py plans/PR-FAQ-SaaS-Demo-Malformed-Artifact-Gate.md
  - passed.
- bash scripts/check_ascii_python.sh
  - passed.
- ATLAS_CURRENT_PR_BODY_FILE=/home/juan-canfield/Desktop/atlas-pr-bodies/faq-saas-demo-malformed-artifact-gate.md bash scripts/local_pr_review.sh
  - passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 77 |
| Tests | 22 |
| **Total** | **99** |
