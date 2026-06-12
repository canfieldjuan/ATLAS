# PR-Content-Ops-Corroboration-Surfacing

## Why this slice exists

Slice 6 (#1488) folds every adversarial finding into the verdict as evidence,
but it gives the editor no way to tell a one-off objection from one that two
independent passes both raised -- which is the strongest signal in the whole
adversarial design (the doc calls independent agreement the thing an editor
should act on). This slice surfaces that: the verify result reports the
objection categories raised by two or more independent passes, and the rendered
verdict shows them as a "Corroborated objections" section. Deterministic, no
server-side LLM.

## Scope (this PR)

Ownership lane: content-ops/review-contract
Slice phase: Functional validation

1. Package: add `corroborated_categories_across(passes, *, min_passes=2)` -- the
   N-pass generalization of slice 5b's pairwise `corroborated_categories`.
2. Host: the result gains `corroborated_objection_categories` (sorted category
   values), surfaced in `as_dict`.
3. Render: the ChatGPT adapter's verdict text shows a "Corroborated objections
   (raised by 2+ passes)" section when non-empty.

### Files touched

- `atlas_brain/_content_ops_review_workflow.py`
- `atlas_brain/mcp/content_ops_marketer_verify_chatgpt_adapter_server.py`
- `extracted_content_pipeline/adversarial_pass.py`
- `plans/INDEX.md`
- `plans/PR-Content-Ops-Corroboration-Surfacing.md`
- `plans/archive/PR-Content-Ops-Render-Messageless-Blocking.md`
- `tests/test_atlas_content_ops_review_workflow.py`
- `tests/test_extracted_content_adversarial_pass.py`
- `tests/test_mcp_content_ops_marketer_verify.py`

### Review Contract

Acceptance criteria:
- A category is corroborated only when at least `min_passes` (default 2) distinct
  passes raise it among their substantiated findings; a single pass raising a
  category twice does not self-corroborate.
- Unsubstantiated findings never contribute to corroboration.
- The result reports sorted category values; the render shows them only when
  non-empty and tolerates a malformed (non-list) shape.
- `min_passes < 1` raises `ValueError`.
- No verdict logic changes -- corroboration is reported, not acted on.

Affected surfaces: the adversarial-pass package, the host result, and the
ChatGPT adapter render. No change to the verify MCP tool args, transport, or the
verdict computation.

Risk areas: distinct-pass counting; value-based matching across enum/decoded
strings; render guard.

Reviewer rules triggered: R1, R2, R5, R10, R14.

## Mechanism

`corroborated_categories_across` counts, per category, the number of distinct
passes that raised it among their substantiated findings (a per-pass set, so a
single pass counts once), and returns the categories meeting `min_passes`.
Counting is value-based, so a category decoded as a plain string in one pass
corroborates the enum member in another. The host helper
`_corroborated_categories_for_request` runs it over the request's adversarial
passes and returns sorted category values (stable, JSON-friendly), attached to
both the verdict and blocked-result paths and serialized in `as_dict`. The
adapter's `_corroborated_lines` reads the serialized list (guarding non-list
input) and `_verdict_text` renders it as its own section.

## Intentional

- **Reported, not acted on.** Corroboration is a prioritization signal for the
  editor; it does not change the deterministic verdict (consistent with the
  "model is evidence, not judge" rule and slice 6's never-blocking findings).
- **Distinct-pass counting, substantiated only.** Self-corroboration within one
  pass would be meaningless, and unsubstantiated findings are already excluded
  from the folded comments -- corroboration uses the same filter for consistency.
- **Sorted category values in the result.** A `frozenset` is unordered;
  surfacing sorted values keeps the output stable and diff-friendly.

## Deferred

- Tenant calibration-library reader port (from #1489) -- server-side persisted
  anchors; the operational follow-up.
- Surfacing the corroborating pass ids / sources alongside each category.
- Parked hardening: none new this slice.

## Verification

- Reviewer rules triggered: R1, R2, R5, R10, R14.
- Passed: pytest of the adversarial, host-workflow, and verify-MCP test files --
  114 passed (6 package + 2 host + 2 render fixtures new).
- Passed: python3 scripts/audit_extracted_pipeline_ci_enrollment.py -- OK, 168 enrolled.
- Passed: python3 scripts/audit_extracted_standalone.py --fail-on-debt -- 0 findings.
- Passed: python3 extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline -- clean.
- Passed: bash scripts/check_ascii_python.sh -- ASCII check passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `atlas_brain/_content_ops_review_workflow.py` | 23 |
| `atlas_brain/mcp/content_ops_marketer_verify_chatgpt_adapter_server.py` | 10 |
| `extracted_content_pipeline/adversarial_pass.py` | 26 |
| `plans/INDEX.md` | 3 |
| `plans/PR-Content-Ops-Corroboration-Surfacing.md` | 100 |
| `plans/archive/PR-Content-Ops-Render-Messageless-Blocking.md` | 0 |
| `tests/test_atlas_content_ops_review_workflow.py` | 35 |
| `tests/test_extracted_content_adversarial_pass.py` | 59 |
| `tests/test_mcp_content_ops_marketer_verify.py` | 27 |
| **Total** | **283** |
