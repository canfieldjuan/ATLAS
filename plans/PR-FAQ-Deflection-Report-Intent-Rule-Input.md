# PR-FAQ-Deflection-Report-Intent-Rule-Input

## Why this slice exists

The deflection report UI can now run the report and configure vocabulary-gap
inputs, but custom intent rules remain CLI-only. That leaves a gap in the
hosted flow: operators can tune the clustering language in the report CLI, but
cannot send the same intent-to-FAQ mapping rules through the Content Ops execute
route or Intel UI.

This slice adds the thinnest end-to-end hosted path for custom FAQ intent rules:
the control-surface catalog advertises the input, the execute plan threads it
into both faq_markdown and faq_deflection_report, and the Intel UI writes it
into the existing inputs JSON.
The diff is over the 400 LOC soft cap because the input is only useful if the
backend catalog, generation plan, execute dispatch, UI control, and regression
tests land together; splitting those would create a visible control that is not
yet honored or a backend capability that remains raw-JSON-only.

## Scope (this PR)

Ownership lane: content-ops/deflection-report-ui

Slice phase: Product polish

1. Add a hosted FAQ intent-rule input contract shaped as one rule per line:
   `topic=keyword,keyword`.
2. Normalize faq_intent_rules in the generation plan and dispatch it to the
   FAQ Markdown and deflection report services.
3. Render an Intel UI textarea for the intent rules whenever FAQ configuration
   outputs are selected.
4. Add focused tests for backend plan/execute threading and UI JSON writing.

### Files touched

| File | Purpose |
|---|---|
| `extracted_content_pipeline/ticket_faq_input_contract.py` | Adds the intent-rule catalog input contract. |
| `extracted_content_pipeline/ticket_faq_markdown.py` | Adds shared intent-rule normalization for hosted callers. |
| `extracted_content_pipeline/generation_plan.py` | Reads the hosted FAQ intent rules into FAQ step config. |
| `extracted_content_pipeline/content_ops_execution.py` | Dispatches intent-rule config to FAQ services. |
| `extracted_content_pipeline/api/control_surfaces.py` | Adds the intent-rule contract to the hosted catalog payload. |
| `tests/test_extracted_content_generation_plan.py` | Verifies deflection report plan config includes intent rules. |
| `tests/test_extracted_content_ops_execution.py` | Verifies hosted execution applies intent rules to grouping. |
| `tests/test_extracted_content_control_surface_api.py` | Verifies the catalog advertises the intent-rule input. |
| `atlas-intel-ui/src/pages/ContentOpsNewRun.tsx` | Adds the FAQ intent-rule textarea backed by inputs JSON. |
| `atlas-intel-ui/scripts/content-ops-faq-configuration-inputs.test.mjs` | Verifies UI visibility and draft parsing helpers. |
| `atlas-intel-ui/src/domain/contentOps/faqConfigurationInputs.ts` | Adds reusable FAQ intent-rule draft helpers. |
| `atlas-intel-ui/src/domain/contentOps/index.ts` | Exports the FAQ intent-rule draft helpers. |
| `atlas-intel-ui/src/api/__fixtures__/contentOps/catalog.json` | Mirrors the backend catalog fixture for local UI runs. |
| `tests/test_extracted_ticket_faq_markdown.py` | Verifies the shared intent-rule normalizer accepts and rejects the expected shapes. |
| `plans/PR-FAQ-Deflection-Report-Intent-Rule-Input.md` | Documents this slice contract. |

## Mechanism

The hosted input key is faq_intent_rules.

The UI textarea accepts one rule per line:

```txt
data freshness=warehouse sync,connector lag
access setup=invite link,new user
```

The UI stores the draft as a JSON string list so partial textarea typing remains
ergonomic:

```json
[
  "data freshness=warehouse sync,connector lag"
]
```

The backend normalizer accepts both that line-based shape and structured
`{topic, keywords}` objects, then returns the same intent-rule tuple shape used
by the FAQ service. The generation plan keeps the existing rule precedence:
custom hosted rules are prepended before the default rules, so host-provided
customer language wins over default taxonomy matches.

## Intentional

- No rule-file upload is added. File upload semantics need a separate UI/API
  contract; this slice proves inline hosted intent rules first.
- No saved presets/defaults are added. The controls continue writing directly
  into the existing inputs JSON.
- The UI stores line strings rather than objects because textarea controls need
  to preserve partial typing without blocking on every intermediate invalid
  character. The backend still accepts structured objects for API callers.

## Deferred

- Parked hardening considered: `atlas-intel-ui npm audit vulnerabilities`
  remains parked in `HARDENING.md`; dependency upgrade work is outside this
  intent-rule slice.
- Future product-polish slice: add rule-file upload or saved configuration
  presets after the hosted API contract for those workflows is defined.

## Verification

- Command: npm run test:content-ops-faq-configuration-inputs
  - Result: 3 passed.
- Command: npm run lint
  - Result: passed.
- Command: npm run build
  - Result: passed.
- Command: pytest tests/test_extracted_ticket_faq_markdown.py::test_normalize_intent_rules_accepts_line_and_object_shapes tests/test_extracted_ticket_faq_markdown.py::test_normalize_intent_rules_rejects_invalid_shapes tests/test_extracted_content_generation_plan.py::test_plan_threads_custom_faq_intent_rules_to_deflection_report tests/test_extracted_content_ops_execution.py::test_execute_applies_hosted_faq_intent_rules tests/test_extracted_content_control_surface_api.py::test_describe_control_surfaces_route_returns_catalog_and_presets
  - Result: 9 passed.
- Command: pytest tests/test_extracted_ticket_faq_markdown.py tests/test_extracted_content_generation_plan.py tests/test_extracted_content_ops_execution.py tests/test_extracted_content_control_surface_api.py
  - Result: 365 passed, 1 skipped.
- Command: bash scripts/validate_extracted_content_pipeline.sh
  - Result: passed.
- Command: python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline
  - Result: passed.
- Command: python scripts/audit_extracted_standalone.py --fail-on-debt
  - Result: passed.
- Command: bash scripts/check_ascii_python.sh
  - Result: passed.
- Command: python scripts/audit_plan_doc.py plans/PR-FAQ-Deflection-Report-Intent-Rule-Input.md
  - Result: passed.
- Command: python scripts/audit_plan_code_consistency.py plans/PR-FAQ-Deflection-Report-Intent-Rule-Input.md
  - Result: passed.
- Command: git diff --check
  - Result: passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Backend contract/threading | 145 |
| Backend tests | 110 |
| UI control + test | 117 |
| Plan doc | 133 |
| **Total** | **509** |
