# PR-FAQ-Deflection-Report-UI-Controls

## Why this slice exists

PR-FAQ-Deflection-Report-UI-Readonly made `faq_deflection_report` visible in the Intel UI, but its configuration controls still only appear for the older `faq_markdown` output. The backend generation plan uses the same `faq_documentation_terms` and `faq_vocabulary_gap_rules` inputs for both `faq_markdown` and `faq_deflection_report`, so a user running the $1,500 deflection report still has to hand-edit raw JSON for the vocabulary-gap part of the real flow.

This slice keeps the first-class UI controls narrow: expose the existing FAQ vocabulary-gap controls when either FAQ output is selected, and lock that selection rule with a focused UI test. The slice intentionally does not add rule-file, custom intent-rule, or persistence controls.

## Scope (this PR)

Ownership lane: content-ops/deflection-report-ui

Slice phase: Product polish

1. Add a small domain helper that treats `faq_markdown` and `faq_deflection_report` as FAQ-configuration outputs.
2. Use that helper in `ContentOpsNewRun` so the existing documentation-terms and vocabulary-gap-rules fields render for `faq_deflection_report` runs.
3. Add focused test coverage proving the controls are tied to both FAQ outputs and not unrelated outputs.

### Files touched

| File | Purpose |
|---|---|
| `atlas-intel-ui/src/domain/contentOps/faqConfigurationInputs.ts` | Adds the FAQ output selection helper and constants. |
| `atlas-intel-ui/src/domain/contentOps/index.ts` | Exports the helper/constants for the screen and focused test. |
| `atlas-intel-ui/src/pages/ContentOpsNewRun.tsx` | Uses the helper to show existing FAQ vocabulary-gap controls for deflection reports. |
| `atlas-intel-ui/scripts/content-ops-faq-configuration-inputs.test.mjs` | Verifies the selection helper and screen wiring. |
| `atlas-intel-ui/package.json` | Adds the focused npm test script. |
| `.github/workflows/atlas_intel_ui_checks.yml` | Enrolls the focused test in the UI CI lane. |
| `plans/PR-FAQ-Deflection-Report-UI-Controls.md` | Documents this slice contract. |

## Mechanism

The backend already maps both FAQ outputs through `_faq_markdown_config_for_request`, which reads these input keys:

```ts
faq_documentation_terms
faq_vocabulary_gap_rules
```

This PR moves the UI's FAQ-output check from a single `faq_markdown` literal to a tested helper:

```ts
faqConfigurationInputsSelected(outputs)
```

The existing textareas, JSON update helpers, and catalog contract display stay unchanged. Only the visibility and contract lookup predicate changes from `faq_markdown` selected to any FAQ-configuration output selected.

## Intentional

- No custom intent-rule control is added because the hosted execute path does not currently expose an input contract for it.
- No rule-file upload is added; that remains a CLI/report-operations surface until a dedicated UI slice defines the upload contract.
- No new persistence or saved defaults are added. The controls continue writing directly into the existing inputs JSON.

## Deferred

- Parked hardening considered: `atlas-intel-ui npm audit vulnerabilities` remains parked in `HARDENING.md`; dependency upgrade work is outside this UI controls slice.
- Future product-polish slice: add first-class custom intent-rule controls after the execute/catalog contract exposes the input.
- Future product-polish slice: add rule-file upload or saved configuration presets if the report operations flow needs it.

## Verification

- Command: npm ci
  - Result: passed; reported the same 6 existing audit findings already parked in `HARDENING.md`.
- Command: npm run test:content-ops-faq-configuration-inputs
  - Result: 2 passed.
- Command: npm run test:content-ops-deflection-report-ui
  - Result: 3 passed.
- Command: npm run lint
  - Result: passed.
- Command: npm run build
  - Result: passed.
- Command: python scripts/audit_plan_doc.py plans/PR-FAQ-Deflection-Report-UI-Controls.md
  - Result: passed.
- Command: python scripts/audit_plan_code_consistency.py plans/PR-FAQ-Deflection-Report-UI-Controls.md
  - Result: passed.
- Command: git diff --check
  - Result: passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Domain helper + exports | 24 |
| Screen wiring | 12 |
| Test + CI enrollment | 62 |
| Plan doc | 87 |
| **Total** | **185** |
