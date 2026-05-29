# PR-FAQ-Config-Controls-Catalog-Gate

## Why this slice exists

PR-FAQ-Deflection-Report-Intent-Rule-Input added the hosted intent-rule control.
The review marked one remaining follow-up: the UI shows FAQ configuration
controls based on selected outputs even if the backend catalog does not
advertise the matching input contracts. That can produce version-skew controls
that write inputs the current backend did not declare.

This slice keeps the fix small: make the FAQ configuration panel and fields
catalog-gated while preserving the existing controls when the catalog advertises
them.

## Scope (this PR)

Ownership lane: content-ops/deflection-report-ui

Slice phase: Product polish

1. Add a small domain helper for FAQ configuration-control visibility.
2. Gate the FAQ configuration panel on both selected FAQ output and at least one
   advertised FAQ input contract.
3. Gate each FAQ textarea on its own advertised contract.
4. Add focused UI helper/source tests for the version-skew case.

### Files touched

| File | Purpose |
|---|---|
| `atlas-intel-ui/src/domain/contentOps/faqConfigurationInputs.ts` | Adds reusable FAQ input keys and catalog-gated visibility helper. |
| `atlas-intel-ui/src/domain/contentOps/index.ts` | Exports the new helper/constants. |
| `atlas-intel-ui/src/pages/ContentOpsNewRun.tsx` | Uses catalog-gated visibility for FAQ config controls. |
| `atlas-intel-ui/scripts/content-ops-faq-configuration-inputs.test.mjs` | Verifies the version-skew visibility behavior and screen wiring. |
| `plans/PR-FAQ-Config-Controls-Catalog-Gate.md` | Documents the slice contract and verification. |

## Mechanism

The screen now asks the domain helper whether FAQ configuration controls are
visible using the selected outputs plus the actual catalog contracts:

```ts
faqConfigurationControlsVisible(outputs, {
  intentRules: faqIntentRulesContract,
  documentationTerms: faqDocumentationTermsContract,
  vocabularyGapRules: faqVocabularyGapRulesContract,
})
```

The panel renders only when the helper returns true, and each textarea is wrapped
by its corresponding contract check. If an older backend advertises only
documentation terms and vocabulary-gap rules, the intent-rule field stays hidden
instead of writing an undeclared input.

## Intentional

- No backend changes are included. This is a UI/catalog alignment fix.
- The existing placeholder fallbacks remain for incomplete contract metadata,
  but a missing contract no longer causes the field to render.

## Deferred

- Parked hardening considered: `atlas-intel-ui npm audit vulnerabilities`
  remains parked in `HARDENING.md`; dependency upgrade work is outside this UI
  catalog-gating slice.

## Verification

- Command: npm run test:content-ops-faq-configuration-inputs
  - Result: 4 passed.
- Command: npm run lint
  - Result: passed.
- Command: npm run build
  - Result: passed.
- Command: python scripts/audit_plan_doc.py plans/PR-FAQ-Config-Controls-Catalog-Gate.md
  - Result: passed.
- Command: python scripts/audit_plan_code_consistency.py plans/PR-FAQ-Config-Controls-Catalog-Gate.md
  - Result: passed.
- Command: git diff --check
  - Result: passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Domain helper + exports | 28 |
| Screen wiring | 119 |
| Test updates | 32 |
| Plan doc | 90 |
| **Total** | **269** |
