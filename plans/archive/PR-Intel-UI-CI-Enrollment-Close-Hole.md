# PR-Intel-UI-CI-Enrollment-Close-Hole

## Why this slice exists

`atlas-intel-ui/package.json` declares 24 `test:*` suites; the CI workflow
`.github/workflows/atlas_intel_ui_checks.yml` ran only 15 of them. The other 9 existed but never
executed in CI — no aggregate `test`/`test:all` script, no other workflow picking
them up — so "green" was partially meaningless on the frontend. This closes that
hole by enrolling the 9 missing suites. Tracked in issue #1318 (which also
documents the durable-audit follow-up that prevents recurrence; see Deferred).

## Scope (this PR)

Ownership lane: intel-ui/ci-enrollment
Slice phase: Workflow/process

1. Enroll the 9 previously-unrun `test:*` suites as explicit
   `run: npm run test:<name>` steps in `.github/workflows/atlas_intel_ui_checks.yml`, placed with the
   other test steps (after `review-source-selection`, before `Build`).
2. Repair one stale assertion in
   `content-ops-faq-source-selection.test.mjs` so the suite passes against current
   source (the suite was failing only because it had silently rotted while outside
   CI — see Mechanism).

### Files touched

- `.github/workflows/atlas_intel_ui_checks.yml`
- `atlas-intel-ui/scripts/content-ops-faq-source-selection.test.mjs`
- `plans/PR-Intel-UI-CI-Enrollment-Close-Hole.md`

## Mechanism

Each suite runs via Node's built-in runner (`node --test scripts/<name>.test.mjs`),
so enrollment is a one-line `run:` step per suite, mirroring the existing 15.

Before enrolling, every one of the 9 was run locally (`npm install` then
`npm run test:<name>`). Eight passed unchanged. One —
`content-ops-faq-source-selection` — failed a single source-grep assertion:

```
assert.ok(newRunSource.includes("const SOURCE_FAQ_IDS_INPUT = 'source_faq_ids'"))
```

The constant was extracted into `src/pages/contentOpsSourceMode.ts` and is now
imported by `ContentOpsNewRun.tsx` (`from './contentOpsSourceMode'`); the feature
is intact (the other 3 subtests, which assert behavior, pass). Because the suite
was not in CI, the behavior-preserving refactor that moved the constant never
tripped it and the assertion silently rotted — exactly the failure class #1318 is
about. The fix asserts the declaration in its new home plus the import in the page,
preserving the test's original intent.

## Intentional

- The stale assertion is **repaired, not deleted.** The feature works and the fix
  is one line; quarantining the suite would drop real coverage of the FAQ
  source-selection flow.
- The fix translates the assertion to the refactored code shape (declaration in
  `contentOpsSourceMode.ts`, imported by the page) rather than loosening it to a
  bare symbol-presence check, so it still proves the page is wired to the
  `source_faq_ids` input key.
- Steps are added as individual named `run:` lines (not collapsed into one
  multi-test invocation) to match the existing workflow style and keep per-suite
  failures legible in the CI log.
- No production/source behavior changes — workflow + one test file only.

## Deferred

- **The durable enrollment audit** — a `scripts/audit_*`-style check that fails
  when a `test:*` script in `atlas-intel-ui/package.json` is not run by
  `.github/workflows/atlas_intel_ui_checks.yml`, with §3h fixtures, wired into `local_pr_review.sh`.
  This is PR-B in #1318; without it, manual drift can recur. This slice deliberately
  closes the current hole first so CI is honest immediately, then PR-B makes it
  un-droppable.
- **Generalize to other UI workflows** (`portfolio-ui`, `atlas-ui`,
  `atlas-churn-ui`, `atlas-admin-ui`, `atlas-mobile`) and workflow `paths:` filter
  drift — the "hunt for siblings" in #1318. Follow-up.
- Parked hardening: none.

## Verification

```bash
cd atlas-intel-ui && npm install --no-audit --no-fund
# all 9 newly-enrolled suites pass:
for t in content-ops-cache-policy-ui content-ops-faq-macro-publish \
  content-ops-faq-source-selection content-ops-ingestion-limits \
  content-ops-input-display content-ops-usage-budget-ui \
  content-ops-usage-summary content-ops-zendesk-credentials sitemap-bridge; do
  npm run test:$t >/dev/null 2>&1 && echo "PASS $t" || echo "FAIL $t"; done
```

Result: all 9 PASS (`content-ops-faq-source-selection` after the assertion repair;
4/4 subtests). The 8 others passed unchanged. Plus `bash scripts/local_pr_review.sh`
for plan-shape / drift / whitespace gates.

## Estimated diff size

| File | LOC |
|---|---:|
| `.github/workflows/atlas_intel_ui_checks.yml` | 27 |
| `atlas-intel-ui/scripts/content-ops-faq-source-selection.test.mjs` | 10 |
| `plans/PR-Intel-UI-CI-Enrollment-Close-Hole.md` | 97 |
| **Total** | **134** |
