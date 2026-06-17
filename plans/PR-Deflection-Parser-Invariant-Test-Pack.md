# PR-Deflection-Parser-Invariant-Test-Pack

## Why this slice exists

#1471 tracks the deterministic half of the parser/admission hardening lane:
recent deflection fixes repeatedly caught matching and projection boundary
bugs after review rather than before push (#1439, #1446, #1453, #1466). The
root cause is that mutable recognition sets and fail-closed snapshot contracts
have only example-level coverage, so a future term/fold/field addition can
break the class while the cited happy path stays green. This slice fixes the
root for those deterministic classes by turning them into invariants.

## Scope (this PR)

Ownership lane: content-ops/deflection-parser-testing
Slice phase: Robust testing

1. Add an inflection-coverage invariant for FAQ resolution action terms.
2. Add a fold-target invariant for support-ticket token and phrase folds.
3. Generalize snapshot summary field-deletion checks across both portfolio-ui
   snapshot projectors.
4. Add the smallest upstream normalizer support needed for irregular
   resolution-action past tense (`chose`, `ran`, `reran`, `sent`).

### Review Contract

Acceptance criteria:
- Every `_RESOLUTION_ACTION_TERMS` entry recognizes generated standard
  inflections through the real `_resolution_signal_token` path.
- Every non-empty `_TOKEN_FOLDS` target and `_PHRASE_FOLDS` replacement token
  survives `support_ticket_tokens(...)`.
- Dropping any required snapshot summary field invalidates `projectSnapshot`
  and the React fallback guard instead of defaulting to `0`/empty values.
- Existing CI enrollment remains intact: touched Python tests are already in
  `run_extracted_pipeline_checks.sh`, and touched portfolio-ui scripts are
  already in `.github/workflows/portfolio_ui_checks.yml`.

Affected surfaces:
- Extracted content-pipeline parser/FAQ tests.
- portfolio-ui result-page/proxy contract tests as legacy/Vite maintenance
  because #1471 names those projectors. This is not buyer-page coverage for
  the atlas-portfolio Next.js result route.

Risk areas:
- Over-generating non-standard inflections and creating brittle tests.
- Source-only React fallback checks are weaker than executing TSX directly;
  this PR keeps the current local test style but sweeps every required field.

Reviewer rules triggered: R1, R2, R8, R10, R12, R13, R14.

### Files touched

- `extracted_content_pipeline/ticket_faq_markdown.py`
- `plans/PR-Deflection-Parser-Invariant-Test-Pack.md`
- `portfolio-ui/scripts/faq-deflection-atlas-proxy.test.mjs`
- `portfolio-ui/scripts/faq-deflection-result-page.test.mjs`
- `tests/test_extracted_content_deflection_submit.py`
- `tests/test_extracted_support_ticket_input_package.py`
- `tests/test_extracted_ticket_faq_markdown.py`

## Mechanism

- Generate a bounded set of standard action-term inflections in the test file
  and assert each maps back to its canonical action term through the real
  token normalizer.
- Import the clustering fold tables in the support-ticket input-package test
  and assert each emitted fold target remains visible after normal
  tokenization.
- Replace single-field snapshot omission checks with required-field sweeps for
  `projectSnapshot` and the React fallback guard.

## Intentional

- Production behavior changes are limited to the upstream
  `_resolution_signal_token` irregular past-tense map needed by the invariant;
  no parser/admission envelope behavior changes.
- The low non-zero CSV admission threshold from #1467 remains deferred until a
  real partial provider CSV justifies warn-vs-reject product policy.
- The React fallback test continues to inspect the TSX source because the
  existing portfolio-ui test harness is source-based for that component.
- The portfolio-ui snapshot field sweep is legacy/Vite projector maintenance,
  not a claim that this PR tests the buyer-facing atlas-portfolio Next.js
  results page.

## Deferred

- #1467 parser-breakage evidence runner: generate the adversarial admission
  matrix and score fail-closed vs fail-open mechanics.
- #1467 low non-zero usable-ratio reject threshold: blocked on real partial
  provider CSV evidence, not synthetic ratios.

Parked hardening: none.

## Verification

- `pytest tests/test_extracted_ticket_faq_markdown.py tests/test_extracted_support_ticket_input_package.py -q` -- 464 passed.
- `pytest tests/test_extracted_content_deflection_submit.py::test_deflection_submit_accepts_zendesk_full_thread_blob -q` -- 1 passed.
- `npm --prefix portfolio-ui run test:deflection-atlas-proxy` -- passed.
- `npm --prefix portfolio-ui run test:deflection-result` -- passed.
- `./scripts/run_extracted_pipeline_checks.sh` -- 4545 passed, 10 skipped.

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/ticket_faq_markdown.py` | 9 |
| `plans/PR-Deflection-Parser-Invariant-Test-Pack.md` | 113 |
| `portfolio-ui/scripts/faq-deflection-atlas-proxy.test.mjs` | 72 |
| `portfolio-ui/scripts/faq-deflection-result-page.test.mjs` | 57 |
| `tests/test_extracted_content_deflection_submit.py` | 13 |
| `tests/test_extracted_support_ticket_input_package.py` | 12 |
| `tests/test_extracted_ticket_faq_markdown.py` | 70 |
| **Total** | **346** |
