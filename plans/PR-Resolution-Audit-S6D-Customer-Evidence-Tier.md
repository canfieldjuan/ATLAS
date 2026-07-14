# PR-Resolution-Audit-S6D-Customer-Evidence-Tier

## Why this slice exists

Issue #2045 extracts customer-evidence semantics from paused #2037. Status
normalization was closed separately in merged #2078; this production-hardening
slice handles only the remaining admitted-customer-wording decision.

### Problem-derived contract

- Root cause: the package reduces customer evidence to a truthiness flag instead
  of preserving an admitted wording value. `_TEXT_KEYS` overlaps title aliases
  (`summary`), compact text falls back to all-quoted bodies that the line-hygiene
  boundary excludes, and examples then use full normalized title-plus-body text.
  The first raw alias can also be stripped and mask a later valid body alias.
  The downstream FAQ item chooses the first row tier, so a later admitted row in
  a mixed cluster is hidden by row order. These paths do not share one
  hygiene-aware semantic decision.
- Correct fix must touch/change: one helper at the normalized ticket seam must
  produce admitted wording from the first line-hygiene-preserved body/comment
  alias distinct from title aliases, or from a question-like sanitized title
  only when no alias survives. Row/package tiers and examples must consume that
  stored value. The grouped FAQ item must select the strongest present row
  evidence tier. Package and direct FAQ/report tests must prove stripped-only,
  ordinary-title-only (including `summary`), quoted-only, admitted-body,
  question-like-subject-only, a stripped-first/later-valid body alias, exact
  example wording, and mixed-cluster order independence.
- Must not change: private/internal admission, HTML/parser/sanitizer grammar,
  scalar-history state, status buckets/outcome diagnostics (#2078), evidence
  tier names, report/snapshot/email/PDF/landing/checkout shape, public schemas,
  dependencies, and #2037 implementation.

## Scope (this PR)

Ownership lane: resolution-audit-csv
Slice phase: Production hardening
Max files: 8

1. Make `csv_customer_text` and `customer_wording_examples` consume one
   hygiene-aware admitted-customer-wording value.
2. Make the direct FAQ/report item choose the strongest evidence tier across its
   grouped rows, independent of source-row order.
3. Prove both decisions through package metadata/rows and direct FAQ/report
   items without adding fields or changing report shape.

### Review Contract

- Acceptance: stripped-only, quoted-only, ordinary title-only, and
  `summary`-only rows remain index metadata and create no wording example;
  sanitized body or public comment text remains customer evidence and is the
  exact example text; a valid later body alias is admitted when an earlier alias
  is stripped; a question-like title with no admitted body/comment is customer
  evidence and its wording example; a grouped FAQ item observes the strongest
  admitted row tier regardless of row order.
- Affected surfaces/risks: package metadata, per-row evidence tier, wording
  examples, and downstream FAQ/report item evidence tier. False admission
  misrepresents index metadata as customer wording; false rejection hides a
  genuine subject-only question.
- Reviewer rules triggered: R1, R2, R10, R13, R14. Reachability uses
  `build_support_ticket_input_package` and `build_ticket_faq_markdown`,
  observing emitted examples and evidence-tier values.

### Files touched

- `extracted_content_pipeline/support_ticket_input_package.py`
- `extracted_content_pipeline/ticket_faq_markdown.py`
- `plans/INDEX.md`
- `plans/PR-Resolution-Audit-S6D-Customer-Evidence-Tier.md`
- `plans/archive/PR-Resolution-Audit-S6D-M9-Status-Outcome-Synonyms.md`
- `tests/test_content_ops_deflection_report.py`
- `tests/test_extracted_support_ticket_input_package.py`
- `tests/test_smoke_content_ops_support_ticket_package.py`

## Mechanism

The normalizer stores one internal admitted-wording value after the existing
line-hygiene boundary. Body aliases that also mean title (`summary`) are
excluded; the first line-hygiene-preserved body/comment alias wins; otherwise
the existing question-like-title recognizer may supply the title. Empty/ordinary
titles and quoted-only text do not qualify. The same value drives row/package
tiers and wording examples. The FAQ reducer ranks the existing tier values instead of
returning the first row's tier, so the output preserves the strongest admitted
evidence in each group.

## Intentional

- The existing question-like title grammar is reused; this slice does not widen
  its vocabulary or infer questions from arbitrary index titles.
- The helper is internal only. No evidence-tier label, report field, or product
  claim changes; only a grouped item's existing tier value now reflects all of
  its rows instead of their incidental order.
- The merged #2078 plan is archived as required housekeeping on this next
  branch; it is unrelated to the runtime decision.

## Deferred

- #2045's status/outcome synonym work is already merged in #2078. Any broader
  evidence-tier semantics outside admitted customer wording remain in #2045 for
  a separately scoped slice.

Parked hardening: none.

## Verification

- Pre-fix: package and direct-report regressions each failed under the prior
  raw-field presence rule.
- Focused regressions: `pytest -q
  tests/test_extracted_support_ticket_input_package.py -k
  customer_wording_uses_admitted_text` (1 passed) and `pytest -q
  tests/test_content_ops_deflection_report.py -k
  direct_subject_only_question_uses_customer_evidence_tier` (1 passed).
- Owning suites: input-package tests (247 passed), direct deflection-report
  tests (199 passed, 4 skipped), and the support-ticket package smoke suite
  (56 passed).
- Extracted-package gauntlet: validation, standalone/dependency, ASCII, and
  maturity-ratchet audits passed; the full extracted-pipeline CI-equivalent
  check completed with 10,735 passed and 21 skipped.

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/support_ticket_input_package.py` | 44 |
| `extracted_content_pipeline/ticket_faq_markdown.py` | 20 |
| `plans/INDEX.md` | 3 |
| `plans/PR-Resolution-Audit-S6D-Customer-Evidence-Tier.md` | 131 |
| `plans/archive/PR-Resolution-Audit-S6D-M9-Status-Outcome-Synonyms.md` | 0 |
| `tests/test_content_ops_deflection_report.py` | 15 |
| `tests/test_extracted_support_ticket_input_package.py` | 68 |
| `tests/test_smoke_content_ops_support_ticket_package.py` | 14 |
| **Total** | **295** |
