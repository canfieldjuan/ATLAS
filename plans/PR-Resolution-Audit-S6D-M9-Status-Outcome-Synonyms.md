# PR-Resolution-Audit-S6D-M9-Status-Outcome-Synonyms

## Why this slice exists

Issue #2050 and resolution-audit finding M9 prove that status ingestion already
owns five canonical lifecycle buckets, but compound exports such as a canonical
leading status followed by a macro name normalize into one unknown key. Rows
remain included, yet resolved/outcome summaries undercount. This independent
production-hardening micro-slice closes that safe-direction diagnostic gap. It
also folds in the mandatory plan archival for merged PR #2076, as AGENTS.md
section 1g explicitly permits on the next branch.

### Problem-derived contract

- Root cause: `_normalize_status_state()` uses a punctuation-erasing full key
  without modeling whether separator tokens are a supported exact lifecycle
  phrase or an arbitrary compound. It can therefore rescue unknown leaders
  (`Re: solved`) while the FAQ/report direct-raw caller inherits the same
  misclassification into outcome diagnostics and reopened-risk counts.
- Correct fix must touch/change: reject a bounded separator at position zero;
  preserve legacy full-key exact matching without a bounded compound separator;
  and for split input admit the full key only when its ordered word sequence
  matches the canonical exact-compound reference table, otherwise try only a
  recognized leading segment. Keep `reopened` precedence and prove package plus
  direct FAQ/report entrypoints across every canonical bucket, supported separators, exact
  punctuated synonyms, and negative near-match/prefix/ordering cases.
- Must not change: the bucket set (`resolved`, `open`, `reopened`, `cancelled`,
  `other`), raw `ticket_status`, row admission, evidence tiers, CSAT, report
  fields, downstream outcome semantics, or customer-facing product shape.

## Scope (this PR)

Ownership lane: resolution-audit-csv
Slice phase: Production hardening
Max files: 6

1. Admit a bounded leading status segment from compound macro-labelled values
   into existing lifecycle buckets only.
2. Add grammar-derived package and direct FAQ/report entrypoint proof for both
   mapping directions and archive merged S6C plan state without touching any
   other plan.

### Review Contract

- Acceptance: legacy exact statuses and canonical-sequence punctuation-normalized
  compounds retain behavior; a recognized leading status plus an explicit
  separator maps to its existing bucket; compact/spaced prefixes, unknown
  leaders, concatenated near-matches, and negations stay `other`; `reopened`
  remains distinct and wins when leading.
- Affected surfaces/risks: support-ticket row normalization, package metadata,
  and direct FAQ/report `outcome_diagnostics` status/reopened counts; false
  positives overstate resolved/reopened outcomes, while false negatives retain
  the existing safe undercount.
- Reviewer rules triggered: R1, R2, R10, R13, R14 with a classifier boundary
  probe. Reachability uses `build_support_ticket_input_package` and
  `build_ticket_faq_markdown`, observing emitted states, package metadata, and
  direct-raw report `outcome_diagnostics`.

### Files touched

- `extracted_content_pipeline/support_ticket_input_package.py`
- `plans/INDEX.md`
- `plans/PR-Resolution-Audit-S6D-M9-Status-Outcome-Synonyms.md`
- `plans/archive/PR-Resolution-Audit-S6C-Scalar-History.md`
- `tests/test_content_ops_deflection_report.py`
- `tests/test_extracted_support_ticket_input_package.py`

## Mechanism

The normalizer first rejects `:`, `-`, `|`, `/`, or `>` at position zero, with
or without following whitespace. Values without the bounded compound grammar
retain existing full-key normalization. For split values, the full key is
allowed only when the ordered words match the reference table of existing
multi-word lifecycle aliases; otherwise only the leading segment is considered.
The suffix never participates in this fallback. No substring search occurs, so
`closedeal`, `unresolved`, `Re: solved`, and `Customer Escalation: resolved`
cannot inherit a lifecycle state. Tests generate every reference-table sequence
across supported separators, then exercise direct-report and prefix boundaries.

## Intentional

- Unknown and malformed compound vocabulary still maps to `other`; this slice
  prefers safe undercount over guessing from a status word anywhere in text.
- Separators are bounded and status must lead. Whitespace-only concatenation,
  arbitrary prefixes, suffix-token search, and fuzzy/LLM classification are
  intentionally rejected.
- The exact-compound reference table models existing ordered lifecycle aliases,
  not new status buckets or a suffix-admission list; an unknown sequence forces
  leading-only classification.
- The prior S6C plan move/index refresh is mandatory merge housekeeping, not a
  product or status-normalization behavior change.

## Deferred

- #2045 owns customer evidence tiers and broader outcome semantics. No bucket,
  report, or product-shape changes are pulled forward.

Parked hardening: none.

## Verification

- Command: focused status/column/absent-column pytest -- 3 passed, 243 deselected.
- Command: owning support-ticket input-package pytest file -- 246 passed.
- Command: focused downstream deflection-report status/CSAT/compound pytest --
  3 passed, 199 deselected.
- Command: archive-plan pytest file -- 8 passed.
- Command: extracted-content validation script -- passed.
- Commands: extracted reasoning-import, standalone-debt, and ASCII audits --
  passed with zero standalone debt.
- Changed-file maturity score remained 9 and the extracted-content maturity
  ratchet reported no new brittleness.
- Command: full extracted-pipeline check script -- 10,733 passed, 21 skipped.

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/support_ticket_input_package.py` | 54 |
| `plans/INDEX.md` | 3 |
| `plans/PR-Resolution-Audit-S6D-M9-Status-Outcome-Synonyms.md` | 124 |
| `plans/archive/PR-Resolution-Audit-S6C-Scalar-History.md` | 0 |
| `tests/test_content_ops_deflection_report.py` | 32 |
| `tests/test_extracted_support_ticket_input_package.py` | 50 |
| **Total** | **263** |
