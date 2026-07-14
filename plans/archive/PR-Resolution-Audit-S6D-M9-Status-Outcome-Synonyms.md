# PR-Resolution-Audit-S6D-M9-Status-Outcome-Synonyms

## Why this slice exists

Issue #2050 and resolution-audit finding M9 prove that status ingestion already
owns five canonical lifecycle buckets, but compound exports such as a canonical
leading status followed by a macro name normalize into one unknown key. Rows
remain included, yet resolved/outcome summaries undercount. This independent
production-hardening micro-slice closes that safe-direction diagnostic gap. It
also folds in the mandatory plan archival for merged PR #2076, as AGENTS.md
section 1g explicitly permits on the next branch.

The punctuation-only suffix closure may exceed the 400-LOC target because the
small runtime predicate is indivisible from its grammar-generated package proof,
direct-report reachability regression, and this corrective contract.

### Problem-derived contract

- Root cause: `_normalize_status_state()` uses a punctuation-erasing full key
  without one lexical admission decision. Its earlier split-only table gate
  leaves whitespace-only, compact, and one-sided separator forms on the broad
  whole-value path, where unknown leaders (`Re solved`, `Re/Solved`, and
  `Can- celled`) reconstruct lifecycle keys. The FAQ/report direct-raw caller
  inherits that misclassification into outcome diagnostics and reopened-risk
  counts.
- Correct fix must touch/change: reject a separator at position zero; model the
  raw value as exactly one of a recognized unseparated alias, a canonical
  ordered multi-word lifecycle phrase, or a recognized leading phrase followed
  by the bounded macro-suffix grammar. Every other punctuation/whitespace word
  boundary must return `other`. Exact phrases may have only internal allowed
  delimiters (no wrapper punctuation); macro admission must scan candidates for
  a suffix with semantic content after a recognized leading phrase, so a
  punctuated compound leader remains recognized. Preserve underscores only as internal delimiters
  of a recognized canonical phrase. The test oracle must define semantic status
  aliases separately from runtime tables and generate every split of each
  one-word alias across the malformed-boundary grammar. Preserve the explicit
  legacy spelling `Re-opened`, keep `reopened` precedence, and prove package
  plus direct FAQ/report entrypoints across every canonical bucket, grammar
  separator, exact punctuated synonym, and negative near-match/prefix/ordering
  case.
- Must not change: the bucket set (`resolved`, `open`, `reopened`, `cancelled`,
  `other`), raw `ticket_status`, row admission, evidence tiers, CSAT, report
  fields, downstream outcome semantics, or customer-facing product shape.

### Corrective contract: punctuation-only macro suffixes (2026-07-13)

- Root cause: the macro branch accepts a recognized leading status whenever its
  remainder is nonblank after trimming. `_clean()` preserves separators, so a
  suffix made only of punctuation (for example `Solved::`, `Open - -`, or
  `Reopened: |`) satisfies that incidental nonblank test even though it has no
  macro content. The parser therefore violates its own fail-closed admission
  invariant and inflates existing lifecycle diagnostics.
- Correct fix must touch/change: the single suffix-admission condition in
  `_normalize_status_state()` must require semantic suffix content -- at least
  one Unicode alphanumeric character, not merely a non-whitespace character.
  The grammar-derived package test must generate punctuation-only suffixes from
  the macro-delimiter alphabet for canonical one-word and phrase leaders; the
  existing direct FAQ/report entrypoint test must prove those rows remain
  `other` in outcome diagnostics. This is the only runtime behavior change.
- Must not change: recognized aliases, canonical phrase parsing, allowed
  internal phrase delimiters, a recognized status with an alphanumeric macro
  suffix, bucket precedence, row admission, report structure, public APIs,
  schemas, dependencies, unrelated formatting, and archived-plan housekeeping.
- Assumption: macro labels are textual metadata. A suffix containing a Unicode
  letter or digit is meaningful; separator-only or whitespace-only suffixes are
  malformed. No repository contract requires punctuation-only macro names.
- Verification: first reproduce the three reported forms through the package
  and direct FAQ/report entrypoints; then run the grammar-derived status test,
  the direct-report status test, both owning test files, extracted-package
  audits, maturity ratchet, and the full extracted CI-equivalent gate.

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

- Acceptance: exact single aliases (including explicit `Re-opened`) and
  canonical-sequence punctuation-normalized compounds retain behavior; a
  recognized leading status plus the bounded macro separator maps to its
  existing bucket; whitespace-only, one-sided, compact, and unknown-leading
  word boundaries, wrapper punctuation, and suffixes without alphanumeric
  content stay `other`;
  punctuated or snake_case multi-word leaders retain their bucket before a
  suffix containing an alphanumeric character; `reopened` remains distinct and
  wins when leading.
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

The normalizer has one lexical admission decision: a raw value is accepted only
as an exact unseparated lifecycle alias, the explicit legacy `Re-opened`
spelling, an ordered token sequence in the canonical phrase table, or a
recognized leading phrase followed by a bounded separator and suffix containing
at least one Unicode alphanumeric character.
Canonical phrase syntax permits delimiters only between its words, rejecting
wrapper punctuation. The macro parser inspects each bounded separator until a
recognized prefix is found, preserving a punctuated compound leader instead of
splitting at its first word. Every other multi-token or separator-bearing value
is `other`; no punctuation/whitespace erasure can reach the bucket table first.
Underscore is one such internal delimiter only for a recognized phrase. Tests
implement a semantic alias oracle separate from runtime tables, then generate
canonical phrases, compound leaders plus suffixes, every internal split of each
one-word alias across the malformed-boundary grammar, and
prefix/compact/one-sided/whitespace/wrapper/contentless-suffix products
through the real package entrypoint, then exercise representative direct-report
diagnostics.

## Intentional

- Unknown and malformed compound vocabulary still maps to `other`; this slice
  prefers safe undercount over guessing from a status word anywhere in text.
- Only recognized canonical phrases may use arbitrary word delimiters; the
  bounded macro grammar alone admits a free-form suffix. Whitespace-only,
  arbitrary prefixes, one-sided/compact malformed compounds, suffix-token
  search, and fuzzy/LLM classification are intentionally rejected.
- A macro delimiter without alphanumeric suffix text is malformed, not a status;
  phrase punctuation is allowed only between canonical words, never as a
  leading or trailing wrapper.
- Snake_case is retained only where it joins an ordered canonical phrase; an
  underscore cannot rescue an unknown prefix or single-word partition.
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

- Pre-fix proof: the grammar-derived package and direct-report punctuation-only
  suffix assertions each failed against the prior nonblank-suffix admission.
- Command: focused grammar-derived status pytest -- 1 passed, 245 deselected.
- Command: owning support-ticket input-package pytest file -- 246 passed.
- Command: focused direct-report compound-status pytest -- 1 passed, 201
  deselected.
- Command: owning deflection-report pytest file -- 198 passed, 4 skipped.
- Command: full extracted-pipeline check script -- 10,733 passed, 21 skipped.
- Command: extracted-content validation script -- passed.
- Commands: extracted reasoning-import, standalone-debt, and ASCII audits --
  passed with zero standalone debt.
- Changed-file maturity score remained 9 and the extracted-content maturity
  ratchet reported no new brittleness.

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/support_ticket_input_package.py` | 84 |
| `plans/INDEX.md` | 3 |
| `plans/PR-Resolution-Audit-S6D-M9-Status-Outcome-Synonyms.md` | 185 |
| `plans/archive/PR-Resolution-Audit-S6C-Scalar-History.md` | 0 |
| `tests/test_content_ops_deflection_report.py` | 46 |
| `tests/test_extracted_support_ticket_input_package.py` | 112 |
| **Total** | **430** |
