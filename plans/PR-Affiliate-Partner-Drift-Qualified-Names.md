# PR-Affiliate-Partner-Drift-Qualified-Names

## Why this slice exists

Follow-up to PR #668 (the affiliate-partner drift audit). That PR was merged
while a final review-driven fix was still being pushed, so the squash-merge
landed the P1 + P2 parser fixes (multi-row VALUES, surfaced parse errors,
`UPDATE`/`DELETE` mutation detection) but **not** the last round: matching
schema-qualified and quoted table names.

As merged, both the INSERT parser and the mutation detector match only the
bare `affiliate_partners`. A migration using valid SQL like
`INSERT INTO public.affiliate_partners (...)`, `UPDATE "affiliate_partners"`,
or `DELETE FROM public."affiliate_partners"` would be silently missed -- the
audit would report `pass` while a seed or mutation went unseen. That is the
exact gap the reviewers (Codex P1 + P2 on #668) flagged. Every migration in
the tree today is unqualified, so the merged audit is correct now; this slice
closes the gap for future migrations.

## Scope (this PR)

1. Add a shared `_AFF_TABLE` regex fragment that matches the
   `affiliate_partners` table with an optional schema qualifier and optional
   double-quoted identifiers, guarded so a longer name like
   `affiliate_partners_history` is not mistaken for it.
2. Use `_AFF_TABLE` in all three matchers: the `INSERT INTO` parser and the
   `UPDATE` / `DELETE FROM` mutation detectors.
3. Add tests for schema-qualified and quoted INSERTs, qualified and quoted
   mutations, and the prefixed-name (`affiliate_partners_history`) negative
   case.

### Files touched

- `scripts/audit_affiliate_partner_drift.py`
- `tests/test_affiliate_partner_drift.py`
- `plans/PR-Affiliate-Partner-Drift-Qualified-Names.md`

## Mechanism

`_AFF_TABLE` is `(?:(?:"[^"]+"|<ident>)\s*\.\s*)?(?:"affiliate_partners"|
affiliate_partners(?![\w$]))`: an optional schema part (quoted or bare
identifier followed by a dot) and the table itself, matched either as an exact
quoted identifier or unquoted with a negative lookahead so it cannot swallow a
longer name. The three matchers concatenate it after their respective
keywords, so qualified/quoted forms are recognized while the existing
unqualified migrations continue to match unchanged. The INSERT matcher still
ends in `\s*\(`, preserving the `m.end() - 1` handoff to the tuple reader.

## Intentional

- **Regex broadening, not a SQL-library swap.** The seed/mutation surface is
  small and now covers the realistic forms (qualified, quoted, multi-row, both
  seed layouts). A dependency-free matcher keeps the audit self-contained and
  fully fixture-tested.
- **Case-insensitive table match.** Quoted identifiers are technically
  case-sensitive in Postgres, but every real table name here is lowercase;
  matching `"AFFILIATE_PARTNERS"` too is harmless over-coverage.

## Deferred

- Fully parsing and *applying* `UPDATE`/`DELETE` effects (vs. detecting and
  warning) remains the deferred follow-up noted in #668, only relevant once a
  mutation migration is actually written.

## Verification

- `python -m pytest tests/test_affiliate_partner_drift.py -q` -> `14 passed in
  0.07s` (adds schema-qualified + quoted INSERTs, qualified + quoted
  mutations, and the `affiliate_partners_history` negative case to the #668
  suite).
- `scripts/audit_affiliate_partner_drift.py` run against live `:5433/atlas`
  -> `summary {seeded_partners: 6, live_partners: 6, pass: 5, warn: 0,
  fail: 0}`, exit `0` -- the real (unqualified) migrations still match, and
  the broadened regex does not false-trigger on the `REFERENCES
  affiliate_partners(id) ON DELETE CASCADE` clause in migration 063.
- `git diff --check` -> passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| `scripts/audit_affiliate_partner_drift.py` (`_AFF_TABLE` + 3 matchers) | ~20 |
| `tests/test_affiliate_partner_drift.py` (3 tests) | ~40 |
| Plan doc | ~95 |
| **Total** | **~155** |
