# PR-Blog-D6-Quote-Source-Attribution

Ownership lane: `content-ops/blog-d6-quote-source`

## Why this slice exists

Defect **D6** (`reports/blog-audit-findings.md`): a quote's prose lead-in names a
different platform than the blockquote attribution and the real source. In
`zoho-crm-deep-dive`, L166 reads "One **verified** reviewer on **G2** describes
the core value:" but the blockquote (L168) attributes the quote to **Slashdot**,
and the DB confirms the quote's `source` is **slashdot** (1 occurrence). The
prose "G2" (and "verified", since Slashdot is a community source) is wrong.

Data-untruthful output, so the published prose is fixed inline. The prose lead-in
is LLM free-text (the quote already carries `source_name`, used correctly in the
blockquote), so deterministic generator-prevention is fuzzy and is parked.

## Scope (this PR)

Data-only thin slice: correct the one published mismatch.

- **Data** (`zoho-crm-deep-dive-2026-04.ts` L166): "One verified reviewer on G2
  describes the core value:" -> "One reviewer on Slashdot describes the core
  value:" -- matches the blockquote and the DB.

### Files touched

- `plans/PR-Blog-D6-Quote-Source-Attribution.md`
- `ATLAS-HARDENING.md`
- `atlas-churn-ui/src/content/blog/zoho-crm-deep-dive-2026-04.ts`

## Mechanism

One contextual phrase replacement (asserted count = 1). No generator change: the
quote carries `source_name` (the blockquote uses it correctly); only the LLM's
free-text lead-in drifted, which can't be fixed deterministically in scope.

A corpus-wide grep confirmed D6's blast radius before editing: "verified reviewer
on <platform>" is widespread but almost all are blockquote ATTRIBUTIONS (the
quote's real source, correct). Only three posts have the prose-lead-in shape:
zoho (this mismatch), top-complaint-every-helpdesk (lead-in "Gartner" matches its
blockquote -- no defect), and salesforce (a different defect -- orphaned lead-in,
no quote follows -- parked, not D6).

## Intentional

- **Data-truthful fix, no generator change.** The source path is correct
  (blockquote uses `source_name`); the defect is LLM lead-in drift, parked.
- **Match the blockquote, drop "verified".** Slashdot is a community source, so
  "verified reviewer" was doubly wrong.

## Deferred

**Parked hardening** in `ATLAS-HARDENING.md` (separate from root `HARDENING.md`,
per the maintainer; root carries a pointer):

- Deep-dive quote lead-ins can name a platform the quote isn't from
  (generator/LLM drift; prompt-guidance fix, uncertain). Effort M, correctness.
- Orphaned quote lead-in in `salesforce-deep-dive` L159 ("...verified reviewer on
  G2 notes ...:" with no quote following) -- a separate orphaned-intro defect,
  pre-existing, surfaced while scoping D6. Effort S (data) / M (generator guard),
  correctness.
- zoho L158 source list omits Slashdot (a quoted source) -- added on the #802
  review. DB-verified the naive "add Slashdot to the 237 community count"
  doesn't reconcile: the Slashdot quote's review is `not_applicable` (non-enriched,
  in-window, Zoho-mention), so it's outside the enriched 261/237 counts. Needs a
  Phase-2 quote-pool-vs-corpus-count reconciliation. M, correctness.

Also remaining: D5 (source-list incompleteness), D2-followup (Zoho/Zoho CRM
registry merge), D3-followup (frequency-view chart).

## Verification

- Re-grep: "verified reviewer on G2" in the zoho post -> 0; L166 lead-in now
  reads "One reviewer on Slashdot describes the core value:", matching the L168
  blockquote ("reviewer on Slashdot") and the DB (`source = slashdot`, 1 row).
- Corpus-wide grep established the fix is a single instance (others are
  attributions or a different defect class).
- `audit-published-posts.js` -> orphaned-quote-reference / form-prompt /
  empty-blockquote all 0. No new defects.
- (No generator change, so no unit test -- data-only slice demonstrated by grep +
  audit, per AGENTS.md 3d.)

## Estimated diff size

| Area | LOC |
|---|---:|
| zoho data (one lead-in) | ~2 |
| ATLAS-HARDENING.md (2 parked entries) | ~16 |
| Plan doc | ~70 |
| **Total** | **~115** |
