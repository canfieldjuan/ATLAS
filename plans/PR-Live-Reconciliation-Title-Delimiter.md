# PR-Live-Reconciliation-Title-Delimiter

## Why this slice exists

The user asked to repair the red checks on the owned H-18 receipt PR #2481.
Its two new P1 findings were fixed, published, and resolved, but the trusted
`live-reconciliation` workflow still reports them as missing from the PR-body
ledger. This is a real CI defect in the base-branch provider, not a missing
disposition: the current connector supplies a title on the first `bodyText`
line and puts its `R... —` explanation on the next line, while the parser first
flattens those lines and only recognizes the legacy inline `R... (` delimiter.

This workflow/process slice is justified by the immediate vertical proof it
unblocks: #2481 cannot pass its required trusted-base check until the provider
on `main` understands the current review shape. It belongs to the existing
CI/CD enforcement arc [#2260](https://github.com/canfieldjuan/ATLAS/issues/2260).

### Problem-derived contract

- Root cause: parsing first flattened `bodyText`, then treated any nonblank or
  alphanumeric prefix as a matching title. Thus `---` was skipped and
  `x\nReal decision R2 (BLOCKER) details` could match `- x -- fixed-in: ...`;
  the snippet fallback kept malformed history eligible for reconciliation.
- Required change: a correlating root needs a full phrase (at least four
  normalized tokens and 24 normalized characters) plus existing `R... (` rule
  evidence on its line (legacy) or the next nonblank line (multiline). Apply
  that floor before exact or containment matching; displays/snippets cannot
  supply a root. The focused tests must prove valid current and legacy forms,
  punctuation recovery, and fail-closed malformed/ambiguous/short forms.
- Evidence: the current trusted-bot corpus's 71 open-PR titles are full phrases
  (minimum 33 normalized characters, five tokens) followed by a rule label.
- Must not change: trusted-base checkout, bot allowlist, open-thread blocking,
  workflow/config/data/product behavior, H-18 code, or matching relaxation.

## Scope (this PR)

Ownership lane: workflow/live-reconciliation-title-delimiter
Slice phase: Workflow/process
Max files: 3

1. Derive and match a correlating decision only when it has bounded full-title
   evidence paired with existing `R... (` rule evidence: same-line legacy or
   next-line multiline.
2. Keep display titles/snippets non-authoritative and remove their flattened
   fallback from history correlation.
3. Fail closed when a trusted bot thread supplies no bounded title evidence,
   including punctuation, ambiguous-prefix, and short-inline shapes.
4. Add direct regression proof for current multiline and full legacy forms,
   varied nonsemantic prefixes, ambiguous-prefix and short-inline failures.
5. Make no workflow, status-policy, or H-18 receipt change; merge this provider
   before refreshing #2481's required trusted-base check.

### Review Contract

- Current multiline `Title\nR... — detail`, full legacy `Title R... (BLOCKER)`,
  and punctuation prefixes before a valid pair correlate to the named root.
- Mismatched roots remain rejected. `---`, `x\nReal decision R2 (BLOCKER)`,
  `x R2 (BLOCKER)`, and no-title bodies remain unreconciled despite generic
  dispositions.
- `.github/workflows/ai_reconciliation_live.yml` still invokes this entrypoint
  from trusted base; direct `evaluate` tests exercise its returned gate result.
- Affected surface: provider parser/tests consumed by #2481. Risks: false
  acceptance, legacy compatibility, and trusted-base behavior. Rules: R1, R2,
  R10, R13, R14.
- Reviewer rules triggered: R1, R2, R10, R13, R14.

### Boundary-change enumeration

- Seam: GitHub `bodyText` -> `_bot_thread_summary` (display vs evidenced
  decision) -> `_thread_root_decision` -> `missing_thread_dispositions` ->
  `evaluate`.
- Multiline uses the next rule label; legacy uses same-line evidence. No bounded
  decision becomes the fixed unparseable failure, never a display/snippet root.
- Guard fields are bot-filtered `bodyText`, structured PR-body roots, and
  resolved state; no fields or allowlists are added.

### Deployed-config probing

N/A - no guard/config boundary change. The trusted workflow's base-SHA checkout
and invocation remain unchanged; this slice only preserves the review payload's
existing title boundary.

### Files touched

- `plans/PR-Live-Reconciliation-Title-Delimiter.md`
- `scripts/check_ai_reconciliation_live.py`
- `tests/test_check_ai_reconciliation_live.py`

## Mechanism

The parser derives a root only from the title/rule-label relationship above and
uses the same floor for exact and containment comparison. Displays remain for
diagnostics only. No title evidence produces the fixed unparseable failure, so
the code does not need a punctuation or prefix vocabulary.

## Intentional

- No PR-body filler, title/punctuation vocabulary, bot-identity change,
  matching relaxation, workflow change, or H-18 change.
- A generic unparseable or short-prefix disposition cannot waive malformed
  history; trusted-base execution remains the security boundary.

## Deferred

None.

Parked hardening: none.

## Verification

- Live evidence covers current `R... —` and current `R... (BLOCKER)` payloads;
  71 open-PR titles establish the bounded-title floor.
- `python -m py_compile scripts/check_ai_reconciliation_live.py`, focused
  Pytest, Ruff, plan/PR-body audits, and whitespace checks must pass. Do not
  run the broad Unit Gate locally; GitHub owns it.
- After provider merge, rerun #2481's trusted-base reconciliation. Rollback is
  a commit revert; no data/configuration/customer behavior changes.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Live-Reconciliation-Title-Delimiter.md` | 126 |
| `scripts/check_ai_reconciliation_live.py` | 83 |
| `tests/test_check_ai_reconciliation_live.py` | 177 |
| **Total** | **386** |
