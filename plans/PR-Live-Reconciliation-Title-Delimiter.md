# PR-Live-Reconciliation-Title-Delimiter

## Why this slice exists

The user asked to repair the red checks on owned H-18 receipt PR #2481. Its two
new P1 findings were fixed and resolved, but trusted `live-reconciliation`
still reports them missing from the PR-body ledger. This base-provider defect
comes from current `bodyText` placing a title on line one and `R... —` evidence
on line two, while the parser flattened input and recognized legacy inline
`R... (` only. It unblocks #2481's required trusted-base proof and belongs to
CI/CD enforcement [#2260](https://github.com/canfieldjuan/ATLAS/issues/2260).

### Problem-derived contract

- Root cause: parsing first flattened `bodyText`, then treated any nonblank or
  alphanumeric prefix as a matching title. Thus `---` was skipped and
  `x\nReal decision R2 (BLOCKER) details` could match `- x -- fixed-in: ...`;
  the snippet fallback kept malformed history eligible for reconciliation.
- Required change: a correlating root needs a full phrase (at least four
  normalized tokens and 24 normalized characters) plus a complete `R...` label
  (balanced optional severity and nonempty detail delimiter) on its line
  (legacy) or the next nonblank line (multiline); displays/snippets cannot
  supply a root. Test valid grammar and fail-closed malformed/ambiguous forms.
- Evidence: the current trusted-bot corpus's 71 open-PR titles are full phrases
  (minimum 33 normalized characters, five tokens) followed by a rule label.
- Must not change: trusted-base checkout, bot allowlist, open-thread blocking,
  workflow/config/data/product behavior, H-18 code, or matching relaxation.

## Scope (this PR)

Ownership lane: workflow/live-reconciliation-title-delimiter
Slice phase: Workflow/process
Max files: 3

1. Derive and match only bounded full titles paired with a complete `R...`
   label: same-line legacy or next-line multiline.
2. Keep display titles/snippets non-authoritative; malformed, ambiguous, and
   short evidence fails closed.
3. Add grammar-derived proof for current/legacy forms, nonsemantic prefixes,
   incomplete labels, and short/ambiguous titles.
4. Change no workflow, status policy, or H-18 receipt code; merge this provider
   before refreshing #2481's trusted-base check.

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

### Closure declaration

- Title text is OPEN and never semantically classified. The finite positive
  evidence is DERIVED from `_COMPLETE_RULE_LABEL_RE`: bounded title plus a full
  rule reference, balanced optional severity, and nonempty delimiter/detail.
- Any unmatched, incomplete, or novel label produces no decision and blocks
  reconciliation; GitHub `bodyText` -> summary -> disposition check is the sole
  choke point.

### Deployed-config probing

N/A: no guard/config boundary change; trusted base-SHA checkout and invocation
remain unchanged.

### Files touched

- `plans/PR-Live-Reconciliation-Title-Delimiter.md`
- `scripts/check_ai_reconciliation_live.py`
- `tests/test_check_ai_reconciliation_live.py`

## Mechanism

The parser derives roots only from the title/rule-label relationship and uses
the same floor for exact and containment matching; no evidence is unparseable.

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
| `plans/PR-Live-Reconciliation-Title-Delimiter.md` | 113 |
| `scripts/check_ai_reconciliation_live.py` | 97 |
| `tests/test_check_ai_reconciliation_live.py` | 207 |
| **Total** | **417** |
