# PR-Live-Reconciliation-Colon-Delimiter

## Why this slice exists

Owned H-18 PR #2484 fixed and resolved its two concrete Codex findings, but the
trusted-base `live-reconciliation` check remains red. Current GitHub
`bodyText` preserves the trusted bot's original finding as a title followed by
`R4 (BLOCKER): detail`. `_COMPLETE_RULE_LABEL_RE` recognizes dash and
whitespace forms but not that complete severity-colon form, so the historical
thread is reported unparseable despite a matching structured disposition. This
is a real CI enforcement defect, not a PR-body problem, and blocks a current
vertical migration-safety proof; it belongs under CI/CD enforcement gap tracker
[#2035](https://github.com/canfieldjuan/ATLAS/issues/2035).

### Problem-derived contract

- Root cause: the trusted parser's complete-label grammar omits `:` after a
  balanced severity label, while the trusted connector emits that grammar.
- Correct fix: expand only the severity-qualified complete-label grammar to
  admit a colon followed by nonempty detail; preserve the single parser choke
  point and make incomplete, severity-less-colon, or unknown forms yield no
  decision and therefore block reconciliation.
- Must not change: bot identity/configuration, trusted-base workflow,
  unresolved-thread blocking, PR-body disposition matching, or #2484's
  migration behavior.

## Scope (this PR)

Ownership lane: workflow/live-reconciliation-colon-delimiter
Slice phase: Workflow/process
Max files: 3

1. Add a complete `R<n> (SEVERITY): nonempty detail` delimiter alternative to
   the trusted title-evidence parser.
2. Prove both multiline and inline severity-colon labels correlate to the
   intended bounded title.
3. Prove bare/trailing severity-colon labels and severity-less-colon labels
   remain unreconciled.

### Review Contract

- `scripts/check_ai_reconciliation_live.py` accepts the exact real producer
  shape `Title\nR4 (BLOCKER): detail`, and `evaluate` accepts a matching
  structured disposition; settled by the focused test.
- The same colon grammar works for inline legacy title/evidence text; settled
  by a direct `evaluate` assertion.
- Incomplete labels and a `R4: detail` label without a balanced severity still
  return no root decision, so no generic PR-body disposition can admit them;
  settled by negative assertions.
- The only live reachability path remains GitHub review `bodyText` ->
  `_evidenced_root_decision` -> `evaluate` -> required
  `live-reconciliation` result. After provider merge, its observable result is
  a fresh green trusted-base check for #2484.
- Affected surfaces: the trusted GitHub CI parser and its direct unit tests.
  Risk areas: false acceptance of malformed review history, false rejection of
  valid trusted history, and trusted-base deployment ordering.
- Reviewer rules triggered: R1, R2, R10, R12, R13, R14.

### Boundary-change enumeration and closure declaration

- Decision seam: `_COMPLETE_RULE_LABEL_RE` is the only grammar that admits a
  review title as reconciliation evidence.
- Inputs are OPEN producer text. The accepted evidence grammar is CLOSED and
  DERIVED at each use from `_COMPLETE_RULE_LABEL_RE`; its added finite form is
  a balanced severity plus `:` plus nonempty detail.
- Existing dash/whitespace forms are preserved. A severity-colon form with
  nonempty detail is intentionally added. A bare/trailing colon, a
  severity-less-colon form, and every other unrecognized producer shape are
  DEFAULTED to no decision, the safe side because false reconciliation could
  conceal a real review finding.

### Deployed-config probing

No configuration or environment fallback changes. The default exact trusted
bot allowlist, its explicit override parsing, and the trusted-base checkout
remain outside this diff. GitHub's producer grammar is externally controlled;
the direct test uses its observed `bodyText` shape and the default parser path.

### Files touched

- `plans/PR-Live-Reconciliation-Colon-Delimiter.md`
- `scripts/check_ai_reconciliation_live.py`
- `tests/test_check_ai_reconciliation_live.py`

## Mechanism

The parser keeps title admission evidence-gated: a bounded title must have a
complete adjacent or inline rule label. The revised severity branch recognizes
only a colon followed by a nonempty detail token, then uses the same root
matching path already applied to dash and whitespace forms. The safe default
remains an empty root, which makes historical reconciliation fail rather than
accepting arbitrary review text.

## Intentional

- No producer-specific title allowlist or PR-body exception: the grammar, not
  the observed title text, is the durable source of admission.
- No workflow edit: the required check must continue to execute trusted base
  code; therefore this provider PR merges before refreshing #2484.
- No acceptance of `R4: detail` without a severity label or of a colon with no
  detail, because those forms would weaken the complete-evidence requirement.

## Deferred

Parking predicate: producer-format variants that are neither confirmed by the
observed live failure nor required by the bounded grammar are parked rather
than added speculatively.

Parked hardening: none.

## Verification

Run the focused parser test file, syntax compilation, Ruff, whitespace check,
and the repository PR wrapper's single local review. Do not run the broad Unit
Gate locally; GitHub owns its full suite. After merge, rerun/observe #2484's
trusted-base `live-reconciliation` check; rollback is a provider commit revert
with no customer, financial, or persistent-data effect.

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Live-Reconciliation-Colon-Delimiter.md` | 126 |
| `scripts/check_ai_reconciliation_live.py` | 5 |
| `tests/test_check_ai_reconciliation_live.py` | 59 |
| **Total** | **190** |
