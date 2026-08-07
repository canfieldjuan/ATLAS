# PR-Permission-Value-Shape-Validation

## Why this slice exists

A post-merge finding on ATLAS #2305. That PR added
`_permissions_are_explicitly_read_only`, which decides whether an enrolled
`pull_request_target` job may run with the base repository's token. It excluded
`id-token` by KEY before validating its VALUE, and compared every other value
against a frozenset.

Two consequences, both reproduced against the merged code on `main` before
writing any fix:

| Input | Before | Should be |
|---|---|---|
| `{id-token: [write]}` | `read_only=True`, `oidc_write=False` | rejected |
| `{contents: [read]}` | `TypeError: unhashable type: 'list'` | rejected |
| `{contents: {a: b}}` | `TypeError: unhashable type: 'dict'` | rejected |

The first is a fail-open through **both** guards at once: the admission
predicate saw no key other than the excluded `id-token` and returned True, while
`_permissions_write_oidc` compared against the scalar `"write"` and a list is
not that scalar. A workflow requesting OIDC write in that shape was admitted and
never reached the OIDC allowlist.

The second is worse than being wrong: `["read"] in frozenset(...)` raises, so
the auditor crashed rather than returning a verdict. A non-zero exit is not the
same as a rejection -- it stops the audit before later workflows are examined,
and it contradicts this predicate's own documented contract that unrecognized
shapes fall on the reject side.

This is a follow-up rather than an amendment because #2305 is already merged and
deployed.

### Problem-derived contract

- Root cause: shape was never validated. The predicate reasoned about MEANING
  (`is this value read-only`) on inputs whose TYPE it had not established, and
  applied a key-based exemption before that check.
- Correct fix must touch/change: `_permissions_are_explicitly_read_only` must
  validate key and value shape before the `id-token` exclusion and before any
  membership test; `_permissions_write_oidc` must not read a non-scalar
  `id-token` value as absence.
- Must not change: which legitimate shapes are admitted. `{contents: read}`,
  `{}`, `read-all`, and `{id-token: write}` must behave exactly as before, and
  `id-token` must remain governed by its own allowlist rather than by this
  predicate.

## Scope (this PR)

Ownership lane: workflow-security-posture
Slice phase: production hardening

1. Validate shape first in `_permissions_are_explicitly_read_only`: a non-string
   key or non-string value returns False before anything else is considered.
2. `_permissions_oidc_state` replaces the OIDC boolean with a tri-state --
   `none` / `write` / `invalid`. An unevaluable `id-token` value is `invalid`,
   which is an ERROR at both workflow and job scope and is **never**
   allowlistable.

### Files touched

- `plans/PR-Permission-Value-Shape-Validation.md`
- `scripts/audit_workflow_security_posture.py`
- `tests/test_audit_workflow_security_posture.py`

### Review Contract

1. Every unevaluable shape reaches the reject verdict rather than raising.
2. `{id-token: [write]}` is rejected by the admission predicate AND errors at
   the OIDC check -- both, since they are separate paths and either alone would
   leave a gap. Critically it errors even on the allowlisted
   `.github/workflows/claude.yml` / `claude` job: the first attempt at this fix returned a plain boolean, so the
   malformed value inherited that job's allowlist and downgraded to a WARN,
   indistinguishable from the reviewed `id-token: write`.
3. Legitimate shapes are unchanged: `{contents: read}`, `{}`, `read-all` admit;
   `write-all` rejects; `{id-token: write}` still admits at this predicate and
   is still caught by the OIDC allowlist.
4. No workflow, no allowlist entry, and no other guard-shape check is touched.

Affected surfaces: the two permission predicates in
`scripts/audit_workflow_security_posture.py`. `_permissions_write_oidc` is used
for **every** workflow, not only enrolled ones, so its blast radius is wider
than #2305's -- which is exactly why the non-scalar case mattered there too.

Risk areas: over-rejection. A shape this auditor refuses to evaluate now fails
the build, so a legitimate-but-unusual block would become a false positive.
Probed by pinning the five known-good shapes on the admit side of the same
table as the reject cases, and by running the auditor against the real workflow
tree (exit 0, so nothing currently in the repo is newly rejected).

- Reviewer rules triggered: R2, R3, R10, R13, R14.

R10 is the path trigger the rule pack assigns to gate-predicate scripts, which
is what `scripts/audit_workflow_security_posture.py` is. Satisfied by the change
being two type checks inserted ahead of existing logic rather than new decision
logic: no verdict for a well-formed input moves, which the admit-side rows pin.

R3 because this governs which identity may run under a privileged event. R13
because it changes membership semantics of `_READ_ONLY_PERMISSION_VALUES` --
values are now shape-checked before being tested against it. R2/R14 because the
claim was verified by executing the merged code, not by reading it.

**boundary-probe:** a 12-row table over both predicates covering list, dict,
bool, int, None, and non-string-key inputs alongside every legitimate shape.
Reject side: `{id-token: [write]}`, `{id-token: {a: b}}`, `{contents: [read]}`,
`{contents: {a: b}}`, `{contents: True}`, `{contents: 1}`, `{contents: None}`,
`{5: read}`, and a mixed block whose second value is a list. Admit side:
`{contents: read}`, `{}`, `read-all`, `{id-token: write}`, `{id-token: none}`.

**Mutation-probe (run, not asserted), twice:** removing the shape check makes 6
tests fail; separately, collapsing `OIDC_INVALID` back into `OIDC_WRITE` also
makes 6 fail. Restored before commit both times. The second probe is the one
that matters here -- it is the exact regression this round fixed.

**Guard-class closure declaration -- amendment to #2305's**

- `_READ_ONLY_PERMISSION_VALUES` remains CLOSED and ENUMERATED (`read`, `none`)
  from GitHub's documented vocabulary. Unchanged.
- **What changes is admission to the test.** A value is now tested for
  membership only after it is established to be a string. Previously a
  non-hashable value reached `in frozenset(...)` and raised.
- **`id-token` is still excluded by key, but only after its value passes the
  shape check.** The exclusion delegates to a different guard; it never meant
  "accept anything here".
- **Out-of-set default remains REJECT**, and now genuinely is reject rather
  than raise.

### Boundary-change enumeration

- Boundary path/seam: permission-block evaluation, both predicates.
- Replaced-path behaviours: unevaluable shapes move from
  `admitted`/`TypeError` to `rejected`. No legitimate shape changes verdict.
- Guard-relevant fields: every key and value of a `permissions` block, at both
  workflow and job scope.
- Caller x input shape: predicate x {list, dict, bool, int, None, non-string
  key, valid scalar}; auditor x the real workflow tree.

**Reachability proof:** `test_enrolled_job_with_list_valued_id_token_is_rejected`
runs the full `audit_workflow` over a workflow file carrying
`id-token:\n  - write`, so the rejection is proven through the audit rather than
at the predicate alone.

## Mechanism

Shape before meaning. Establish that a key and value are strings; then apply the
`id-token` delegation; then test membership.

## Intentional

- **`invalid` is its own state, not a flavour of `write`.** The first version
  of this fix made a non-scalar `id-token` return the same boolean as a real
  write request. That was still wrong, for a reason worth stating: the allowlist
  exists to permit a REVIEWED value on a REVIEWED job, so routing an
  unevaluable shape into it means permitting something nobody looked at. An
  `invalid` value therefore errors regardless of allowlist membership, while a
  valid `id-token: write` on the owner-gated Claude job keeps its WARN.
- **Reject rather than raise.** Both fail the build, but a raise aborts the
  audit before later workflows are examined and gives a stack trace instead of
  a finding.
- **A separate PR, not a rewrite of #2305.** That PR is merged and deployed;
  this is the narrowest change that closes the hole it left.

## Deferred

- The execution boundary after the base checkout: still ATLAS #2307.

Parking predicate: this slice parks everything except shape validation in the
two permission predicates.

Parked hardening: none.

## Verification

```
$ python -m pytest tests/test_audit_workflow_security_posture.py -q
66 passed

$ python scripts/audit_workflow_security_posture.py
(exit 0 -- nothing in the real workflow tree is newly rejected)
```

Before/after, run against the merged code on `main` and then against this tree:

```
input                  before                          after
{id-token: [write]}    read_only=True  oidc=False      read_only=False  oidc=True
{contents: [read]}     TypeError                       read_only=False
{contents: {a: b}}     TypeError                       read_only=False
{contents: read}       read_only=True                  read_only=True   (unchanged)
{id-token: write}      read_only=True  oidc=True       unchanged
{}                     read_only=True                  unchanged
read-all               read_only=True                  unchanged
write-all              read_only=False oidc=True       unchanged
```

## Estimated diff size

| File | LOC |
|---|---:|
| `plans/PR-Permission-Value-Shape-Validation.md` | 204 |
| `scripts/audit_workflow_security_posture.py` | 72 |
| `tests/test_audit_workflow_security_posture.py` | 169 |
| **Total** | **445** |
