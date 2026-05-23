# PR-Content-Ops-Marketing-Claim-Audit

## Why this slice exists
Support Ticket Deflection copy is moving in parallel with FAQ generator work.
The product can support strong claims, but public copy must not drift into live
integrations, auto-publishing, guaranteed reduction, semantic clustering, cost
ranking without cost data, or hosted unlimited/50k synchronous uploads.

This slice adds a local audit so product/landing-page docs can be checked before
PR review. Review follow-up tightened the audit so it fails loud on empty scan
scopes, reports unreadable files as audit errors, and catches the named support
platform/doc-site claims used in the funnel.

## Scope (this PR)
Ownership lane: content-ops/product-positioning

Slice phase: Workflow/process.

1. Add a script that scans repo-relative Markdown/text paths for prohibited
   Support Ticket Deflection claims.
2. Emit file/line/code diagnostics for each unsafe claim.
3. Fail loud when no Markdown/text files are scanned or a candidate file cannot
   be read.
4. Add fixture tests for happy path, unsafe claims, named platform/doc targets,
   allow comments, path-safety rejection, and `main()` success/error exits.

### Files touched

- `plans/PR-Content-Ops-Marketing-Claim-Audit.md`
- `scripts/audit_content_ops_marketing_claims.py`
- `tests/test_audit_content_ops_marketing_claims.py`

## Mechanism

`scripts/audit_content_ops_marketing_claims.py` accepts repo-relative files or
directories, defaults to `docs/products`, walks Markdown/text files, and reports
unsafe claim lines. `claim-audit: allow <reason>` permits quoted anti-patterns.
The allow marker must live in a trailing HTML comment with a substantive reason.
Absolute paths, `..` traversal, empty scan scopes, missing paths, and files that
resolve outside the repository are rejected.

The script returns:

- `0` when at least one file was scanned and no unsupported claim was found.
- `1` when unsupported claim findings were found.
- `2` when the audit could not reliably scan the requested scope.

## Intentional

- No existing product copy is changed in this slice.
- The audit is phrase-based, not a semantic classifier. It catches known risky
  claims cheaply and deterministically.
- Denial copy can still match the phrase rules. Authors should use the explicit
  allow comment for quoted anti-patterns or honest "we do not support X" copy.
- The script is not wired into the global local PR review yet; teams can run it
  against candidate marketing docs first, then we can promote it if it earns its
  keep.

## Deferred

- Updating or landing the in-progress ticket-deflection GTM doc remains with the
  product-copy session that created it.
- CI/local-review integration for this audit is a later workflow slice.
- Adding a richer claim policy file is deferred until the first rule churn
  appears.

## Verification

- Audit unit tests passed: 10 passed.
- Py compile passed for the audit script and tests.
- Audit against tracked `docs/products` files passed and reported scanned file
  count.
- Local PR review passed: `bash scripts/local_pr_review.sh --allow-dirty`.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | 88 |
| Audit script | 274 |
| Tests | 251 |
| **Total** | 613 |

Actual diff after review fixes is expected to exceed the 400 LOC target because
the review identified correctness gaps in the first version of this audit: dead
guaranteed-deflection phrasing, missing named platform targets, empty scan
success, read-error behavior, and missing `main()` exit coverage. Splitting the
fixes would leave a known-false-green audit open.
