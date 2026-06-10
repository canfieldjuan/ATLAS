# PR-Deflection-Synonym-Clustering-Recall

## Why this slice exists

PR #1410 added deterministic raw support-ticket clustering for the deflection
paid-funnel path. Later review left one in-lane residual: themes with no
shared raw token still fragment even when customers are describing the same
support issue with common synonyms.

This slice keeps the no-LLM trust story and closes one narrow, launch-relevant
piece of that ceiling: login/access complaints such as "locked out", "account
access denied", and "sign-in rejected" should converge deterministically before
the free preview and paid FAQ report present cluster-quality diagnostics.

## Scope (this PR)

Ownership lane: deflection/clustering-raw-data
Slice phase: Production hardening

1. Add conservative support-ticket phrase/token folds for common login/access
   synonyms that otherwise share no raw token.
2. Keep clustering deterministic and local: no LLM, embeddings, fuzzy matching,
   or customer text leaving the package.
3. Add a regression test with varied held-out wording that clusters only after
   canonicalization, not because the rows share an obvious raw token.

### Review Contract

- Acceptance criteria:
  - [ ] "locked out", "account access denied", and "sign-in rejected" style
        rows converge into one deterministic login cluster.
  - [ ] API/webhook authentication failures do not borrow the login anchor.
  - [ ] The proof rows do not rely on explicit categories, synthetic ticket
        IDs, or a shared raw topic token.
  - [ ] Existing export, SSO, billing, dashboard, HTML-strip, and untagged CSV
        clustering tests keep their current behavior.
  - [ ] No LLM/embedding/fuzzy semantic path is introduced.
- Affected surfaces: support-ticket token normalization and deterministic
  clustering in `extracted_content_pipeline`.
- Risk areas: over-broad synonym folds, false-positive merging, regression of
  existing anchor labels.
- Reviewer rules triggered: R1, R2, R10, R13.

### Files touched

- `extracted_content_pipeline/support_ticket_clustering.py`
- `plans/PR-Deflection-Synonym-Clustering-Recall.md`
- `tests/test_extracted_support_ticket_input_package.py`

## Mechanism

The support-ticket tokenizer already normalizes phrase and token variants
before cluster matching. This PR extends that deterministic normalization with
a small login/access synonym set:

- phrase folds for `sign-in`, `locked out`, `account access`, and direct
  "cannot access account" wording;
- low-signal anchor treatment for generic auth/authentication variants so they
  cannot merge API/webhook auth failures into customer login complaints.

The existing cluster matcher then sees the canonical `login` token and uses
the current anchor/overlap path. There is no new clustering engine; this only
improves the normalized input vocabulary for a bounded support-domain concept.

## Intentional

- This is not a broad semantic synonym system. It adds a conservative,
  inspectable support-ticket vocabulary fold for one common launch-relevant
  class.
- This does not map every use of `access` to `login`; only account-access
  phrases fold, because access can also mean permissions or file access.
- This does not map generic `auth` or `authentication` tokens to `login`;
  those terms remain visible but cannot become the shared anchor on their own.
- This does not change preview UI or report rendering. Those surfaces consume
  the existing cluster summary/quality outputs.

## Deferred

- Future robust-testing slice: expand deterministic concept folds only from
  sanitized real exports and held-out mutation probes.

Parked hardening: none.

## Verification

- Focused support-ticket input pytest:
  python -m pytest tests/test_extracted_support_ticket_input_package.py -q
  - Result: `41 passed in 0.29s`.
- Extracted content package validation:
  bash scripts/validate_extracted_content_pipeline.sh
  - Result: passed.
- Reasoning-import audit:
  python extracted/_shared/scripts/forbid_atlas_reasoning_imports.py extracted_content_pipeline
  - Result: clean.
- Extracted standalone audit:
  python scripts/audit_extracted_standalone.py --fail-on-debt
  - Result: `Atlas runtime import findings: 0`.
- ASCII policy:
  bash scripts/check_ascii_python.sh
  - Result: passed.
- Full extracted pipeline CI mirror:
  bash scripts/run_extracted_pipeline_checks.sh
  - Result: `3566 passed, 10 skipped, 1 warning in 53.98s`.
- Local PR review:
  bash scripts/local_pr_review.sh
  - Result: passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `extracted_content_pipeline/support_ticket_clustering.py` | 18 |
| `plans/PR-Deflection-Synonym-Clustering-Recall.md` | 115 |
| `tests/test_extracted_support_ticket_input_package.py` | 56 |
| **Total** | **189** |
