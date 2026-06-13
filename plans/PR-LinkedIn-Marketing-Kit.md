# PR-LinkedIn-Marketing-Kit

## Why this slice exists

PR #1385 was opened as a draft personal-reference artifact without the normal
Atlas PR contract, then review blocked it for two concrete reasons: no
plan-backed PR shape, and marketing copy that overstated the shipped deflection
product outcome. This slice converts the artifact into an AGENTS-compliant docs
PR and rewrites the deflection-report example to describe defensible
capabilities rather than guaranteed ticket-volume or SEO outcomes.
Follow-up review also caught a narrower truthfulness issue: the copy must not
claim every answer traces to resolution history, because valid reports can flag
draft answers for review when no proven resolution evidence exists.

## Scope (this PR)

Ownership lane: linkedin-marketing-kit/reference
Slice phase: Product polish

1. Land the LinkedIn positioning kit as a docs reference artifact, not product
   code.
2. Soften the anchored deflection-report post from outcome guarantees to
   capability-level wording backed by current product behavior.
3. Convert the PR body/plan shape from draft/no-review into the normal
   AGENTS.md contract.

### Review Contract

Acceptance criteria:
- The PR is ready for review, not draft.
- The PR body starts with `Plan:` and `Slice phase:` and includes Intentional,
  Deferred, Parked hardening, Verification, and Diff size sections.
- The kit does not claim guaranteed fewer tickets, ticket prevention, or SEO
  ranking outcomes.
- The kit does not claim every answer is resolution-backed; it distinguishes
  proven-resolution answers from review-needed gaps.
- The artifact lives under `docs/`, not at repository root.

Affected surfaces:
- `docs/linkedin_developer_marketer_kit.md`
- `plans/PR-LinkedIn-Marketing-Kit.md`

Risk areas:
- Marketing claim overreach: copy must stay at the defensible capability level.
- Repository clutter: personal/reference material should not sit at root.

Triggered reviewer rules:
- R1 Requirements match
- R2 Test evidence
- R6 Output truthfulness
- R14 Codebase verification

### Files touched

- `docs/linkedin_developer_marketer_kit.md`
- `plans/PR-LinkedIn-Marketing-Kit.md`

## Mechanism

The existing kit remains a static Markdown reference. The only content behavior
change is in the anchored support-ticket post: it now says the system identifies
repeated support-ticket patterns and drafts reviewable FAQ/content candidates
from uploaded evidence, instead of promising fewer repeat tickets or ranking.
The evidence sentence also says draft answers trace to proven resolution
history only when present, and that gaps are flagged for review.
The file is moved under `docs/` so the repo root stays for project entry points
and top-level contracts.

## Intentional

- No executable code or product surface changes. This is a reference/document
  artifact, so verification is contract/diff hygiene rather than runtime tests.
- The kit is not placed in `plans/` because it is not an in-flight PR plan; it
  belongs under `docs/` if it lands in the repository at all.

## Deferred

None.

Parked hardening: none.

## Verification

- python scripts/sync_pr_plan.py plans/PR-LinkedIn-Marketing-Kit.md: passed.
- python scripts/sync_pr_plan.py --check plans/PR-LinkedIn-Marketing-Kit.md: passed.
- python scripts/audit_plan_doc.py plans/PR-LinkedIn-Marketing-Kit.md: passed.
- git diff --check: passed.
- bash scripts/push_pr.sh /tmp/atlas-pr-1385-body.md --force-with-lease origin HEAD: passed.

## Estimated diff size

| File | LOC |
|---|---:|
| `docs/linkedin_developer_marketer_kit.md` | 166 |
| `plans/PR-LinkedIn-Marketing-Kit.md` | 96 |
| **Total** | **262** |
