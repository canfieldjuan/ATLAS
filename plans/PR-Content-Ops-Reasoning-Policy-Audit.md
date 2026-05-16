# PR: Content Ops Reasoning Policy Audit

## Why this slice exists

The active AI Content Ops backlog now points at reasoning product depth after
the source-adapter consolidation work landed. The runtime already has several
reasoning seams: file-backed context, DB-backed context, single-pass campaign
reasoning, and the multi-pass extracted reasoning-core bridge. What is missing
is the policy layer that says which richer reasoning controls should be exposed
per content type.

This slice is docs-only. It audits the shipped seams and records the next small
wiring sequence before any new falsification, narrative, or validation behavior
is added.

## Scope

1. Add a host-facing reasoning policy audit for AI Content Ops.
2. Classify current assets by reasoning consumption and appropriate depth.
3. Separate Content Ops reasoning policy from the Evidence-to-Story product.
4. Update the active backlog/status docs with the audit result and next pick.

### Files Touched

- `docs/audits/content_ops_reasoning_policy_audit_2026-05-16.md`
- `docs/audits/ai_content_ops_deferred_backlog_2026-05-11.md`
- `extracted_content_pipeline/STATUS.md`
- `plans/PR-Content-Ops-Reasoning-Policy-Audit.md`

## Mechanism

The audit reads the actual shipped seams:

- `ContentOpsExecutionServices.with_reasoning_context(...)`
- generated asset services with `with_reasoning_context(...)`
- `MultiPassCampaignReasoningProvider` and its optional policy config
- campaign operations API config for single-pass and multi-pass reasoning
- control-surface catalog `reasoning_requirement` metadata

It produces a policy table and a concrete follow-up order rather than adding
new runtime code in this PR.

## Intentional

- No runtime behavior changes.
- No new reasoning provider config fields.
- No prompt or quality-pack changes.
- No Evidence-to-Story work; that remains a separate product.
- Coordination row was used while the PR was open and removed before merge.

## Deferred

- Control-surface policy presets for reasoning depth.
- Host API fields for richer multi-pass policies.
- Per-asset enforcement tests for any new policy presets.
- Evidence-to-Story orchestration and storytelling state management.

## Verification

- Run `rg` checks for the new audit references and stale wording.
- Run the local review script at `scripts/local_pr_review.sh`.

## Estimated Diff Size

| Area | Estimated LOC |
|---|---:|
| Reasoning policy audit | ~170 |
| Backlog/status updates | ~35 |
| Plan | ~60 |
| **Total** | ~270 |
