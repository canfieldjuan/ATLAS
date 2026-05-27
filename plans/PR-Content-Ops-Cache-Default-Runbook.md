# PR: Content Ops Cache Default Runbook

## Why this slice exists

PR-Content-Ops-Cache-Policy-Default wired an env-backed hosted cache-policy
default for Content Ops. The implementation deliberately fails loud when the
global env value is invalid, because silently changing cache posture would be
dangerous. Operators need the runbook to name the accepted values, the failure
mode, and the customer-data privacy behavior before they set the env in a
hosted environment.

## Scope (this PR)

Ownership lane: content-ops/cost-surfacing
Slice phase: Product polish

1. Add a host runbook section for the Content Ops cache-policy default.
2. Document accepted default values and their normalized meanings.
3. Document that invalid values fail hosted Content Ops generation requests
   loudly instead of falling back silently.
4. Document that support-ticket/customer-upload runs remain no-store unless the
   separate customer-data exact-cache setting is enabled.

### Files touched

| File | Change |
|---|---|
| `plans/PR-Content-Ops-Cache-Default-Runbook.md` | Plan doc for the operator runbook slice. |
| `extracted_content_pipeline/docs/host_install_runbook.md` | Document hosted Content Ops cache-policy default configuration and failure behavior. |

## Mechanism

The runbook gets a new configuration subsection near database and provider env
setup. It names `ATLAS_B2B_CAMPAIGN_CONTENT_OPS_CACHE_POLICY_DEFAULT` as the
Atlas host env default, lists the accepted spellings that flow through
`normalize_content_ops_cache_policy`, and explains that the default only fills
the existing request field when a run did not explicitly choose a cache policy.

## Intentional

- This is docs-only. The cache default behavior and host wiring already landed
  in the prior production-hardening slice.
- This does not add DB-backed tenant settings or UI editing. Those are larger
  product decisions and remain separate from the runbook gap.
- This does not recommend enabling exact cache for customer uploads by default.
  The docs keep the same conservative privacy posture as the implementation.

## Deferred

- Future PR: DB-backed tenant cache settings if operators need durable in-app
  defaults instead of an environment-level default.
- Future PR: UI surface for editing or displaying the tenant cache default.
- Parked hardening: none. Root `HARDENING.md` has no active
  cost-surfacing parked items; `ATLAS-HARDENING.md` contains blog/deep-dive
  content items outside this lane.

## Verification

- python scripts/audit_plan_doc.py plans/PR-Content-Ops-Cache-Default-Runbook.md
  — passed.
- rg -n "ATLAS_B2B_CAMPAIGN_CONTENT_OPS_CACHE_POLICY_DEFAULT|EXTRACTED_CAMPAIGN_LLM_CUSTOMER_DATA_EXACT_CACHE_ENABLED|exact-cache|no-store" extracted_content_pipeline/docs/host_install_runbook.md extracted_content_pipeline/content_ops_cache_policy.py atlas_brain/config.py
  — confirmed the docs values match the implementation constants and host env
  name.
- git diff --check — passed.
- bash scripts/local_pr_review.sh --current-pr-body-file <git-path>/codex-pr-bodies/cache-default-runbook.md
  — passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Plan doc | ~65 |
| Runbook docs | ~35 |
| **Total** | **~100** |

This stays below the 400 LOC soft cap.
