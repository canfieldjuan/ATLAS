# Extracted Quality Gate

Standalone quality-gate core for AI outputs, claims, and evidence-backed render decisions.

The package contains:

- deterministic evidence posture derivation
- deterministic confidence derivation
- render/report gate decisions
- stable product claim IDs
- a policy registry for claim-type-specific thresholds
- generic quality-report types and integration ports
- deterministic safety-gate primitives (`check_content`, `assess_risk`)
- deterministic blog quality pack (`evaluate_blog_post`)
- deterministic campaign quality pack (`evaluate_campaign`)

The package intentionally has no Atlas runtime dependency. Product-specific behavior belongs in packs or adapters layered on top of the public API.

## Public Imports

```python
from extracted_quality_gate.api import (
    ClaimScope,
    build_product_claim,
    decide_render_gates,
)
from extracted_quality_gate.safety_gate import (
    assess_risk,
    check_content,
)
from extracted_quality_gate.blog_pack import evaluate_blog_post
from extracted_quality_gate.campaign_pack import evaluate_campaign
```

Products should import from:

- `extracted_quality_gate.api`
- `extracted_quality_gate.blog_pack`
- `extracted_quality_gate.campaign_pack`
- `extracted_quality_gate.safety_gate`
- `extracted_quality_gate.types`
- `extracted_quality_gate.ports`

Do not import from private internals when product packs are added.

## Safety gate split (PR-B3)

`safety_gate.py` is deterministic by construction: `check_content` is a regex scan against a stable label catalogue and `assess_risk` composes upstream signals into a `RiskAssessment`. Neither touches the database, network, or clock.

The Atlas-side wrapper (`atlas_brain/services/safety_gate.py`) layers the I/O surface on top:

- `intervention_approvals` table writes (request / approve / reject / list pending)
- `atlas_events` audit-log writes
- composite `gate_check()` that coordinates pure scan + DB writes

The wrapper consumes the core via direct import; the `ApprovalStore` and `AuditLog` Protocols defined in `ports.py` describe the I/O surface for future packs that want to plug a different backend.

## Blog quality pack (PR-B4a)

`blog_pack.evaluate_blog_post` is a pure validator over a `QualityInput`
(cleaned body in `content`, blueprint metadata in `context`) and a
`QualityPolicy` (per-topic word-count thresholds, pass score). It returns
a `QualityReport` whose `findings` enumerate every gate that fired:
content-too-short, missing/duplicate/unknown chart placeholders,
unresolved `{{token}}` placeholders, quote-count, review-period mention,
methodology disclaimer, required vendors, placeholder/internal links,
title/vendor match, category-outcome support, ungrounded data claims
(two-strategy: known-vendor lookup + multi-word capitalized-name regex
with configurable skip words), chart-scope ambiguity, numeric
consistency, migration-direction drift.

Sanitization (markdown cleanup + unmatched-quote removal) lives in the
Atlas wrapper, not the pack -- the pack validates only. Specificity
checks (witness-anchor support, evidence coverage) also stay
Atlas-side; PR-B5 ships them as their own pack.

The wrapper at `atlas_brain/autonomous/tasks/b2b_blog_post_generation.py:_apply_blog_quality_gate`
sanitizes, builds the input, calls the pack, then layers the
specificity findings on top. Public dict shape is preserved so existing
call sites need no changes.

## Campaign quality pack (PR-B4b)

`campaign_pack.evaluate_campaign` is a pure validator over a
`QualityInput` (subject, body, CTA, the campaign payload) and a
`QualityPolicy` (currently just `require_anchor_support`). It returns
a `QualityReport` whose `findings` enumerate every gate that fired:
proof-term coverage (`missing_exact_proof_term`), report-tier banned
language (`report_tier_language:<word>`), forbidden competitor name in
cold email (`competitor_name_in_email_cold:<vendor>`), forbidden
incumbent name in challenger-intel cold email
(`incumbent_name_in_email_cold:<vendor>`), and private-account leak
(`private_account_name_leak:<company>`).

The atlas-side specificity audit stays in the wrapper -- the wrapper
runs `specificity_audit_snapshot` first, passes its blocking issues
and warnings into the pack as `context['specificity_blocking_issues']`
/ `context['specificity_warnings']`, and the pack appends its own
findings. The wrapper at
`atlas_brain/autonomous/tasks/_b2b_specificity.py:campaign_policy_audit_snapshot`
returns the same dict shape it always did, so the existing caller in
`atlas_brain/services/campaign_quality.py` needs no changes.
