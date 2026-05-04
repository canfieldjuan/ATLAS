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
- deterministic witness specificity pack (`evaluate_witness_specificity` + 6 legacy entry points)
- async evidence-coverage gate (`evaluate_evidence_coverage` + lifted `audit_witness_evidence_coverage`)

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
from extracted_quality_gate.witness_pack import (
    evaluate_witness_specificity,
    surface_specificity_context,
    specificity_audit_snapshot,
)
from extracted_quality_gate.evidence_pack import (
    audit_witness_evidence_coverage,
    evaluate_evidence_coverage,
)
```

Products should import from:

- `extracted_quality_gate.api`
- `extracted_quality_gate.blog_pack`
- `extracted_quality_gate.campaign_pack`
- `extracted_quality_gate.evidence_pack`
- `extracted_quality_gate.safety_gate`
- `extracted_quality_gate.witness_pack`
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

## Witness specificity pack (PR-B5b)

`witness_pack.py` ships two API styles. The legacy entry points
(`surface_specificity_context`, `merge_specificity_contexts`,
`specificity_signal_terms`, `evaluate_specificity_support`,
`specificity_audit_snapshot`, `campaign_proof_terms_from_audit`)
keep their original signatures; the atlas-side
`atlas_brain/autonomous/tasks/_b2b_specificity.py` re-exports them
verbatim so existing imports continue to resolve. The pack-contract
entry point `evaluate_witness_specificity(input, *, policy)` returns
a standard `QualityReport`.

The validators are pure: they operate on already-fetched anchor /
witness rows + candidate text. The pack does not touch the database,
the clock, or the network. ``QualityPolicy.thresholds`` accepts:

- `min_anchor_hits` (int, default 1)
- `require_anchor_support` (bool, default True)
- `require_timing_or_numeric_when_available` (bool, default False)
- `include_competitor_terms` (bool, default True)
- `allow_company_names` (bool, default depends on `surface`)

The blog and campaign packs compose against this pack: today they
call `evaluate_specificity_support` directly through the atlas
re-export; future revisions can pass `evaluate_witness_specificity`
findings through `QualityInput.context` for cleaner pack-to-pack
composition.

## Evidence-coverage gate (PR-B5a)

`evidence_pack.py` ships two entry points:

- `audit_witness_evidence_coverage(pool, *, vendor_name,
  source_review_ids, min_pain_confidence, valid_status,
  coverage_precision)` -- legacy, lifted verbatim from
  `atlas_brain/services/b2b/evidence_gate.py`. Returns the same
  `dict[str, Any]` shape, two new keyword args are additive
  (default to the legacy values).
- `evaluate_evidence_coverage(pool, input, *, policy)` -- pack
  contract. Reads `vendor_name`, `source_review_ids`, optional
  `min_pain_confidence` from `input.context`. Reads thresholds
  (`coverage_block_threshold`, `coverage_warn_threshold`,
  `min_pain_confidence`, `valid_status`, `coverage_precision`) from
  `policy.thresholds`. Returns a `QualityReport`.

The pain-confidence rank map is part of the contract
(`b2b_evidence_claims.pain_confidence_rank` is a STORED generated
column with values 0/1/2 for strong/weak/none) and not parametric.
The coverage thresholds, the valid-status string, and the rounding
precision ARE parametric.

The atlas-side `atlas_brain/services/b2b/evidence_gate.py` is a
thin re-export wrapper, so the existing shadow-mode call site in
`atlas_brain/autonomous/tasks/b2b_campaign_generation.py` keeps
working without change.
