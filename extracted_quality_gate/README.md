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
```

Products should import from:

- `extracted_quality_gate.api`
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
