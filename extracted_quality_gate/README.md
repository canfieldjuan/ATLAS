# Extracted Quality Gate

Standalone quality-gate core for AI outputs, claims, and evidence-backed render decisions.

This first slice contains the product-claim contract:

- deterministic evidence posture derivation
- deterministic confidence derivation
- render/report gate decisions
- stable product claim IDs
- a policy registry for claim-type-specific thresholds
- generic quality-report types and integration ports

The package intentionally has no Atlas runtime dependency. Product-specific behavior belongs in packs or adapters layered on top of the public API.

## Public Imports

```python
from extracted_quality_gate.api import (
    ClaimScope,
    build_product_claim,
    decide_render_gates,
)
```

Products should import from:

- `extracted_quality_gate.api`
- `extracted_quality_gate.types`
- `extracted_quality_gate.ports`

Do not import from private internals when product packs are added.
