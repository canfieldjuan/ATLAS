# Extracted Quality Gate Status

Date: 2026-05-03

## Current Slice

PR-B3: split safety-gate primitives.

- Deterministic core (`safety_gate.py`) -- `check_content` + `assess_risk`
- Atlas-side wrapper (`atlas_brain/services/safety_gate.py`) now delegates
  the pure functions to this package; approvals + audit-log + DB writes
  remain Atlas-side behind the existing `ApprovalStore` and `AuditLog`
  port protocols.

PR-B2 (merged via #85) is the prior slice:

- `product_claim.py`
- `api.py`
- `types.py`
- `ports.py`

The module is deterministic and imports without Atlas.

## Included

- Product claim envelope
- Evidence posture derivation
- Confidence derivation
- Render/report gate decisions
- Claim policy registry
- Generic quality report types
- Integration port protocols
- Safety gate (deterministic core: `check_content` + `assess_risk`) -- PR-B3

## Not Yet Included

- Safety gate Atlas adapter wrapper (approvals + audit log + DB stay
  in `atlas_brain/services/safety_gate.py`; the deterministic core
  is now in `extracted_quality_gate/safety_gate.py` and the wrapper
  delegates to it -- but the wrapper itself is not yet extracted)
- Blog quality pack (PR-B4)
- Campaign quality pack (PR-B4)
- Witness render policy pack (PR-B5)
- Evidence-claim coverage pack (PR-B5)
- Source-quality ingest pack (PR-B5)
- Memory quality pack
