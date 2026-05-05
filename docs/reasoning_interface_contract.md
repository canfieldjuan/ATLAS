# Reasoning Interface Contract (Hybrid Extraction PR-1)

This document defines the canonical interface contract between reasoning producers and reasoning consumers across Atlas and extracted products.

## Objectives

1. Keep consumer payloads stable while producer internals evolve.
2. Preserve backward compatibility for current v1/v2 synthesis-backed consumers.
3. Standardize provenance and confidence semantics for cross-product reuse.

## Scope

In-scope:
- Consumer-facing reasoning payload shape.
- Provenance and lineage fields.
- Confidence semantics and required invariants.
- Backward compatibility and versioning policy.

Out-of-scope:
- Producer-specific prompt design.
- Product-specific ontology extensions.
- Non-reasoning API/domain payloads.

## Contract layers

- **Layer A: Producer Port Contract**
  - Host-owned interface used to supply or compute reasoning payloads.
- **Layer B: Canonical Reasoning Contract**
  - Stable shape consumed by MCP/API/UI and extracted products.
- **Layer C: Consumer Adapter Contract**
  - Per-product adapter that maps canonical contract to local view model.

## Canonical reasoning payload

Required top-level keys:
- `contract_version` (string)
- `vendor_name` (string)
- `as_of_date` (ISO date string)
- `mode` (string)
- `risk_level` (string)
- `confidence` (number in [0,1])
- `confidence_label` (string)
- `executive_summary` (string)
- `reasoning_contracts` (object)
- `reference_ids` (object)
- `packet_artifacts` (object)

Optional top-level keys:
- `quality_status` (string)
- `quality_reasons` (array of strings)
- `archetype` (string)
- `uncertainty_sources` (array)
- `falsification_conditions` (array)

### reasoning_contracts (required object)

Must include these logical blocks (may be empty if confidence is insufficient):
- `vendor_core_reasoning`
- `displacement_reasoning`
- `category_reasoning`
- `account_reasoning`

### reference_ids (required object)

- `metric_ids`: array of strings (deduplicated)
- `witness_ids`: array of strings (deduplicated)

Invariant:
- If a section has non-empty evidence claims/citations, at least one ID must resolve into `metric_ids` or `witness_ids`.

### packet_artifacts (required object)

If present in source synthesis payload, must be carried through unchanged except for additive normalization.

Known subkeys:
- `witness_pack`
- `section_packets`

## Confidence semantics

Canonical confidence labels:
- `high`
- `medium`
- `low`
- `insufficient`

Mapping rule from numeric confidence (float [0,1]):
- `high` if >= 0.75
- `medium` if >= 0.45 and < 0.75
- `low` if >= 0.15 and < 0.45
- `insufficient` if < 0.15 or missing/invalid

Invariants:
1. `confidence_label` must match mapped band from `confidence`.
2. `risk_level` can differ by product, but must be explicit and non-empty.
3. Consumers must not infer higher confidence than contract declares.

## Provenance semantics

1. `metric_ids` are IDs for aggregate/metric evidence anchors.
2. `witness_ids` are IDs for witness/source-row anchors.
3. IDs must be stable strings within the producer scope and analysis window.
4. Consumers may display provenance badges only when at least one ID is present.

## Backward compatibility policy

### Contract versioning

- `contract_version` format: `major.minor` (string).
- Minor bump (`1.x` -> `1.y`) for additive fields.
- Major bump (`1.x` -> `2.0`) for removals/renames/semantic breaks.

### Compatibility guarantees

1. Existing consumers must continue to function if only additive fields are introduced.
2. Existing keys in the canonical payload cannot change type in the same major version.
3. Missing optional fields must degrade gracefully.

### v1/v2 synthesis source compatibility

Adapters must normalize both source forms into this canonical contract by:
- preserving reference-id extraction behavior,
- preserving packet artifact fallback/merge behavior,
- preserving confidence normalization rules.

## Producer Port Contract

Producer interface requirements:

- Input:
  - subject key (`vendor_name` or equivalent)
  - analysis window metadata (`as_of_date`, `analysis_window_days`)
  - optional product-specific context object
- Output:
  - canonical reasoning payload
- Error behavior:
  - fail closed with explicit structured error payload (no silent partials)

Producer implementation constraints:
- No direct consumer-specific schema shaping.
- No hidden side effects outside configured persistence path.

## Consumer Adapter Contract

Consumer adapter requirements:

1. Accept canonical reasoning payload only.
2. Produce local DTO/view-model without mutating canonical payload.
3. Preserve provenance fields in local model where relevant.
4. Preserve confidence label and numeric confidence.

## Validation checklist

Each adapter/producer PR must validate:

1. Canonical payload includes required top-level keys.
2. `confidence_label` matches confidence mapping bands.
3. `reference_ids.metric_ids` and `reference_ids.witness_ids` are deduplicated string arrays.
4. Canonical payload remains parseable when optional fields are absent.
5. Existing MCP/API schema remains unchanged unless explicitly versioned.

## CI guardrails (recommended)

1. Schema conformance test for canonical payload.
2. Snapshot tests for representative high/medium/low/insufficient cases.
3. Regression test ensuring existing consumers still read v1/v2-derived canonical payloads.
4. Import-boundary test to prevent forbidden runtime coupling for extracted packages.

## Ownership

- **Contract owner**: Platform Architecture
- **Producer implementations**: Product teams
- **Consumer adapters**: Product teams with platform review

## Change process

1. Propose change with example payload diff and compatibility statement.
2. Classify as additive (minor) or breaking (major).
3. Update this contract and linked execution board in same PR.
4. Run schema/regression checks before merge.

## Domain-agnostic envelope (M5-alpha)

The vendor-pressure-flavored canonical payload above is the v1 contract for
the churn-reasoning domain. M5-alpha (`docs/hybrid_extraction_plan_status_2026-05-05.md`)
introduces a domain-agnostic envelope so the same producer/consumer pattern
can be applied to other domains (call transcripts, internal company entities,
etc.) without rewriting the core.

The envelope lives in `extracted_reasoning_core.domains`:

- `ReasoningSubject` Protocol -- anything reasoning is performed over (`id`,
  `domain`, `payload`).
- `DomainReasoningResult[PayloadT]` -- universal core fields (`confidence`,
  `executive_summary`, `key_signals`, `uncertainty_sources`,
  `falsification_conditions`, `reference_ids`, etc.) plus a typed
  `domain_payload`.
- `ReasoningProducerPort[SubjectT, PayloadT]` and
  `ReasoningConsumerPort[PayloadT]` -- Protocols for the producer (subject
  in, result out) and consumer (result in, overlay-fields dict out) sides.
- `register_domain(name, subject_type, payload_type)` -- registry so each
  new domain ships its own typed triple without touching the core.

The vendor-pressure flow (current churn reasoning) becomes one specialization
of this envelope in M5-beta:

- `domain="vendor_pressure"`, `domain_payload=VendorPressurePayload(wedge=...,
  proof_points=..., account_signals=..., ...)`.
- `CampaignReasoningProviderPort` becomes
  `ReasoningProducerPort[VendorOpportunity, VendorPressurePayload]`.
- The vendor-specific `reasoning_contracts` keys (`vendor_core_reasoning`,
  `displacement_reasoning`, `category_reasoning`, `account_reasoning`) move
  under `domain_payload`; the universal core stays at the envelope level.

A second domain ships purely additively: define a new
`Subject` / `Payload` / `Producer` / `Consumer` triple, call
`register_domain`, and consumers that only need universal fields work without
code changes.
