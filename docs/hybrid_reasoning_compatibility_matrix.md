# Hybrid Reasoning Compatibility Matrix (M6)

**Status:** PR-6 / M6 deliverable — closes the M1–M6 hybrid extraction program described in `docs/hybrid_extraction_plan_status_2026-05-05.md`.

**Audience:** product leads + platform engineers deciding whether a new reasoning use case can **reuse** the existing producer / consumer / envelope, or whether it has to **rebuild** the producer side.

**Scope:** the choice between reusing the M5-α typed-envelope abstraction (`extracted_reasoning_core.domains.DomainReasoningResult[PayloadT]`) for a new domain, and standing up a separate producer because the ontology diverges too far. This document is the rubric; the implementation steps for the chosen path live in `docs/reasoning_provider_port_migration.md`.

---

## When to consult this matrix

You're starting a new product or feature that wants to "reason over X" — call transcripts, internal company data, contract clauses, support tickets, anything where the system needs to:

1. Take a subject (a transcript, a company record, a ticket).
2. Compute structured reasoning (confidence, supporting signals, falsification conditions, …) over evidence.
3. Surface the reasoning in an MCP / API / UI overlay.

Before writing code, walk through the four decision dimensions below to land on a path. Get the answer wrong and you either (a) rebuild things that already work, or (b) wedge a domain into a producer that doesn't fit and pay for it in maintenance forever.

---

## Decision dimensions

The four dimensions that matter:

### 1. Ontology

**Question:** does your domain's "what we conclude" map onto an existing producer's conclusion shape?

- **Vendor pressure** concludes: which vendor is at risk, what wedge is driving it, what the displacement narrative looks like, what proof points support it.
- **Call transcript** concludes: what topics surfaced, sentiment, action items, who was on the call.
- **Company entity** concludes: which facts are reconciled across sources, where coverage gaps are, what the canonical record should be.
- **Support ticket** concludes: severity, escalation path, related tickets, root-cause hypotheses.

These are genuinely different ontologies. A vendor-pressure producer cannot output "topics + sentiment" without violating its own contract. Conversely, a transcript producer cannot meaningfully output "wedge + displacement_narrative".

**Compatible:** ontology overlaps ≥ ~70% with an existing domain payload.
**Incompatible:** ontology centers on different concepts. Build a new producer + payload.

### 2. Evidence semantics

**Question:** what counts as "evidence" in your domain, and where does it live?

- **Vendor pressure** evidence: review atoms (`b2b_review_evidence_atoms`), pricing mentions, exec-change signals — discrete witnessed claims with metric IDs and witness IDs.
- **Call transcript** evidence: spans of text within the transcript itself, with timestamps + speaker IDs.
- **Company entity** evidence: rows from CRM / contract / contacts tables, with source-system IDs.

These are all "evidence with lineage", but the **shape** of the evidence atom differs. The existing `EvidenceItem` (`source_type`, `source_id`, `text`, `metrics`, `metadata`) in `extracted_reasoning_core.types` covers them all generically — that's a deliberate design choice from PR-1.

**Compatible:** evidence fits the universal `EvidenceItem` shape, even if the `source_type` values are new.
**Incompatible:** evidence requires a structurally different shape (e.g. graph edges instead of items). Build a new core, not just a new producer.

### 3. Output contract (consumer side)

**Question:** does the consumer want flat overlay fields (the M5-α `Consumer.to_summary_fields` / `to_detail_fields` pattern), or a different output shape?

- MCP signal overlays, REST API responses, dashboard cards: flat overlay dict. ✓ Existing pattern works.
- Custom UI rendering with nested objects, conditional sections: still works — the consumer projection is just a method, not a wire shape.
- A producer that needs to feed *into another producer* (multi-stage reasoning): different role; that's the `CampaignReasoningContext` shape (data-input bundle), see PR #235's role-distinction doc.

**Compatible:** consumer surface is a flat overlay or projects to one.
**Incompatible:** consumer needs the producer's output as input to *another reasoning stage*. That's a producer-port composition, not a consumer-port projection.

### 4. Lineage requirements

**Question:** does your domain need `metric_ids` / `witness_ids` provenance threaded through to consumers?

The M5-α envelope carries `reference_ids: ReferenceIds(metric_ids, witness_ids)` for any domain that wants it. Domains that don't (purely textual conclusions, no source-tracing requirement) leave them empty.

**Compatible:** lineage either fits the two-bucket `metric_ids` / `witness_ids` model or is genuinely absent.
**Incompatible:** lineage requires a different model (e.g. a chain-of-thought trace with intermediate provenance per step). That's an envelope extension, not a domain choice — coordinate with platform.

---

## Compatibility matrix (known / likely domains)

| Domain | Status | Ontology | Evidence | Consumer | Lineage | Recommended path |
|---|---|---|---|---|---|---|
| `vendor_pressure` | ✅ shipped (M5-β) | vendor-displacement / wedge | `b2b_review_evidence_atoms` (review atoms, metric ledger, witness pack) | `signals.py` MCP overlay (flat dict) | `metric_ids` / `witness_ids` from `SynthesisView.reference_ids` | **Reuse**. `vendor_pressure_result_from_synthesis_view` already wraps the producer. |
| `call_transcript` | ✅ shipped (M5-γ) | topics / sentiment / action_items | per-transcript span (no formal evidence atoms yet) | transcript-detail overlay (flat dict) | optional; supported via nested `reference_ids` | **Reuse**. M5-γ ships the proof-of-life. A real producer reads the transcript and calls `call_transcript_result_from_entry`. |
| `competitive_intel` (cross-vendor) | partial | cross-vendor-edge / displacement-direction | shared with `vendor_pressure` (review atoms) | battle-card / vendor-briefing renderers | shared with `vendor_pressure` | **Reuse**. Same producer (`b2b_reasoning_synthesis`); consumer-side decoupling is the M5 / phase-3 work that's on the competitive-intel STATUS roadmap. |
| `content_ops` (campaign generation) | shipped (#189) but **not** typed-envelope | campaign-pain / wedge / proof-points | host-provided (`CampaignReasoningProviderPort` data-input bundle) | campaign generator's prompt builder | implicit in the bundle | **Reuse the data-input port**, not the typed envelope. Different role — see PR #235 role-distinction doc. |
| `company_entity` | not yet shipped | reconciled facts / coverage gaps / source authority | rows across CRM / contracts / contacts tables, with source-system IDs | company-detail UI overlay | per-fact `metric_ids` (the source-row IDs) and `witness_ids` (corroborating sources) | **Reuse** the typed envelope; build a domain payload that carries `reconciled_facts: tuple[FactDecision, ...]` / `coverage_gaps`. ~250-line additive PR mirroring `call_transcript`. |
| `support_ticket` | not yet shipped | severity / escalation / related-tickets / root-cause-hypotheses | ticket history rows + linked customer-success records | ticket-detail UI overlay or alerting card | optional | **Reuse**. Same shape as `call_transcript` — a payload with `severity`, `escalation_path`, `related_ticket_ids`, `root_cause_hypotheses`. |
| `contract_clause` | not yet shipped | risk_flags / counterparty_obligations / negotiated_deviations | contract paragraphs + clause references | redline / negotiation UI | per-clause `witness_ids` | **Reuse**. New payload with `clause_id`, `risk_flags`, `obligations`, `deviations_from_template`. |
| (hypothetical) **graph reasoning** over knowledge-graph triples | — | structured edges with reasoning at each hop | graph triples (different shape from `EvidenceItem`) | graph-traversal UI | chain-of-thought across hops | **Rebuild core**. Doesn't fit `EvidenceItem` or `DomainReasoningResult` cleanly. Coordinate with platform on whether the reasoning core itself needs a new envelope variant. |

---

## Walk-through: deciding a new domain in 5 minutes

Use this checklist when scoping a new use case. Most rows answer "yes" → reuse path, ~200-line additive PR mirroring `call_transcript`.

```
[ ] Subject has a stable id and a domain identifier             (yes -> ReasoningSubject works)
[ ] Conclusions fit a small fixed set of fields (≤ ~10)         (yes -> DomainPayload works)
[ ] Evidence shape fits {source_type, source_id, text,          (yes -> existing EvidenceItem works)
    metrics, metadata}
[ ] Consumer surface is flat dict or projects to one            (yes -> ReasoningConsumerPort works)
[ ] Lineage fits metric_ids / witness_ids buckets               (yes -> ReferenceIds works)

If 4-5 yes  -> Reuse. ~200-line additive PR, no core changes. Pattern: `call_transcript`.
If 2-3 yes  -> Reuse with caveats. New domain payload, possibly new universal-field semantics
                (talk to platform about whether to extend ReasoningResult vs adding to payload).
If 0-1 yes  -> Rebuild. The domain wants a different reasoning shape; consult the M1
                (`docs/reasoning_interface_contract.md`) to decide whether that warrants a new
                envelope variant in extracted_reasoning_core or a separate package altogether.
```

---

## Reuse path: what to ship

For domains that pass the "reuse" checklist, the additive PR has five pieces (~200 lines total):

1. `atlas_brain/reasoning/<domain>.py` (new) — `Subject` + `Payload` + `<domain>_result_from_entry` + `<DomainName>Consumer` + `register_domain(...)` at module import. Mirror `call_transcript.py` line-for-line.
2. `tests/test_atlas_reasoning_<domain>.py` (new) — Subject/Payload protocol checks, registration, sparse + full builder, consumer projections, cross-domain isolation test.
3. `scripts/run_extracted_pipeline_checks.sh` — add the test file to the pytest matrix.
4. `.github/workflows/extracted_pipeline_checks.yml` — add `atlas_brain/reasoning/<domain>.py` and `tests/test_atlas_reasoning_<domain>.py` to `paths:` filters.
5. (Optional) `atlas_brain/autonomous/tasks/<domain>_producer.py` — the actual producer that fetches subjects and calls `<domain>_result_from_entry`. This is the *use* of the abstraction, not the abstraction itself; ship it whenever the consuming product surface is ready.

For implementation step-by-step, see `docs/reasoning_provider_port_migration.md`.

---

## Rebuild path: what to ship

Rare. If you actually need to rebuild:

1. Document why the existing envelope doesn't fit (which dimension fails — usually evidence shape or output-contract role).
2. Coordinate with platform on whether the core needs a new envelope variant or a separate package. The M1 contract doc (`docs/reasoning_interface_contract.md`) is the authority.
3. Ship the new core first (additive — keep the existing `DomainReasoningResult` working), then build your producer on top.

Rebuild is an architectural escalation, not a product decision. If you find yourself reaching for it without one of the four dimensions clearly failing, walk back through the matrix.

---

## Anti-patterns

These come up enough that they're worth naming:

- **Wedging a transcript into vendor_pressure.** "It's all just text + signals". Wrong — the vendor-pressure producer is bound to vendor synthesis tables and its output keys (`wedge`, `displacement_narrative`) have hard semantics. Use a `call_transcript` producer.
- **Calling a "data-input provider" a "reasoning producer".** `CampaignReasoningProviderPort` returns a `CampaignReasoningContext` (input bundle). `ReasoningProducerPort` returns a `DomainReasoningResult` (typed envelope). Different roles — see PR #235.
- **Adding fields to `DomainReasoningResult` for a single domain.** Use `domain_payload`. The envelope's universal fields are universal precisely because every domain populates them.
- **Skipping lineage because "we don't have IDs yet".** You will. Wire `reference_ids` through from day one even if the values are empty tuples; future you will thank past you when audit/compliance comes asking.

---

## Where this doc fits

- **M1** — `docs/reasoning_interface_contract.md` — the canonical envelope contract.
- **M5-α** — `extracted_reasoning_core.domains` — the typed-envelope abstraction.
- **M5-β/γ** — `atlas_brain/reasoning/{vendor_pressure, call_transcript}.py` — two specializations.
- **M6 (this doc)** — `docs/hybrid_reasoning_compatibility_matrix.md` — when to reuse vs rebuild.
- **Implementation steps** — `docs/reasoning_provider_port_migration.md` (#198) — host migration to the provider port.
- **PR scope guard** — `docs/hybrid_scope_guard.md` (#195) — whether your slice is in-program.
- **PR body template** — `docs/hybrid_pr_body_template.md` (#198) — what to include.
