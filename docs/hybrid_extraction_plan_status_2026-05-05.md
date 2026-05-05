# Hybrid Extraction Plan — Status & Hand-off (2026-05-05)

**Purpose**: snapshot of where the original plan in `docs/hybrid_extraction_implementation_plan.md` and `docs/hybrid_extraction_execution_board.md` actually stands today, and what's left to drive directly without further codex-cloud slicing.

This document is the source of truth. The execution-board "Progress ledger" section is now redundant and can be ignored in favor of this snapshot.

---

## What's done (M1–M3)

| Plan slice | Plan status | Reality |
|---|---|---|
| **PR-1** Shared reasoning interface spec | docs only | ✅ shipped via #178 (`reasoning_interface_contract.md`, `churn_reasoning_engine_map.md`, the implementation plan and execution board themselves) |
| **PR-2** Consumer adapter package | typed reader façade | ✅ shipped via #178/#187 (`atlas_brain/autonomous/tasks/_b2b_reasoning_consumer_adapter.py` + MCP overlay rewiring in `atlas_brain/mcp/b2b/signals.py` + adapter & overlay regression tests + sparse-entry guard via #184) |
| **PR-3** Host provider port for reasoning producer input | provider port + entrypoint wiring | ✅ shipped via #189 / #195 / #198 / #200 (`extracted_content_pipeline/services/reasoning_provider_port.py`, `load_reasoning_provider_port`, example/postgres CLI rewiring, compat-checks + scoped pytest + JSON report runner + markdown summary renderer) |

The on-main test `tests/test_extracted_campaign_reasoning_data.py::test_load_reasoning_provider_port_is_protocol_compatible` enforces the boundary (asserts both the concrete `FileCampaignReasoningContextProvider` and the runtime-checkable `CampaignReasoningProviderPort` Protocol).

## What's left (M4 + M5 + M6)

The three remaining milestones from the execution board, with **concrete file-level scope** so they can be driven directly.

---

### M4 / PR-4 — Shared substrate enforcement

**Plan as written**: "Audit and enforce all new reasoning paths use `extracted_llm_infrastructure` services. Add guardrails/checks to block direct atlas-core LLM service coupling in extracted products."

**Reality check**: The three extracted packages already use a **lazy-bridge pattern** (top-level imports are clean; `atlas_brain` is reached only via `try: from atlas_brain.X import …` or `_importlib.import_module("atlas_brain.X")` inside conditional blocks gated by standalone-mode env flags). Verified count of hard top-level `from atlas_brain` / `import atlas_brain` imports in `*.py`:

| Package | Hard imports |
|---|---:|
| `extracted_content_pipeline/` | 0 |
| `extracted_competitive_intelligence/` | 0 |
| `extracted_llm_infrastructure/` | 7 — all inside `try:`/`except ImportError:` fallbacks (see `skills/__init__.py:28`, `config.py:27`, `services/registry.py:23`, `services/base.py:22`, `services/protocols.py:21`, `services/tracing.py:7`, `storage/database.py:22`) |

So the actual M4 work is **regression prevention**, not removal. Two concrete pieces:

#### M4-a · Forbidden-import guard in `validate_extracted.sh`

`extracted/_shared/scripts/validate_extracted.sh` today only checks `manifest.json` mappings (byte-equality against `atlas_brain` sources for non-owned files). It does **not** scan for hard imports.

Add a pre-check that fails closed when a non-bridge `*.py` file under the extracted package contains a top-level `from atlas_brain` or `import atlas_brain` outside a `try:`/`except ImportError:` block. Acceptable patterns to allow:

- Inside `try:` blocks (lazy bridge fallback).
- Inside functions (deferred import for runtime resolution).
- In docstrings/comments (regex must check leading whitespace + lack of `#`/`"""`).

Owner files to touch:
- `extracted/_shared/scripts/validate_extracted.sh` — add the forbidden-import scan stanza.
- `extracted/_shared/scripts/forbid_hard_atlas_imports.py` (new) — Python helper that enumerates `*.py` under a package, parses with `ast`, and flags top-level `Import` / `ImportFrom` nodes whose target starts with `atlas_brain` and that are **not** inside a `Try` block. Exit nonzero on any hit.

Acceptance: running `bash scripts/validate_extracted_competitive_intelligence.sh` (and the other two) on current `main` is a no-op (zero hits). Adding a top-level `from atlas_brain.config import settings` to any extracted file makes the script fail.

#### M4-b · Bridge-block discipline for the 7 LLM-infra fallbacks

The 7 hard imports in `extracted_llm_infrastructure/` are all inside `try:` blocks — but the validator from M4-a will need to know which ones are legitimate. Two options:

1. **Allow all `try:`-guarded `from atlas_brain` imports unconditionally** (simplest; matches the existing pattern; lets the bridge keep working).
2. **Add an explicit `_atlas_bridge.py` allowlist file** under each extracted package and validate against it.

Recommend option 1 for first pass — the `try:` block itself is the contract. If we later want fine-grained control we can layer in option 2.

**Estimate**: 3-5 days as the plan said. Single PR.

---

### M5 / PR-5 — Competitive-intel phase-3 decoupling slice

**Plan as written**: "Remove one high-impact remaining phase-3 coupling path per PR (iterative). Start with deep-builder access behind explicit host adapter protocols."

**Reality check**: `extracted_competitive_intelligence/STATUS.md:73` already enumerates the exact Phase-3 backlog. Most of it is **non-reasoning** (PDF, write tools), so trim PR-5's scope to just the items that actually intersect with the reasoning hybrid-extraction goal:

#### M5-a · Rewire battle-card / vendor-briefing LLM call sites onto `extracted_llm_infrastructure`

Two file ranges, named explicitly in `STATUS.md`:

- `extracted_competitive_intelligence/autonomous/tasks/b2b_battle_cards.py:3140` — `call_llm_with_skill` / `get_pipeline_llm` already routes through `pipelines.llm`; exact-cache and Anthropic batch bridges already route through extracted LLM infra. **Remaining**: any direct `services.llm.*` reach-throughs at this call site that bypass `pipelines.llm`. Audit needed.
- `extracted_competitive_intelligence/autonomous/tasks/b2b_vendor_briefing.py:1199-1202` — `get_llm` call site. Replace with the same `pipelines.llm` access pattern used in battle-cards.

Acceptance: `grep -n "from .*services\.llm\." extracted_competitive_intelligence/autonomous/tasks/b2b_{battle_cards,vendor_briefing}.py` returns zero direct provider imports; only `pipelines.llm` access remains.

#### M5-b · Replace `_b2b_shared.py` cross-imports with explicit `Protocol` interfaces

`STATUS.md` calls out: `vendor_briefing.py:40-47` reads from `_b2b_shared` for vendor intelligence records.

`_b2b_shared.py` is currently a lazy-bridge to `atlas_brain.autonomous.tasks._b2b_shared`. Define a `VendorIntelligenceReader` `Protocol` in a new `extracted_competitive_intelligence/autonomous/tasks/_b2b_shared_ports.py`, port the four functions `vendor_briefing.py:40-47` actually uses to that protocol surface, and have `vendor_briefing.py` accept the reader via constructor injection (mirroring the pattern PR-C4c used in `atlas_brain/reasoning/port_adapters.py`).

#### M5-c · Promote `mcp/b2b/write_ports.py` deep-builder ports to actual Protocols (already partly done)

`extracted_competitive_intelligence/mcp/b2b/write_ports.py` already defines `ChallengerBriefBuilder` etc. as `Protocol` classes. The remaining gap (per STATUS.md) is providing **host adapters** for those builders — i.e., the atlas-side concrete implementations that satisfy the protocols and get injected at process startup.

This is the same pattern PR-C4c (#192) is doing for `EventSink`/`TraceSink` in `atlas_brain/reasoning/port_adapters.py`. Do the equivalent for `ChallengerBriefBuilder` — likely a new file `atlas_brain/mcp/b2b/competitive_write_adapters.py` — and wire it through whatever startup path Atlas already uses for the write-tools.

**Estimate**: 1-2 weeks as the plan said, but split into one PR per sub-slice (M5-a, M5-b, M5-c) — they can land independently.

---

### M6 / PR-6 — Migration runbook + compatibility matrix

**Plan as written**: two new docs (`docs/hybrid_reasoning_migration_runbook.md`, `docs/hybrid_reasoning_compatibility_matrix.md`).

**Reality check**: We already have `docs/reasoning_provider_port_migration.md` (#198) and `docs/hybrid_pr_body_template.md` (#198). Either rename the missing two to those, or fold them into a single runbook. PR-6's value is mostly checklist content — a 2-day doc PR.

Bare-minimum content for the compatibility matrix:

| Product | Reasoning ontology | Evidence semantics | Recommended path |
|---|---|---|---|
| Churn intelligence | vendor-pressure / displacement | `b2b_review_evidence_atoms` | reuse producer (`b2b_reasoning_synthesis`) |
| Content ops | campaign-pain / wedge | host-provided via `CampaignReasoningProviderPort` | reuse consumer contract; producer host-owned |
| Competitive intel | cross-vendor-edge | shared with churn | reuse producer; gate via `_b2b_cross_vendor_synthesis` |
| New product domain X | (TBD) | (TBD) | rebuild producer if ontology diverges |

---

## Followup-resolution log

PRs that close out the deferred items from M5-β / M5-γ:

- **Wire b2b regression tests into CI** — *resolved in PR #229
  (open, awaiting merge)*. Adds the two regression tests to
  `.github/workflows/extracted_competitive_intelligence_checks.yml`
  (the heavier CI tier that already does
  `pip install -r requirements.txt` and so already carries
  `apscheduler` / `asyncpg` / `mcp` / `torch` — no install-line
  expansion needed).
- **Lineage / freshness in the typed envelope** — *resolved in PR
  #232 (open, awaiting merge)*. Threads `view.reference_ids` /
  `view.as_of_date_iso` (property) /
  `view.confidence("causal_narrative")` through the synthesis-view
  wrapper into `DomainReasoningResult.reference_ids` / `as_of` /
  `confidence_label`. Wire shape preserved (consumer projections
  still don't surface the new fields; that's a follow-up).
- **Provider-port retyping (`CampaignReasoningProviderPort` →
  `ReasoningProducerPort[...]`)** — *resolved in this PR (open,
  awaiting merge) as not-applicable*.
  On closer inspection, `CampaignReasoningProviderPort` is a
  **data-input provider** for the campaign generator, not a reasoning
  compute port. It returns a `CampaignReasoningContext` (anchor
  examples, witness highlights, top theses, account signals, proof
  points -- a vendor-pressure-domain-shaped input bundle), whereas
  `ReasoningProducerPort` returns a `DomainReasoningResult[PayloadT]`
  (typed reasoning envelope). Different role, different return type.
  Forcing one to be a Protocol-level specialization of the other would
  break every implementer. Resolution: **document the role
  distinction in `extracted_content_pipeline/services/reasoning_provider_port.py`**
  so the next reader doesn't repeat the conflation. Future
  enrichment can let `CampaignReasoningContext` carry a typed
  envelope as it gains production-side use; that's enrichment-of-the-
  bundle, not retyping-of-the-port.
- **`VendorPressurePayload` enrichment** — deferred. Add
  displacement narrative / account-signals / proof-points fields when
  a producer-side caller needs them.
- **Second-domain proof-of-life** — *resolved* by M5-γ
  (`call_transcript`) — already merged.

---

## Lessons from the codex-cloud loop

This is for the record so we don't repeat the pattern:

- The same Codex task (`task_e_69f8f8c68d24832eb51e047d5e5a0225`) opened **eight separate PRs** (#178 merged, #184/#186 superseded, #191/#196/#197/#199/#202 closed-as-superseded) for what the plan estimated as PR-2 + PR-3 (≈10 days).
- Each new branch reopened essentially the same +1400-line diff — adapter + provider-port + 6 docs + tests + ≥4 scripts — based on a stale snapshot of `main`.
- Cumulative residual unique content carried forward (#195 / #198 / #200 / #204) was about **170 net lines** (2 new docs, 1 progress ledger, 4 scripts, 1 wrapper, 1 gitignore line, 4 small Copilot-fix patches).
- #202 in particular **reverted six fixes** that had already shipped (the test-file `NameError` from #189, the sparse-entry guard from #184, four #200 review fixes, the C4b CI trigger paths from #182). It was closed; the residual 14-line wrapper was carried forward in #204.

**Implication for M4/M5/M6**: drive these directly off this status doc rather than feeding the whole plan back into a Codex task. Each milestone here is small enough for one focused PR per slice.

---

## Quick reference

- **Plan PR (sealed)**: #178 — introduces `hybrid_extraction_implementation_plan.md`, `hybrid_extraction_execution_board.md`, `churn_reasoning_engine_map.md`, `reasoning_interface_contract.md`.
- **PR-2 / PR-3 landings**: #189 (provider-port + test), #195 (scope guard + postgres-CLI test), #198 (PR body template + migration guide + compat scripts), #200 (hybrid-checks wrapper + JSON report + summary renderer + four review fixes), #204 (one-shot wrapper).
- **Open in this workstream**: #192 (PR-C4c — port adapters; orthogonal but related; on hold).
- **Closed as superseded**: #184, #186, #191, #196, #197, #199, #202.
