# AI Content Ops Reasoning Handoff Contract

Date: 2026-05-03

This package treats reasoning as an input, not as an embedded subsystem.
AI Content Ops owns campaign/content generation, orchestration, prompt
contracts, delivery, sequencing, and customer-data adapters. It does not own
the long-running reasoning producer that pools evidence, builds graph state, or
decides multi-hop strategy.

## Decision

Reasoning generation is host-owned and reaches the product through the
`CampaignReasoningContextProvider` port in
`extracted_content_pipeline/campaign_ports.py`.

That keeps the package composable:

- A buyer can use the extracted reasoning product once it is available.
- A buyer can provide their own reasoning engine.
- Atlas can provide its existing synthesis/compression output through an
adapter.
- Lightweight content types can run with no reasoning provider.

The content package must not import `atlas_brain` reasoning producers,
`extracted_reasoning_core` internals, Neo4j clients, entity locks, event buses,
or pool-compression tasks directly.

## Public Port

```python
class CampaignReasoningContextProvider(Protocol):
    async def read_campaign_reasoning_context(
        self,
        *,
        scope: TenantScope,
        target_id: str,
        target_mode: str,
        opportunity: Mapping[str, Any],
    ) -> CampaignReasoningContext | Mapping[str, Any] | None:
        ...
```

`CampaignGenerationService` calls this provider once per normalized
opportunity. Returned mappings are normalized through
`normalize_campaign_reasoning_context(...)`, then stored on draft metadata and
included in the prompt-visible opportunity payload.

Provider failures are isolated per opportunity. A failed reasoning read skips
that opportunity and records an error; it does not stop the whole generation
run.

## Accepted Context Shape

Providers may return an already-built `CampaignReasoningContext` or a mapping
with any of these keys:

| Key | Purpose |
|---|---|
| `campaign_reasoning_context` | Preferred wrapper for normalized prompt material. |
| `reasoning_context` | Canonical reasoning fields such as `wedge`, `confidence`, `summary`, `why_now`, and product-specific signal fields. |
| `reasoning_anchor_examples` / `anchor_examples` | Labelled evidence rows for proof anchors. |
| `reasoning_witness_highlights` / `witness_highlights` | Quote/witness rows that can ground copy. |
| `reasoning_reference_ids` / `reference_ids` | Source ids for auditability. |
| `reasoning_top_theses` / `top_theses` | Top strategic theses for the opportunity. |
| `reasoning_account_signals` / `account_signals` | Account-level buying or churn signals. |
| `reasoning_timing_windows` / `timing_windows` | Why-now windows such as renewal, budget, or migration triggers. |
| `reasoning_proof_points` / `proof_points` | Compact metrics or facts the prompt can cite. |
| `reasoning_coverage_limits` / `coverage_limits` | Explicit caveats, missing evidence, or confidence limits. |
| `reasoning_scope_summary` / `scope_summary` | How much evidence the host considered. |
| `reasoning_delta_summary` / `delta_summary` | What changed since the prior reasoning snapshot. |

Unknown scalar/list/dict fields inside `campaign_reasoning_context`,
`reasoning_atom_context`, or `reasoning_context` are preserved as canonical
reasoning fields unless they are one of the structure keys above.

## Minimal Example

```python
class HostReasoningProvider:
    async def read_campaign_reasoning_context(
        self,
        *,
        scope,
        target_id,
        target_mode,
        opportunity,
    ):
        return {
            "reasoning_context": {
                "wedge": "renewal pressure",
                "confidence": "high",
                "summary": "Acme is reviewing vendors before renewal.",
                "key_signals": ["pricing_mentions", "renewal_window"],
            },
            "campaign_reasoning_context": {
                "proof_points": [
                    {"label": "pricing_mentions", "value": 12},
                ],
                "timing_windows": [
                    {"window_type": "renewal", "anchor": "Q3"},
                ],
                "account_signals": [
                    {"company": "Acme", "primary_pain": "pricing"},
                ],
                "coverage_limits": ["thin_account_signals"],
            },
        }
```

## File-Backed Example Adapter

`campaign_reasoning_data.FileCampaignReasoningContextProvider` is the reference
adapter for hosts that already have reasoning output as JSON. It accepts
context rows keyed by target id, company, email, or vendor, normalizes them into
`CampaignReasoningContext`, and keeps AI Content Ops independent from any
reasoning producer.

Use `campaign_reasoning_data.load_reasoning_provider_port(...)` when wiring this
adapter into host CLI/runtime entrypoints.

```bash
python scripts/run_extracted_campaign_generation_example.py \
  --reasoning-context extracted_content_pipeline/examples/campaign_reasoning_context.json
```

## Integration Modes

| Mode | Who produces reasoning? | Content package behavior |
|---|---|---|
| Atlas-hosted | Atlas synthesis/compression adapters. | Adapter returns the contract shape above. No direct Atlas imports in product code. |
| Extracted reasoning product | `extracted_reasoning_core` or a future reasoning-producer package. | Adapter converts reasoning output to `CampaignReasoningContext`. |
| Buyer-owned reasoning | Customer engine, warehouse job, agent workflow, or CRM scoring layer. | Customer adapter implements the provider port. |
| No reasoning | No provider configured. | Generator uses embedded opportunity fields only; quality is lower but standalone operation remains valid. |

## Product Boundaries

AI Content Ops may own:

- Customer-data normalization and source adapters.
- Prompt assembly and channel-specific generation.
- Sequence progression, suppression, audit, analytics, webhooks, and send
  orchestration.
- Prompt-visible reasoning context normalization.
- Metadata persistence of the context it consumed.

AI Content Ops must not own:

- Multi-hop reasoning graph traversal.
- Evidence-pool compression.
- Cross-vendor synthesis.
- Entity locks, event bus orchestration, or reasoning-agent state.
- Domain-specific reasoning producer internals from Atlas or the reasoning
  core product.

## Fit By Content Type

| Product direction | Reasoning provider required? | Notes |
|---|---|---|
| B2B email/campaign generation | Recommended. | Best outputs use account, witness, timing, and proof context through this contract. |
| Podcast repurposing | Usually no. | The episode transcript is the compressed context. |
| Long-form creative stories | Yes, but not B2B reasoning. | Needs a domain-specific planner/state tracker; use the same provider pattern, not the B2B synthesis stack. |

## Acceptance Criteria For Future Work

- New campaign/content flows use `CampaignReasoningContextProvider` or a
  parallel content-type-specific provider port.
- No product code imports `atlas_brain` or `extracted_reasoning_core`
  internals.
- Tests prove generation still works when a provider returns a mapping, a
  `CampaignReasoningContext`, `None`, or an exception.
- Docs for any new content type state whether reasoning is required,
  optional, or intentionally absent.
