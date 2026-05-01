# B2B Enrichment Execution Notes

**Created**: 2026-04-29
**Status**: Active
**Owner**: Codex + Juan

---

## Tier 2 Single-Call Fallback

### Verified behavior

The current `single_call_fallback` path for Tier 2 is **not** a semantic repair
pass and is **not** a "Sonnet missed it, let Haiku repair it" step.

It is a **backend fallback** for the same logical Tier 2 stage.

That means:

- same stage id: `b2b_enrichment.tier2`
- same prompt family
- same stage-ledger identity rules
- same output contract
- different execution backend when Anthropic batch is unavailable

### When it runs

The fallback only runs when:

1. Tier 1 indicates extraction gaps via `_tier1_has_extraction_gaps(...)`
2. Tier 2 batch execution is unavailable in `_enrich_rows(...)`

So the fallback exists to keep the pipeline moving when the batch executor
cannot run Tier 2, not to add a second-opinion repair step.

### Model-selection chain

Tier 2 OpenRouter / Anthropic path resolves model in this order:

1. `cfg.enrichment_tier2_openrouter_model`
2. `cfg.enrichment_openrouter_model`
3. `"anthropic/claude-haiku-4-5"`

Tier 2 local direct path resolves model in this order:

1. `cfg.enrichment_tier2_model`
2. `cfg.enrichment_tier1_model`

Implication:

- if production config points Tier 2 at Sonnet, the fallback uses Sonnet
- if production config points Tier 2 at Haiku, the fallback uses Haiku
- Haiku is the hardcoded final default only when Tier 2 OpenRouter config is empty

### Architectural conclusion

The current fallback name is slightly misleading because it sounds like
"semantic fallback" or "repair pass". It is not.

It is better understood as:

- **Tier 2 stage**
- with multiple executors:
  - direct
  - anthropic batch
  - single-call backend fallback

### Decision for the larger refactor

Do **not** overload the current Tier 2 fallback path with repair semantics.

If the product should have a true repair/audit pass, that should be a
**separate stage** after Tier 2, with its own:

- stage id
- prompt
- model policy
- ledger row
- success / defer / execute controller

The current centralized-work refactor should continue treating Tier 2 as one
logical stage with multiple execution backends.

---

## Next Architecture Slice

Move from centralized stage-state resolution to a centralized
**stage execution controller** that owns:

- stage reuse decision
- submitted/defer decision
- batch artifact reconciliation
- batch submission decision
- direct execution decision

The controller should replace the remaining split authority between:

- stage ledger state
- batch artifact reconciliation
- local orchestration branches in `b2b_enrichment.py`
