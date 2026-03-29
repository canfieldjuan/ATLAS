# Atlas Plan

This file tracks upcoming work. Progress, decisions, and retrospectives remain in `BUILD.md`.

## Active Goals (Phase 3 – AI Model Integration)

1. **Stabilize inference**
   - Capture timing/VRAM stats so we know current headroom.
   - Persist helpful debug logs to `logs/` for future troubleshooting.

## Near-Term Backlog

- Improve STT pipeline with faster-whisper and feed transcription results back through the intent router.
- Define response schemas for text/audio/vision requests so API consumers get consistent shape + metadata.
- Add CI-friendly smoke tests (e.g., start FastAPI, hit `/ping`) to catch regressions quickly.
- Productize intelligence reports (see `docs/INTELLIGENCE_REPORTS.md`).
- Unify the B2B reasoning pipeline around contracts-first synthesis so reports, blogs, battle cards, and campaigns consume one reasoning source of truth.

## Edit Plan: B2B Reasoning Pipeline Redesign

Objective: replace the current hybrid legacy-plus-synthesis reasoning stack with one canonical contracts-first reasoning layer, while preserving deterministic evidence assembly and improving downstream report/blog/campaign quality.

### Current-State Summary

- Enrichment is deterministic after extraction: review-level `urgency_score`, `pain_category`, `would_recommend`, and `sentiment_trajectory.direction` are derived in `atlas_brain/autonomous/tasks/b2b_enrichment.py`.
- There are two post-enrichment reasoning paths today:
  - Legacy stratified reasoning in `atlas_brain/autonomous/tasks/b2b_churn_intelligence.py` writing archetype-style output into `b2b_churn_signals`.
  - New contracts-first synthesis in `atlas_brain/autonomous/tasks/b2b_reasoning_synthesis.py` writing `reasoning_contracts` into `b2b_reasoning_synthesis`.
- Downstream consumers are mixed:
  - Battle cards already consume contracts-first reasonably well.
  - Reports still build from deterministic facts plus legacy `reasoning_lookup`, then attach synthesis later.
  - Blogs use deterministic topic selection first, then inject synthesis for blueprint/rendering.
  - Campaigns still read `archetype_context` from `b2b_churn_signals`.

### Target Architecture

```text
enrichment
  -> deterministic canonical evidence layers
  -> reasoning-ready packet
  -> single synthesis model
  -> canonical reasoning contracts
  -> consumer adapters
     reports / blogs / battle cards / campaigns
  -> optional consumer-specific render LLMs
```

Design rules:

- Keep deterministic evidence assembly.
- Make `b2b_reasoning_synthesis` the primary reasoning source.
- Treat legacy archetype reasoning as fallback only.
- Separate shared analytical reasoning from consumer-specific render language.
- Expose scope, contradiction, confidence, and retention signals explicitly rather than forcing the model to infer them from compressed top-N evidence.

### Phase 1: Canonical Reasoning Source

Goal: make synthesis the default reasoning layer everywhere.

Files:

- `atlas_brain/autonomous/tasks/_b2b_synthesis_reader.py`
- `atlas_brain/autonomous/tasks/b2b_campaign_generation.py`
- `atlas_brain/autonomous/tasks/b2b_churn_reports.py`
- `atlas_brain/autonomous/tasks/b2b_vendor_briefing.py`

Implementation:

- Add `load_best_reasoning_view(pool, vendor_name, *, as_of, analysis_window_days)` in `_b2b_synthesis_reader.py`.
  - Prefer latest `b2b_reasoning_synthesis`.
  - Fall back to legacy `b2b_churn_signals`.
  - Return a normalized `SynthesisView`-like object.
- Add `legacy_reasoning_to_contracts(...)` in `_b2b_synthesis_reader.py`.
  - Convert legacy `archetype`, `executive_summary`, `key_signals`, `falsification_conditions`, `uncertainty_sources` into minimal fallback contracts.
- Extend `_CONSUMER_REQUIRED_CONTRACTS` to cover campaign and blog-reranker consumers.
- Replace direct `b2b_churn_signals` reads in campaigns and other downstream consumers with the shared loader.

Acceptance criteria:

- If fresh synthesis exists, reports and campaigns do not read `archetype` directly from `b2b_churn_signals`.
- One vendor yields one effective wedge/core reasoning object across battle cards, reports, blogs, and campaigns.

### Phase 2: Reasoning Packet Upgrade

Goal: improve what the model thinks over without removing deterministic prep.

Files:

- `atlas_brain/autonomous/tasks/_b2b_pool_compression.py`
- `atlas_brain/autonomous/tasks/_b2b_shared.py`

Implementation:

- Extend `_b2b_pool_compression.py` with:
  - `_build_metric_ledger(layers)`
  - `_build_contradiction_rows(layers)`
  - `_build_minority_signals(layers)`
  - `_build_coverage_gaps(layers)`
  - `_build_retention_proof(layers)`
- Extend `CompressedPacket.to_llm_payload()` to emit:
  - `metric_ledger`
  - `contradiction_rows`
  - `minority_signals`
  - `coverage_gaps`
  - `retention_proof`
- Keep existing scored pools and source IDs.
- Do not remove current caps yet; first widen the packet with explicit missing/tension signals.

Payload targets:

```json
{
  "metric_ledger": [
    {
      "label": "pricing_mentions",
      "value": 16,
      "scope": "pricing_mentions",
      "time_window_days": 90,
      "allowed_surfaces": ["report", "battle_card", "blog"],
      "_sid": "vault:metric:pricing_mentions"
    }
  ],
  "contradiction_rows": [
    {
      "dimension": "support",
      "segment_a": "smb",
      "segment_b": "enterprise",
      "statement_a": "positive",
      "statement_b": "negative",
      "_sid": "segment:contradiction:support"
    }
  ],
  "minority_signals": [
    {
      "label": "security_blocker",
      "urgency": 8.7,
      "count": 3,
      "reason": "rare_but_severe",
      "_sid": "vault:minority:security"
    }
  ],
  "coverage_gaps": [
    {
      "type": "thin_segment_sample",
      "area": "enterprise_finance",
      "sample_size": 3
    }
  ],
  "retention_proof": [
    {
      "area": "integrations",
      "strength": "ecosystem breadth reduces churn despite frustration",
      "_sid": "vault:strength:integrations"
    }
  ]
}
```

Acceptance criteria:

- The model packet explicitly labels numeric scope.
- Contradictions and thin-evidence areas are represented explicitly instead of being silently dropped.

### Phase 3: Contract Expansion

Goal: make the shared synthesis reusable for reports, blogs, battle cards, and campaigns.

Files:

- `atlas_brain/autonomous/tasks/_b2b_reasoning_contracts.py`
- `atlas_brain/autonomous/tasks/_b2b_synthesis_reader.py`
- `atlas_brain/autonomous/tasks/_b2b_synthesis_validation.py`

Implementation:

- Add canonicalizers/builders for:
  - `vendor_core_reasoning.why_they_stay`
  - `vendor_core_reasoning.confidence_posture`
  - `displacement_reasoning.switch_triggers`
  - `evidence_governance.metric_ledger`
  - `evidence_governance.contradictions`
  - `evidence_governance.coverage_gaps`
- Preserve `build_persistable_synthesis(...)` as the persistence choke point.
- Add validation rules:
  - reject unsupported numbers not present in `metric_ledger`
  - require contradiction-aware hedging when contradiction rows exist
  - require `why_they_stay` when retention proof exists
  - require `confidence_posture.limits` when coverage gaps are non-empty

Suggested contract expansion:

```json
{
  "reasoning_contracts": {
    "vendor_core_reasoning": {
      "why_they_stay": {
        "summary": "...",
        "strengths": [
          {
            "area": "integrations",
            "evidence": "...",
            "neutralization": "..."
          }
        ]
      },
      "confidence_posture": {
        "overall": "medium",
        "limits": ["thin enterprise sample", "weak displacement density"]
      }
    },
    "displacement_reasoning": {
      "switch_triggers": []
    },
    "evidence_governance": {
      "metric_ledger": [],
      "contradictions": [],
      "coverage_gaps": []
    }
  }
}
```

Acceptance criteria:

- Shared reasoning includes both churn pressure and retention/inertia.
- Downstream consumers can read uncertainty/governance directly instead of inferring it.

### Phase 4: Prompt Separation

Goal: make shared synthesis analytical and consumer-neutral.

Files:

- `atlas_brain/reasoning/single_pass_prompts/reasoning_synthesis.py` (new)
- `atlas_brain/reasoning/single_pass_prompts/battle_card_reasoning.py`
- `atlas_brain/autonomous/tasks/b2b_reasoning_synthesis.py`

Implementation:

- Create a new neutral prompt module for shared reasoning synthesis.
- Stop using battle-card-shaped prompt language as the shared reasoning source.
- Keep battle card prompt only for battle-card rendering.
- Update `b2b_reasoning_synthesis.py` to use the new prompt while keeping current retry/validation flow.

Prompt rules:

- No sales language.
- No blog/editorial language.
- Every numeric claim must tie back to `metric_ledger`.
- Explicitly surface contradictions and coverage gaps.
- Output contracts only.

Acceptance criteria:

- Shared synthesis JSON is analytical, not GTM-styled.
- Render prompts translate contracts; they do not invent new reasoning.

### Phase 5: Campaign Migration

Goal: replace legacy archetype-driven campaign context with synthesis-driven reasoning context.

Files:

- `atlas_brain/autonomous/tasks/b2b_campaign_generation.py`
- `atlas_brain/mcp/b2b/write_intelligence.py`

Implementation:

- Replace `archetype_context` with `reasoning_context`.
- Pull campaign context from:
  - `vendor_core_reasoning.causal_narrative`
  - `vendor_core_reasoning.why_they_stay`
  - `vendor_core_reasoning.timing_intelligence`
  - `displacement_reasoning.switch_triggers`
  - `account_reasoning`
  - `evidence_governance.metric_ledger`
- Ensure blog suggestions and campaign copy can explain both pain and inertia.

Acceptance criteria:

- Campaigns no longer depend on `b2b_churn_signals.archetype` when synthesis exists.
- Generated campaign context can explain why buyers are unhappy and why they still have not switched.

### Phase 6: Report Migration

Goal: make reports contracts-first instead of deterministic-first with synthesis attached later.

Files:

- `atlas_brain/autonomous/tasks/b2b_churn_reports.py`
- `atlas_brain/autonomous/tasks/_b2b_shared.py`

Implementation:

- Refactor report assembly so deterministic facts and materialized contracts are composed together before final entry shaping.
- Reduce reliance on legacy `reasoning_lookup`.
- Add helpers to surface:
  - `coverage_gaps`
  - `data_stale`
  - `confidence_posture`
  - `metric_ledger`
- Remove mixed narratives where scorecards use legacy `reasoning_summary` alongside synthesis wedge/contracts.

Acceptance criteria:

- Reports do not present conflicting legacy and synthesis narratives for the same vendor.
- Report entries can expose both confidence limits and evidence freshness.

### Phase 7: Blog Topic Reranker

Goal: make reasoning affect story selection, not only story framing.

Files:

- `atlas_brain/autonomous/tasks/b2b_blog_post_generation.py`
- `atlas_brain/autonomous/tasks/_blog_matching.py` if reused

Implementation:

- Keep current deterministic candidate generation.
- Add `rerank_topic_candidates_with_reasoning(pool, candidates, as_of, window_days)` after `_select_topic()` candidate scoring and before final selection.
- Feed reranker:
  - synthesis wedge
  - timing intelligence
  - account reasoning
  - category reasoning
  - coverage gaps
- In `_load_pool_layers_for_blog(...)`, expose:
  - `evidence_governance`
  - `why_they_stay`
  - `confidence_posture`

Acceptance criteria:

- A reasoning-backed topic can outrank a raw-volume topic.
- Thin-evidence topics are deprioritized even when deterministic scores are high.

### Phase 8: Battle Card Governance Enforcement

Goal: keep battle cards contracts-first while improving scope and calibration discipline.

Files:

- `atlas_brain/autonomous/tasks/b2b_battle_cards.py`
- `atlas_brain/autonomous/tasks/_b2b_shared.py`

Implementation:

- Ensure battle-card render payload explicitly includes `why_they_stay` and `evidence_governance.metric_ledger` if not already present via `reasoning_contracts`.
- Extend validation to use governance fields for:
  - scoped-number checks
  - overreach suppression when contradictions exist
  - low-confidence phrasing when coverage gaps exist

Acceptance criteria:

- Battle cards cannot casually mix scopes or overstate conclusions against contradiction/gap signals.
- Reps get both churn pressure and incumbent strength context.

### Tests

Files to extend:

- `tests/test_b2b_reasoning_synthesis.py`
- `tests/test_b2b_synthesis_reader.py`
- `tests/test_b2b_campaign_generation.py`
- `tests/test_b2b_blog_post_generation.py`
- `tests/test_b2b_churn_reports.py`
- `tests/test_b2b_battle_cards.py`

Required test cases:

- synthesis preferred over legacy when both exist
- fallback to legacy contracts when synthesis missing
- scoped-number rejection when a rendered number is not in `metric_ledger`
- contradiction rows force softer language / governance warnings
- `why_they_stay` reaches campaigns and battle cards
- blog reranker changes winning topic when reasoning context changes
- report + campaign + battle card share the same wedge/core reasoning for one vendor

### Recommended PR Split

1. Shared loader + legacy fallback normalization
2. Pool compression packet widening
3. Contract expansion + synthesis validation
4. Neutral synthesis prompt swap
5. Campaign migration
6. Report migration
7. Blog reranker
8. Battle-card governance follow-through

### Audit Checklist For The Follow-Up Session

- Verify no direct `b2b_churn_signals.archetype` reads remain in downstream consumers that should be synthesis-first.
- Verify `reasoning_contract_gaps` remains populated correctly for all consumer types.
- Verify metric scope survives from packet -> contracts -> consumer payload -> render validation.
- Verify the new governance fields are additive and backward compatible with older synthesis rows.
- Verify the blog pipeline uses reasoning in candidate ranking, not only in blueprint copy.
- Verify campaigns and reports produce aligned vendor narratives after migration.

## Future Considerations

- Terminal authentication + multi-tenant session management for remote clients.
- Model management (version pinning, hot reload hooks, telemetry).
- Observability stack (structured logs, metrics, tracing) once Atlas begins serving real terminals.
