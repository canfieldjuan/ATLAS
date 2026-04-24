## Phrase Schema Cleanup Investigation (Deferred)

Date: 2026-04-22

### Decision

Defer schema cleanup of the six legacy phrase arrays until after the
Phase 1-4 semantic-tagging rollout is proven.

Current rollout path remains:

- keep legacy arrays as `list[str]`
- add `phrase_metadata` in parallel
- migrate new consumers to helpers over `phrase_metadata`
- revisit full schema cleanup only after adoption telemetry and consumer
  inventory are in hand

### Why Cleanup Is Deferred

The current codebase still assumes string semantics for the legacy fields:

- `specific_complaints`
- `pricing_phrases`
- `feature_gaps`
- `quotable_phrases`
- `recommendation_language`
- `positive_aspects`

Changing those arrays to contain dicts now would create broad compatibility
risk and, in some cases, silent corruption instead of loud failures.

Concrete examples confirmed in current code:

- [`_b2b_witnesses.py`](/home/juan-canfield/Desktop/Atlas/atlas_brain/autonomous/tasks/_b2b_witnesses.py)
  - `_normalize_list(...)` at line 171 coerces each element through
    `str(value).strip()`
  - if a phrase entry became a dict, the code would quietly turn it into a
    Python repr string
  - `price_text = " ".join(_normalize_list(result.get("pricing_phrases")))`
    at line 499 assumes joinable strings

- [`b2b_enrichment.py`](/home/juan-canfield/Desktop/Atlas/atlas_brain/autonomous/tasks/b2b_enrichment.py)
  - multiple `_normalize_text_list(...)` and direct list reads still assume
    string arrays, including pricing and complaint derivation paths
  - representative sites:
    - lines 981-982
    - line 1909
    - lines 4629-4630

### Blast Radius

The repo already contains a generated enrichment-field inventory:

- [`B2B_FIELD_ACCESS_INVENTORY_2026-04-03.md`](/home/juan-canfield/Desktop/Atlas/docs/B2B_FIELD_ACCESS_INVENTORY_2026-04-03.md)

Key numbers from that inventory:

- total files with enrichment reads: `26`
- total read sites: `633`
- non-exempt ad-hoc read sites: `238`

Legacy phrase-field read frequency from that inventory:

- `quotable_phrases`: `17`
- `feature_gaps`: `9`
- `specific_complaints`: `8`
- `pricing_phrases`: `6`
- `positive_aspects`: `3`
- `recommendation_language`: `2`

The 26 files with enrichment reads are concentrated in production code, not
just docs/tests:

- tasks: `10`
- api: `6`
- scripts: `7`
- services: `2`
- mcp: `1`

High-risk production modules in the current inventory include:

- [`_b2b_shared.py`](/home/juan-canfield/Desktop/Atlas/atlas_brain/autonomous/tasks/_b2b_shared.py)
- [`b2b_dashboard.py`](/home/juan-canfield/Desktop/Atlas/atlas_brain/api/b2b_dashboard.py)
- [`b2b_tenant_dashboard.py`](/home/juan-canfield/Desktop/Atlas/atlas_brain/api/b2b_tenant_dashboard.py)
- [`b2b_churn_intelligence.py`](/home/juan-canfield/Desktop/Atlas/atlas_brain/autonomous/tasks/b2b_churn_intelligence.py)
- [`b2b_product_profiles.py`](/home/juan-canfield/Desktop/Atlas/atlas_brain/autonomous/tasks/b2b_product_profiles.py)
- [`b2b_blog_post_generation.py`](/home/juan-canfield/Desktop/Atlas/atlas_brain/autonomous/tasks/b2b_blog_post_generation.py)

### Why Option D Is The Right Near-Term Contract

Using a parallel `phrase_metadata` field:

- preserves all existing `list[str]` consumers
- avoids 30-plus file churn during the semantic-tagging rollout
- gives new code a stable place to read subject / polarity / role / verbatim
- lets Phase 2/3/4 adopt the new contract via helper functions instead of
  ad-hoc enrichment parsing

The small storage duplication in `phrase_metadata[].text` is acceptable at
review scale and much cheaper than a broad schema migration now.

### What A Later Cleanup Would Need

Schema cleanup should be treated as a separate migration track after the
semantic-tagging rollout proves useful.

Minimum prerequisites:

1. `phrase_metadata` is live and used by new semantic consumers.
2. Witness quality improvements are demonstrated on the validation cohort.
3. Central helper functions exist for reading phrase metadata.
4. Telemetry or inventory confirms which legacy arrays are still materially
   consumed.
5. A consumer migration plan exists before introducing `str | dict` or
   dict-only arrays.

### Recommended Later Cleanup Sequence

1. Re-run and refresh the field access inventory.
2. Classify consumers:
   - must stay legacy-compatible
   - can migrate to `phrase_metadata`
   - should stop reading phrase arrays entirely
3. Introduce central phrase-access helpers and ban new ad-hoc array reads.
4. Migrate consumers in batches.
5. Add telemetry for remaining legacy reads.
6. Only then consider:
   - `str | dict` transitional arrays, or
   - a fully normalized dict-only phrase schema

### Replacement Paths To Track

The current phrase-tagging and grounding work adds replacement
infrastructure. It does not make the old paths dead yet.

| Area | New path | Later deprecation target | Current status |
| --- | --- | --- | --- |
| Phrase semantics | `phrase_metadata` read through `_b2b_phrase_metadata.py` helpers | Ad-hoc semantic interpretation of the six legacy phrase arrays | Not dead; keep legacy reads until consumers migrate |
| Enrichment pain classification | `subject` / `polarity` / `role` / `category_hint` on phrase metadata | `b2b_enrichment._PAIN_KEYWORDS_RAW`, `_PAIN_PATTERNS`, `_pain_scores`, `_derive_pain_categories` | Future fallback removal after v4 coverage and benchmark validation |
| Witness pain classification | Phrase metadata tags used for witness eligibility and pain category | `_b2b_witnesses._PAIN_KEYWORDS_BY_CATEGORY` and `_classify_complaint_pain` | Future removal after Phase 2-4 gates are trusted |
| Quote grounding | `_b2b_grounding.check_phrase_grounded(...)` and persisted grounding status | Raw lowercase substring checks in witness excerpt and evidence APIs | Future removal after grounding backfill and active-row coverage |
| Witness span identity | `source_span_id` and grounded phrase metadata | Fuzzy text-overlap matching against `evidence_spans` | Future removal after all active witnesses have deterministic span identity |
| UI highlighting | API-provided `highlight_start`, `highlight_end`, and `highlight_source` | Frontend substring-search fallback in `EvidenceDrawer.tsx` | Future removal after API contract is complete for all served witnesses |

### Deletion Gates

Do not delete the legacy paths above until these gates are true:

1. v4 enrichment coverage is high enough for the active vendor cohort.
2. Phase 2-4 semantic gates beat the Phase 0 benchmark on precision without
   unacceptable recall loss.
3. Latest-snapshot witness packets no longer depend on inferred or fuzzy
   excerpt matching for quote-grade display.
4. Active witness rows have `source_span_id` and grounding status populated.
5. Consumer inventory shows direct semantic reads of legacy phrase arrays have
   been migrated or explicitly retained as compatibility fallbacks.
6. The UI and APIs consume the same grounding/highlight contract, with no
   product surface depending on client-side quote discovery.

### Not Safe To Remove Yet

The following remain active compatibility paths:

- the six legacy phrase arrays
- `evidence_spans`
- enrichment keyword pain scoring
- witness keyword pain classification
- API/UI substring fallback highlighting
- v3 legacy enrichment path

They are serving old rows, unbackfilled data, and production consumers that
have not moved to `phrase_metadata`.

### Non-Goals For The Current Rollout

The current semantic-tagging implementation should not:

- place dicts into the original six arrays
- attempt a 30-plus file consumer migration during Phase 1
- mix schema cleanup with signal-quality hardening

### Bottom Line

Schema cleanup is worth revisiting later, but it is not the right work now.
The current repo state supports additive `phrase_metadata`; it does not
support safely replacing `list[str]` phrase arrays without broad migration
and material risk.
