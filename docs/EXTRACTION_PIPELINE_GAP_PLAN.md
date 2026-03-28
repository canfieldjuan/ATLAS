# Extraction Pipeline Gap Remediation Plan

**Created:** 2026-03-27
**Status:** Plan scoped, awaiting approval
**Tests:** 70/70 passing (test fixes committed)

---

## Current Architecture

```
Review (pending)
  -> Tier 1 LLM (vLLM, extract 47 fields, no inference)
    -> _compute_derived_fields() (evidence engine, 7 derived fields)
      -> _validate_enrichment() (structure, types, taxonomy)
        -> Low-fidelity detection (noisy source quarantine)
          -> Storage (JSONB enrichment + indexed columns)
```

Evidence engine rules live in `evidence_map.yaml`. All thresholds, weights, patterns are YAML-driven. Code is pure evaluation machinery.

---

## Gap 1+2: Urgency indicator wiring (combined fix)

**Problem:** Two related issues:
- LLM extracts 9 urgency boolean flags, then `_derive_urgency_indicators()` replaces them entirely with a regex re-scan of the raw text. Extracted flags are discarded.
- `evidence_map.yaml` defines weights for `price_pressure_language` (1.5) and `reconsideration_language` (1.5) but `_derive_urgency_indicators()` never sets them. These weights are dead — every review scores 0 on them.

**Fix:** Merge extracted flags with derived flags instead of replacing. Add the 2 missing derivations.

**Files:**
- `atlas_brain/autonomous/tasks/b2b_enrichment.py` — `_derive_urgency_indicators()` (~line 507)
- `atlas_brain/reasoning/evidence_map.yaml` — verify weight names match

**Changes:**
1. In `_derive_urgency_indicators()`:
   - Accept the LLM-extracted `urgency_indicators` dict as input
   - Start with extracted flags as base (trust the LLM for pattern matching)
   - Overlay derived flags only when the LLM missed them (OR logic, not replace)
   - Add `price_pressure_language` derivation: True if `budget_signals.price_increase_mentioned` OR pricing keywords in text
   - Add `reconsideration_language` derivation: True if patterns like "reconsidering", "second thoughts", "rethinking" in text
2. Verify all 16 evidence_map weight keys have a source (extracted or derived)

**Breaking changes:** None — urgency_score computation uses the same indicator dict, just with more complete data.

**Test additions:** Verify merged indicators include both extracted and derived flags. Verify price_pressure and reconsideration are derived.

---

## Gap 3: Wire pain_override rule

**Problem:** Evidence engine has `override_pain()` method and YAML has keyword-to-category mappings for 12 pain categories. But `_compute_derived_fields()` never calls it. The method is dead code.

**Current flow:**
```
_derive_pain_categories() -> returns [{category, severity}, ...] array
result["pain_category"] = pain_categories[0]["category"]  (scalar, for backward compat)
```

**Missing step:** After setting `pain_category`, call `engine.override_pain()` to catch cases where `_derive_pain_categories()` returned "other" but the evidence_map keywords would catch a real category.

**Fix:**

**File:** `atlas_brain/autonomous/tasks/b2b_enrichment.py` — `_compute_derived_fields()` (~line 660)

**Changes:**
1. After `pain_category = pain_categories[0]["category"]` (line ~665):
   ```python
   pain_category = engine.override_pain(
       pain_category,
       result.get("specific_complaints", []),
       result.get("quotable_phrases", []),
       result.get("pricing_phrases", []),
       result.get("feature_gaps", []),
       result.get("recommendation_language", []),
   )
   result["pain_category"] = pain_category
   ```
2. If the override changes the category, also update `pain_categories[0]["category"]` for consistency

**Breaking changes:** None — only affects reviews where pain_category was "other" and keywords exist.

**Test additions:** Verify that pain_category "other" gets overridden when keywords match.

---

## Gap 4: Quarantine retry task

**Problem:** `enrichment_status='quarantined'` is permanent. No automatic retry path. Reviews quarantined due to `evidence_engine_compute_failure` (a bug we may have fixed) stay quarantined forever.

**Fix:** New scheduled task that retries quarantined reviews.

**Files:**
- `atlas_brain/autonomous/tasks/b2b_enrichment_repair.py` — add `retry_quarantined()` function
- `atlas_brain/config.py` — add config for retry (cron, max retries, reason filter)

**Design:**
1. Query `b2b_reviews WHERE enrichment_status = 'quarantined'`
2. Filter by `low_fidelity_reasons`:
   - `['evidence_engine_compute_failure']` — always retry (bug may be fixed)
   - `['vendor_absent_noisy_source']` etc. — skip (intentional quarantine)
3. For retryable reviews:
   - If raw `enrichment` JSONB has tier1 extract data (schema_version >= 1), skip LLM, re-run only `_compute_derived_fields()`
   - If no extract data, reset to `pending` for full re-enrichment
4. On success: update status to `enriched`
5. On failure: increment attempts, keep quarantined, update reason
6. Configurable: `quarantine_retry_cron`, `quarantine_retry_max_attempts`, `quarantine_retry_reasons` (comma-separated list of retryable reasons)

**Breaking changes:** None — new task, opt-in via config.

**Test additions:** Verify retry logic for each reason code. Verify re-derivation without LLM call.

---

## Gap 5: Evidence engine path validation

**Problem:** Invalid `ATLAS_B2B_CHURN_EVIDENCE_MAP_PATH` causes `FileNotFoundError` at enrichment time, not at startup. First review in a batch fails, entire batch affected.

**Fix:** Validate at app startup.

**Files:**
- `atlas_brain/reasoning/evidence_engine.py` — `get_evidence_engine()`
- `atlas_brain/main.py` — startup hook

**Changes:**
1. In `get_evidence_engine()`: wrap file open in try/except, log clear error + fall back to default path
2. In app startup (`main.py` lifespan): call `get_evidence_engine()` to trigger early validation + YAML parse
3. Log the resolved path and rule count at startup

**Breaking changes:** None — fails earlier with better error message.

**Test additions:** Verify fallback to default path on invalid config.

---

## Gap 6: Evidence map version tracking

**Problem:** If YAML rules change (urgency weights, pain keywords, etc.), old reviews keep stale derived fields. No recomputation mechanism.

**Fix:** Track evidence_map hash in enrichment metadata. Add recomputation trigger.

**Files:**
- `atlas_brain/reasoning/evidence_engine.py` — compute + expose map hash
- `atlas_brain/autonomous/tasks/b2b_enrichment.py` — store hash in enrichment JSONB
- `atlas_brain/autonomous/tasks/b2b_enrichment_repair.py` — add `recompute_stale_derivations()` task

**Design:**
1. `EvidenceEngine` computes SHA256 of the YAML file at load time, exposes as `engine.map_hash`
2. During enrichment, store `enrichment.evidence_map_hash = engine.map_hash` alongside schema_version
3. New repair task `recompute_stale_derivations()`:
   - Query reviews WHERE `enrichment->>'evidence_map_hash' != current_hash` AND schema_version >= 1
   - Re-run `_compute_derived_fields()` only (skip LLM)
   - Update enrichment JSONB + hash
4. Run after evidence_map edits or on schedule (weekly)

**Breaking changes:** None — additive field in JSONB.

**Test additions:** Verify hash changes when YAML changes. Verify recomputation updates derived fields.

---

## Execution Order

```
Gap 1+2 (urgency indicator wiring)    — highest impact, fixes dead weights
  |
Gap 3   (pain_override wiring)        — small, clean fix
  |
Gap 4   (quarantine retry)            — new task, no existing code changed
  |
Gap 5   (startup validation)          — safety net
  |
Gap 6   (map version tracking)        — long-term correctness
```

Gaps 1-3 are code changes to existing files. Gaps 4-6 are additive (new functions/tasks).

---

## Key Files

| File | Gaps |
|------|------|
| `autonomous/tasks/b2b_enrichment.py` | 1, 2, 3 |
| `reasoning/evidence_map.yaml` | 1 (verify weight names) |
| `reasoning/evidence_engine.py` | 5, 6 |
| `autonomous/tasks/b2b_enrichment_repair.py` | 4 |
| `config.py` | 4, 5 |
| `main.py` | 5 |

## Schema Notes

- No migrations needed for Gaps 1-5
- Gap 6 adds `evidence_map_hash` to enrichment JSONB (no migration, additive)
- All changes are backward-compatible with existing enriched reviews
