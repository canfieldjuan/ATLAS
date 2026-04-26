# EvidenceClaim post-nightly review checklist

Run this every morning during the Step 7 soak. Each step is a single
shell command and a pass/fail rule. The goal is to make tomorrow's
review mechanical -- no judgement calls until something fails a rule
and needs a human.

## 0. One-time precondition

Before the first nightly cycle:

- [ ] `ATLAS_B2B_CHURN_EVIDENCE_CLAIM_SHADOW_ENABLED=true` set in the
      environment the synthesis task reads
- [ ] Synthesis service restarted so the flag is picked up
- [ ] Scheduled audit row enabled (cron `45 4 * * *` is fine; the seed
      defaults to `enabled=False`)

## 1. Did the capture run?

```bash
PGPASSWORD=atlas psql -U atlas -h localhost -p 5433 -d atlas -tAc \
  "SELECT count(*), count(DISTINCT vendor_name)
   FROM b2b_evidence_claims WHERE as_of_date = CURRENT_DATE;"
```

- **Pass:** count > 0 and distinct vendors >= the number of vendors the
  prior synthesis cycle reasoned over.
- **Fail (zero rows):** the audit task will already have alerted via
  ntfy. Check synthesis logs for the `Evidence claim shadow capture`
  line and confirm the flag is on. The autonomous task fires alerts
  on `total_rows == 0`, `cannot_validate / total > 60%`, or any single
  rejection_reason exceeding 25% of invalid total.

## 2. Are the totals in a healthy range?

```bash
python scripts/audit_evidence_claims.py --as-of $(date -I)
```

Expected ranges based on the canary baseline (3 vendors, 64 rows):

| metric              | per-vendor expected | red flag                          |
|---------------------|---------------------|-----------------------------------|
| valid / total       | 5%-25%              | < 2% or > 50% on a v4-clean vendor|
| cannot_validate / total | 0%-50%          | > 70% on a vendor with v4 dominance |
| invalid / total     | 30%-60%             | > 80% (validator over-rejecting)  |
| antecedent_trap rows| 0-3 per vendor      | 0 on a vendor that historically had traps (regression) |

Scale linearly with vendor count: 30 vendors -> ~640 rows under the
canary distribution.

## 3. Are antecedent traps still firing where expected?

```bash
PGPASSWORD=atlas psql -U atlas -h localhost -p 5433 -d atlas -c "
SELECT vendor_name, count(*) AS traps
FROM b2b_evidence_claims
WHERE as_of_date = CURRENT_DATE
  AND rejection_reason = 'antecedent_trap'
GROUP BY 1 ORDER BY 2 DESC;
"
```

- **Pass:** Monday.com appears with at least 1 trap (the original canary
  failure source). Other vendors may also surface traps -- that is fine
  if the source text genuinely contains a transition phrase.
- **Fail (Monday.com missing):** the source-window plumbing or the
  regex regressed. Check that `source_reviews` is being passed into
  `write_evidence_claims_for_synthesis` and that
  `known_vendor_names` is non-empty.

## 4. What are the top rejection reasons?

```bash
python scripts/audit_evidence_claims.py --as-of $(date -I) --json | \
  jq '.rejection_reasons_by_claim_type'
```

Per claim_type, the top-3 reasons should match the gate intent:

| claim_type                  | top 3 rejection reasons (expected)                    |
|-----------------------------|-------------------------------------------------------|
| pain_claim_about_vendor     | subject_not_subject_vendor / pain_confidence_none / not_grounded |
| counterevidence_about_vendor| (rare; eligibility filters most out -- 0-2 rows OK)   |
| named_account_anchor        | role_passing_mention / phrase_role_unavailable        |
| feature_gap_claim           | not_grounded / pain_confidence_none / antecedent_trap |
| pricing_urgency_claim       | subject_not_subject_vendor                            |

A single reason exceeding 25% of an invalid claim_type's count is the
autonomous task's trigger; investigate why a gate is rejecting at that
rate.

## 5. Generic-pain-category headline candidates

These are the rows tagged `pain_category=overall_dissatisfaction` (or
other generics from `GENERIC_PAIN_CATEGORIES`) that VALIDATED. The
validator is permissive on these by design; battle_card / briefing
migrations should filter them at consumer-level. Worth tracking the
volume so the eventual `headline_safe=true` filter has known behavior.

```bash
PGPASSWORD=atlas psql -U atlas -h localhost -p 5433 -d atlas -c "
SELECT vendor_name, claim_type, count(*) AS valid_generic
FROM b2b_evidence_claims
WHERE as_of_date = CURRENT_DATE
  AND status = 'valid'
  AND claim_payload->>'pain_category' IN (
    'overall_dissatisfaction', 'general_dissatisfaction'
  )
GROUP BY 1, 2 ORDER BY 1, 3 DESC;
"
```

- Track the count over time; the consumer-level filter (Step 8+) will
  drop these from headline surfaces. Detail views and EvidenceDrawer
  should keep them as fallback evidence.

## 6. Idempotency / staleness sanity

```bash
PGPASSWORD=atlas psql -U atlas -h localhost -p 5433 -d atlas -c "
SELECT artifact_id, count(*) FILTER (WHERE created_at::date = CURRENT_DATE) AS today_rows,
       count(*) AS total_rows
FROM b2b_evidence_claims
WHERE as_of_date = CURRENT_DATE
GROUP BY 1
HAVING count(*) FILTER (WHERE created_at::date = CURRENT_DATE) <> count(*);
"
```

- **Pass:** zero rows returned. Every row's `created_at` is today
  because the per-artifact purge wipes prior rows before re-emitting.
- **Fail (rows returned):** the purge regressed. Check
  `write_evidence_claims_for_synthesis` for the leading
  `DELETE FROM b2b_evidence_claims WHERE artifact_type = ... AND
  artifact_id = ...`.

## 7. Two-cycle compare

After the second consecutive cycle completes:

```bash
python scripts/audit_evidence_claims.py --as-of $(date -I -d yesterday) --json > /tmp/y.json
python scripts/audit_evidence_claims.py --as-of $(date -I) --json > /tmp/t.json
diff <(jq '.totals' /tmp/y.json) <(jq '.totals' /tmp/t.json)
diff <(jq '.by_claim_type' /tmp/y.json) <(jq '.by_claim_type' /tmp/t.json)
```

- **Pass:** day-over-day shifts are within +/- 20% on each total. Larger
  shifts that are NOT explained by a vendor-set change probably indicate
  a synthesis or enrichment regression.

## After the Monday April 27 batch

Once the Monday batch (the weekly-bound consumers like challenger_brief
and weekly_churn_feed) has populated b2b_evidence_claims:

- [ ] Confirm distinct_artifacts has grown to cover the weekly batch
      vendors, not just the daily-only ones.
- [ ] Re-run sections 2-5 against Monday's as_of_date.
- [ ] If everything still passes, Step 8 (API exposure behind feature
      flag) is unblocked.
