# EvidenceClaim canary audit — 2026-04-26

Pre-Monday-batch snapshot. Scoped synthesis run on three canary vendors
covering the diverse failure modes the contract was designed to catch:

- **Pipedrive** — clean v4 baseline (does the contract reject legitimate claims?)
- **Monday.com** — antecedent-trap surface (does the regex catch the original failure?)
- **ClickUp** — v3-dominant (does cannot_validate stay distinct from invalid?)

Run command:

```
ATLAS_B2B_CHURN_EVIDENCE_CLAIM_SHADOW_ENABLED=true \
  python scripts/seed_evidence_claim_shadow.py
```

Cost: ~$0.30 in OpenRouter Sonnet 4.5 calls. Wall time: 88s.

## Headline figures (post Step-7 eligibility tightening)

```
total            64
valid            10
invalid          33
cannot_validate  21
```

| vendor      | total | valid | invalid | cannot_validate |
|-------------|------:|------:|--------:|----------------:|
| ClickUp     |   25  |   4   |    4    |       17        |
| Monday.com  |   22  |   2   |   16    |        4        |
| Pipedrive   |   17  |   4   |   13    |        0        |

## Antecedent traps caught

Three traps, all on Monday.com, zero false positives on Pipedrive or
ClickUp. Same witness produced two of the three (the regex fires once
per claim_type emitted for that witness):

```
Monday.com | pain_claim_about_vendor | witness:c9296400...:0
  excerpt: "com, we were using HubSpot, which, even though it is a very good
            technical tool ..."

Monday.com | pain_claim_about_vendor | witness:c9296400...:1
  excerpt: "Having a simpler UI with technical functionalities of a backend
            and a good user-..."

Monday.com | feature_gap_claim       | witness:c9296400...:1
  (same witness as above)
```

The first match is the literal "Before Monday.com, we were using HubSpot"
self-transition. Self-transition pattern 1 fires deterministically without
needing `known_vendor_names`. The second/third are the same witness in a
review whose source window also contains the transition phrase; the regex
correctly treats the trait as belonging to the prior tool.

This is the original canary failure mode. **The contract is now blocking it.**

## v3 vs v4 distinction

Pipedrive carries 0 `cannot_validate` rows: clean v4 throughout.
ClickUp carries 17, almost all `phrase_subject_unavailable` on
pricing/UX witnesses. The contract is correctly conservative on tagless
data instead of misrendering it as pain.

## Step-7 eligibility tightening — before/after

The full canary run produced 98 rows pre-tightening; after tightening
counterevidence eligibility to `phrase_polarity == 'positive'` and
adding the per-artifact pre-write purge, the same canary produced 64.

| metric              | before | after | delta |
|---------------------|-------:|------:|-----:|
| Total rows          |    98  |   64  | -35% |
| Valid               |    10  |   10  |  0   |
| Invalid             |    58  |   33  | -43% |
| cannot_validate     |    30  |   21  | -30% |
| Counterevidence rows|    36  |    2  | structural noise gone |
| Antecedent traps    |     3  |    3  |  0   |

Valid count held flat at 10 while total attempts dropped 35%: removed
audit noise, did not lose evidence. The 2 surviving counterevidence
rows are both `valid` with `phrase_polarity='positive'`, exactly the
witnesses the validator should see.

## Where the 10 valid claims live

| vendor      | valid | overall_dissatisfaction | pain_claim | counter | named_account |
|-------------|------:|------------------------:|-----------:|--------:|--------------:|
| ClickUp     |   4   |          4              |     0      |   0     |       4       |
| Monday.com  |   2   |          0              |     1      |   1     |       0       |
| Pipedrive   |   4   |          3              |     2      |   1     |       1       |

Notes:

- ClickUp's 4 valid `overall_dissatisfaction` rows are all on
  `named_account_anchor` claims, not pain claims. The pain_category is
  propagated context, not the claim subject -- lower headline risk than
  the global figure suggested.
- Pipedrive carries 3 valid pain-side rows tagged
  `overall_dissatisfaction`. These are the generic-pain-category
  headline candidates flagged below.

## Open design decision (deferred)

`overall_dissatisfaction` is one of `GENERIC_PAIN_CATEGORIES` per
`enrichment_contract.py`. The plan says:

> generic categories are disallowed for headline claims unless
> explicitly requested.

The validator currently accepts it as a valid `pain_claim_about_vendor`,
which means a battle-card or briefing renderer that consumes
`select_best_claim()` could surface a generic catch-all phrase as the
vendor's headline pain. **Decision: keep validator permissive; enforce
"not headline-safe" at claim-selection / API consumer level later.**
Filtering at the consumer stops false-positive headlines while
preserving the row as fallback evidence in detail views.

Concrete enforcement points (Steps 8+):

1. `select_best_claim()` accepts a `disallow_generic_pain_category`
   option, default off; battle_card / briefing migrations turn it on.
2. The API surface (`atlas_brain/api/b2b_evidence.py`) exposes a
   `headline_safe=true` filter that applies the same rule.
3. Detail views and EvidenceDrawer pass through the unfiltered set.

## Tests pinning the canary state

```
tests/test_evidence_claim_fixtures.py            (24 / 24)
tests/test_evidence_claim_repository_live.py     (12 / 12)
tests/test_evidence_claim_builder_live.py        ( 7 /  7)
tests/test_evidence_claim_audit_live.py          ( 5 /  5)
tests/test_evidence_claim_audit_wiring.py        ( 4 /  4)
                                                ----------
                                                  52 / 52
```

## Status

Pre-Monday-batch shadow capture is live and clean. No code blockers
identified for the full nightly run. Step 8 (API exposure) waits on
the post-Monday review.
