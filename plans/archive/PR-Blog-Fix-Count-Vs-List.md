# PR-Blog-Fix-Count-Vs-List

Ownership lane: `content-ops/blog-fix-count-vs-list`

## Why this slice exists

Phase-2 fix for the `count_vs_list` detector class (prose states a count of
alternatives/competitors/vendors, then enumerates a different number inline). The
auditor flagged two posts; on inspection they are NOT the same:

- **salesforce-deep-dive L225** -- a real defect. "Reviewers frequently compare
  Salesforce to **six** alternatives: HubSpot, Zoho, Pipedrive, and ServiceNow."
  lists **four**. DB-verified, all four names are real in
  `b2b_product_profiles.commonly_compared_to` (HubSpot 23, Zoho 10, Pipedrive 8,
  ServiceNow 6); the section below it gives each of those four a dedicated
  paragraph and L230 summarizes exactly those four. Only the count word is wrong.
- **teamwork-deep-dive L230** -- a detector **false positive**. The body lists 6
  (Trello, Asana, Basecamp, Monday.com, Slack, Chaser) and says "six"; the `.` in
  "Monday.com" truncated the auditor's list-capture regex to 4 items. No content
  defect in the count itself.

## Scope (this PR)

- **Data** (`salesforce-deep-dive-2026-04.ts` L225): "six alternatives" ->
  "four alternatives". Minimal honest fix -- align the count word to the
  enumerated (and DB-grounded) four-vendor list; no invention, no list expansion.
- **Park** (`ATLAS-HARDENING.md`): record the teamwork grounding/consistency
  finding surfaced while clearing this class -- "Slack" is named at L230 but is
  absent from the DB `commonly_compared_to`, and L145 (FAQ, 5 alternatives) vs
  L230 (body, 6) disagree. That is a grounding/prose-consistency issue, a
  different class from count-vs-list; parked for the Phase-2 deep pass.

### Files touched

- `plans/PR-Blog-Fix-Count-Vs-List.md`
- `atlas-churn-ui/src/content/blog/salesforce-deep-dive-2026-04.ts`
- `ATLAS-HARDENING.md`

## Mechanism

One word changed in published prose ("six" -> "four") so the count matches the
four enumerated, DB-verified competitors. The detector false positive is fixed
out-of-band in the untracked auditor skill (see Verification) so the class reads
genuinely clean; teamwork's separate grounding issue is parked, not silently
absorbed.

## Intentional

- **Count -> list, not list -> DB.** The honest minimal fix is to correct the
  count to the four names already shown (all real), not to expand the list to
  six names from a profile that was recomputed after the post shipped.
- **No generator edit.** The D2 fix (`_blueprint_vendor_deep_dive` derives the
  count and the list from the same capped `comp_names`) already prevents new
  posts from this mismatch; this slice is data-correction for an already-shipped
  post, so no `b2b_blog_post_generation.py` change and no mirror sync.
- **Teamwork left content-unchanged, parked.** Its count_vs_list flag was a
  detector artifact; its real grounding/consistency issue is a different class
  and goes to ATLAS-HARDENING for the deep pass.

## Deferred

- teamwork "Slack" grounding + FAQ(5)/body(6) count mismatch -- parked in
  `ATLAS-HARDENING.md` (grounding/prose-consistency class, not count-vs-list).
- The rest of the Phase-2 triage: `prose_vs_chart_metric` (2 posts) and the big
  `undeclared_quoted_source` class (15, needs an editorial scoping decision).

## Verification

- Detector (untracked skill `audit-published-posts.js`): widened
  `detectCountVsList`'s list-capture to permit a mid-token `.` ("Monday.com") and
  terminate only on a sentence period (`.`+space/end/tag) or `</p>`; added a
  dotted-name regression fixture. `--self-test`: ALL PASS (23 cases, incl. the 5
  count-vs-list fixtures).
- Post-fix corpus run (`--repo=atlas-churn-ui`, 78 posts): `count_vs_list`
  total = **0** -- teamwork drops off (FP fixed), salesforce cleared (six->four),
  no new true positives surfaced from un-truncating dotted-name lists.
- salesforce L225 now reads "four alternatives" and is consistent with the four
  dedicated competitor paragraphs (L226-229) and the L230 summary.

## Estimated diff size

| Area | LOC |
|---|---:|
| salesforce data (count word) | ~2 |
| ATLAS-HARDENING park entry | ~8 |
| Plan doc | ~87 |
| **Total** | **~97** |
