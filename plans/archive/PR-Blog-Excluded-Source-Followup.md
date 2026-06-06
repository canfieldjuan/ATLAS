# PR-Blog-Excluded-Source-Followup

Ownership lane: `content-ops/blog-excluded-source-followup`

## Why this slice exists

Follow-up to the excluded-source-quote class (#834/#837/#838). The #837 review
caught that the detector's original on-form gate (`reviewer on <Source>`) missed
the adjective form (`<Source> reviewer`); widening it (in #837/#838) surfaced 4
posts that were NEVER in the original census because all their excluded-source
quotes use the adjective/reported-speech form:
best-project-management-for-201-1000, gusto-vs-workday, help-scout-vs-zendesk,
pipedrive-deep-dive. This slice paraphrases them, taking corpus-wide
`excluded_source_quote` to 0.

## Scope (this PR)

Drop the excluded-source name + verbatim form, keep substance (allowlist quotes --
G2/Reddit/Slashdot -- left verbatim):
- **best-project-management-for-201-1000** L363: Software Advice ($265 renewal) ->
  reported speech.
- **gusto-vs-workday** L148 (Capterra, "great features"/"behind in design"), L150
  (Software Advice, Elevate transition), L181 (Capterra, feature breadth).
- **help-scout-vs-zendesk** L155 (TrustRadius, folders/customization), L161
  (Trustpilot, support interaction), L193 (Software Advice, "pricey and complex").
- **pipedrive-deep-dive** L165 (Software Advice "great for keeping sales
  organized"), L172/L183 (Trustpilot "no service/no support" support quote),
  L184/L208/L349/L388 (Software Advice "limited built-in marketing tools" -- the
  same quote repeated across the post). Reworded each to reported speech; kept the
  Slashdot/Reddit blockquotes.

### Files touched

- `plans/PR-Blog-Excluded-Source-Followup.md`
- `atlas-churn-ui/src/content/blog/best-project-management-for-201-1000-2026-04.ts`
- `atlas-churn-ui/src/content/blog/gusto-vs-workday-2026-04.ts`
- `atlas-churn-ui/src/content/blog/help-scout-vs-zendesk-2026-04.ts`
- `atlas-churn-ui/src/content/blog/pipedrive-deep-dive-2026-04.ts`

## Mechanism

Same paraphrase pattern as the earlier excluded-source slices: the excluded
platform can't be quoted/attributed per the blog allowlist, so each quote becomes
reported speech (source name + verbatim form dropped, substance + figures kept).
No invention.

## Intentional

- **Allowlist quotes untouched** (pipedrive's Slashdot pipeline-visualization and
  Reddit CRM-search blockquotes stay verbatim).
- **Repeated quotes generalized consistently** (pipedrive's "great for keeping
  sales organized / limited built-in marketing tools" appears 3-4x; each
  occurrence reworded to the same reported-speech form).

## Deferred

- **DD3 (prose_vs_chart)**: asana, linode, mailchimp still flagged -- deferred
  (chart-provenance entanglement), separate slice.
- Recon on the un-swept post types (landscape / top-complaint / why-teams-leave).

Parked hardening: none.

## Verification

- All 4 posts re-audited under the WIDENED detector (`--slug=`):
  `excluded_source_quote` = 0, no new findings; grep confirms no residual
  `<excluded> reviewer` / `reviewer on <excluded>` attributions.
- Full corpus (78 posts): `excluded_source_quote` = **0** -- the class is now
  complete corpus-wide.

## Estimated diff size

| Area | LOC |
|---|---:|
| 4 posts (paraphrases) | ~28 |
| Plan doc | ~70 |
| **Total** | **~98** |
