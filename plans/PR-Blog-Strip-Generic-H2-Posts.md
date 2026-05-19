# PR-Blog-Strip-Generic-H2-Posts

## Why this slice exists

The `seo-geo-aeo-blog-post` skill's audit + the AI-engine
citation-shape research both flag generic section headings
(`<h2>Introduction</h2>`, `<h2>Overview</h2>`, `<h2>Conclusion</h2>`,
etc.) as an anti-pattern: AI engines (ChatGPT, Perplexity, Google AI
Overviews) skip these sections when picking extraction targets
because the heading doesn't predict what's underneath. Featured-
snippet eligibility drops too -- Google extracts based on
heading/content correlation.

The skill's analyzer reports **77 of 79 published blog posts** carry
`<h2 id="introduction">Introduction</h2>` as the first H2 (and one
also carries `<h2 id="conclusion">Conclusion</h2>`). This is by far
the highest-prevalence remaining content-quality issue.

The opening section's actual content is a hook: review counts, data
sources, what the analysis covers, methodology nuance. That content
should lead -- it doesn't need a generic header above it. Strong
blog opening paragraphs in journalism and analyst writing
conventionally have no boilerplate H2; the first H2 reader sees is
the first substantive section.

This slice strips the generic H2 from the 77 already-published
posts. A companion PR adds a generator-side filter so future drafts
never publish with these generic headings again.

## Scope (this PR)

1. Remove `<h2 id="introduction">Introduction</h2>` from every post
   in `atlas-churn-ui/src/content/blog/*.ts` that carries it
   (77 occurrences across 77 posts).
2. Remove `<h2 id="conclusion">Conclusion</h2>` from the one post
   that carries it (`marketing-automation-landscape-2026-04`).
3. Preserve all other content verbatim: the methodology note (if
   present), the prose underneath the stripped H2, the rest of the
   post structure, and every other H2/H3 that names substantive
   sections.

### Files touched

- `atlas-churn-ui/src/content/blog/amazon-web-services-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/asana-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/azure-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/azure-vs-salesforce-2026-03.ts`
- `atlas-churn-ui/src/content/blog/b2b-software-landscape-2026-03.ts`
- `atlas-churn-ui/src/content/blog/b2b-software-landscape-2026-04.ts`
- `atlas-churn-ui/src/content/blog/basecamp-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/best-b2b-software-for-1000-2026-03.ts`
- `atlas-churn-ui/src/content/blog/best-crm-for-51-200-2026-04.ts`
- `atlas-churn-ui/src/content/blog/best-hr-hcm-for-51-200-2026-04.ts`
- `atlas-churn-ui/src/content/blog/best-project-management-for-201-1000-2026-04.ts`
- `atlas-churn-ui/src/content/blog/brevo-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/clickup-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/close-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/close-vs-zoho-crm-2026-04.ts`
- `atlas-churn-ui/src/content/blog/communication-landscape-2026-04.ts`
- `atlas-churn-ui/src/content/blog/copper-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/crm-landscape-2026-04.ts`
- `atlas-churn-ui/src/content/blog/fortinet-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/gusto-vs-workday-2026-04.ts`
- `atlas-churn-ui/src/content/blog/helpdesk-landscape-2026-04.ts`
- `atlas-churn-ui/src/content/blog/help-scout-vs-zendesk-2026-04.ts`
- `atlas-churn-ui/src/content/blog/hr-hcm-landscape-2026-04.ts`
- `atlas-churn-ui/src/content/blog/hubspot-deep-dive-2026-03.ts`
- `atlas-churn-ui/src/content/blog/hubspot-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/hubspot-vs-power-bi-2026-04.ts`
- `atlas-churn-ui/src/content/blog/insightly-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/insightly-vs-zoho-crm-2026-04.ts`
- `atlas-churn-ui/src/content/blog/intercom-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/jira-vs-mondaycom-2026-04.ts`
- `atlas-churn-ui/src/content/blog/jira-vs-trello-2026-03.ts`
- `atlas-churn-ui/src/content/blog/linode-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/looker-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/magento-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/mailchimp-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/marketing-automation-landscape-2026-04.ts`
- `atlas-churn-ui/src/content/blog/metabase-vs-tableau-2026-04.ts`
- `atlas-churn-ui/src/content/blog/microsoft-defender-for-endpoint-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/microsoft-teams-vs-notion-2026-04.ts`
- `atlas-churn-ui/src/content/blog/microsoft-teams-vs-salesforce-2026-04.ts`
- `atlas-churn-ui/src/content/blog/notion-vs-salesforce-2026-03.ts`
- `atlas-churn-ui/src/content/blog/pipedrive-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/power-bi-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/project-management-landscape-2026-04.ts`
- `atlas-churn-ui/src/content/blog/real-cost-of-copper-2026-04.ts`
- `atlas-churn-ui/src/content/blog/real-cost-of-shopify-2026-04.ts`
- `atlas-churn-ui/src/content/blog/real-cost-of-woocommerce-2026-04.ts`
- `atlas-churn-ui/src/content/blog/salesforce-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/sentinelone-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/shopify-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/slack-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/slack-vs-zoom-2026-04.ts`
- `atlas-churn-ui/src/content/blog/switch-to-asana-2026-04.ts`
- `atlas-churn-ui/src/content/blog/switch-to-clickup-2026-03.ts`
- `atlas-churn-ui/src/content/blog/switch-to-clickup-2026-04.ts`
- `atlas-churn-ui/src/content/blog/switch-to-klaviyo-2026-04.ts`
- `atlas-churn-ui/src/content/blog/switch-to-salesforce-2026-04.ts`
- `atlas-churn-ui/src/content/blog/switch-to-sentinelone-2026-04.ts`
- `atlas-churn-ui/src/content/blog/switch-to-shopify-2026-03.ts`
- `atlas-churn-ui/src/content/blog/switch-to-shopify-2026-04.ts`
- `atlas-churn-ui/src/content/blog/switch-to-woocommerce-2026-04.ts`
- `atlas-churn-ui/src/content/blog/switch-to-zoho-crm-2026-04.ts`
- `atlas-churn-ui/src/content/blog/tableau-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/top-complaint-every-b2b-software-2026-03.ts`
- `atlas-churn-ui/src/content/blog/top-complaint-every-communication-2026-04.ts`
- `atlas-churn-ui/src/content/blog/top-complaint-every-crm-2026-04.ts`
- `atlas-churn-ui/src/content/blog/top-complaint-every-e-commerce-2026-04.ts`
- `atlas-churn-ui/src/content/blog/top-complaint-every-helpdesk-2026-04.ts`
- `atlas-churn-ui/src/content/blog/top-complaint-every-marketing-automation-2026-04.ts`
- `atlas-churn-ui/src/content/blog/top-complaint-every-project-management-2026-04.ts`
- `atlas-churn-ui/src/content/blog/why-teams-leave-azure-2026-03.ts`
- `atlas-churn-ui/src/content/blog/why-teams-leave-azure-2026-04.ts`
- `atlas-churn-ui/src/content/blog/why-teams-leave-slack-2026-04.ts`
- `atlas-churn-ui/src/content/blog/woocommerce-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/workday-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/zoho-crm-deep-dive-2026-04.ts`
- `atlas-churn-ui/src/content/blog/zoom-deep-dive-2026-04.ts`
- `plans/PR-Blog-Strip-Generic-H2-Posts.md`

## Mechanism

A one-off Node script applies the surgical regex strip:
`/^<h2 id="introduction">Introduction<\/h2>\n/m` (and the
parallel `conclusion` variant). The `^...\n` anchoring ensures
exactly one line is removed -- the H2 line and the newline that
follows it. No surrounding content is touched.

Posts that don't carry the literal anti-pattern lines are skipped
untouched. The script reports the changed-count after running so
we can verify against the analyzer's pre-strip baseline (77).

## Intentional

- Strip-only: the H2 is removed, the section's prose is preserved
  as the post's new opening section. The methodology note (one line
  above) becomes the visible cue that introductory content follows.
- One literal regex per anti-pattern variant. The strip refuses to
  match anything except the exact rendered H2 the generator emits.
  Future variants (e.g., `<h2 id="overview">Overview</h2>`) are
  out of scope; the generator-side companion PR catches new shapes
  at draft time, and a future strip pass can clean any specific
  variants that slip through.
- Headings stay literal even if they include emojis or punctuation
  variations -- the regex only matches the exact `id` + text pair
  the generator emits. Misspellings or hand-edited variants are
  untouched by design.

## Deferred

- Generator-side prevention. Lives in a separate slice
  (`PR-Blog-Strip-Generic-H2-Generator`) so the diff stays focused
  on data cleanup.
- Replacing the generic H2 with a topic-specific heading per post.
  Doable but requires per-post copy decisions; not worth the lift
  when the prose underneath already serves as a hook.
- Other anti-pattern headings (`Overview`, `Summary`,
  `Background`, `Conclusion` in non-`id` form). The audit doesn't
  flag them in current production data; cleanup-when-needed
  pattern.

## Verification

- Skill analyzer `Generic <h2>Introduction</h2>` count: 77 / 78 -> 0 / 0
  (`~/.claude/skills/seo-geo-aeo-blog-post/scripts/audit-published-posts.js`).
  Inline aggregating grep fallback for contributors without the skill
  (a single number, not per-file counts):
  `grep -R '<h2 id="introduction">Introduction</h2>' atlas-churn-ui/src/content/blog/ | wc -l` -> `0` after the change.
  Same form for the conclusion case:
  `grep -R '<h2 id="conclusion">Conclusion</h2>' atlas-churn-ui/src/content/blog/ | wc -l` -> `0`.
- `npm run build` in `atlas-churn-ui` -> 83-URL sitemap, no TS errors.
- `git diff --check` -> passed.
- Spot-check on `hubspot-deep-dive-2026-04`: the H2 at line 152
  is removed; the methodology note and the prose underneath are
  preserved verbatim.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| Affected post `.ts` files (77; one-line removal each) | ~78 |
| One post with both Introduction + Conclusion variants stripped | already counted |
| Plan doc | ~120 |
| **Total** | **~200** |
