# Atlas Affiliate-Links System Investigation

**Date:** 2026-05-19
**Context:** Triggered by discovery that 27 published blog posts had placeholder
affiliate data (\`example.com/atlas-live-test-partner\`) and 1 post had a real
HubSpot affiliate URL on a post that doesn't analyze HubSpot. User requested
investigation before building any further affiliate tooling.

---

## 1. Where the system lives

**Schema:** \`atlas_brain/storage/migrations/063_affiliate_partners.sql\`

Tables:
- \`affiliate_partners(id, name, product_name, product_aliases[], category, affiliate_url, commission_type, commission_value, notes, enabled)\`
- \`affiliate_clicks(id, partner_id, review_id, referrer, clicked_at)\`

**API:** \`atlas_brain/api/b2b_affiliates.py\`

Full CRUD: list / create / update / delete partners; record clicks. Mounted at
\`/b2b/tenant/affiliates/\`.

**Generator integration:** \`atlas_brain/autonomous/tasks/b2b_blog_post_generation.py\`

- Lines 1279–1303: \`_inject_affiliate_links()\` — markdown placeholder
  replacement and bare-URL wrapping in post body.
- Lines 6020–6041: attachment of affiliate data to \`data_context\` during
  blueprint construction.
- Lines 6470–6495: \`_fetch_affiliate_partner()\` (by ID) and
  \`_fetch_affiliate_partner_by_category()\` lookup functions.

**Seeding:** \`atlas_brain/storage/migrations/088_seed_amazon_affiliate.sql\`

Only one partner seeded: Amazon Associates with \`category='consumer'\`. No
B2B-category partners are version-controlled.

---

## 2. How it's supposed to work

1. **Partner registration:** admins POST to \`/b2b/tenant/affiliates/\` with
   partner metadata (name, product, category, URL, commission terms).
2. **Generator lookup at draft time:**
   - For \`vendor_alternative\` topic type: explicit lookup by \`affiliate_id\`
     pulled from the topic context.
   - For all other topic types: category-based fallback —
     \`SELECT ... FROM affiliate_partners WHERE enabled=true AND
     LOWER(category) = LOWER(:category) LIMIT 1\`.
3. **Injection:** the partner's \`affiliate_url\` + name + product_name + slug
   are written into \`data_context\` in the generated \`.ts\` file.
4. **Markdown rendering:** during post-finalization, the generator scans
   \`content\` for \`{{affiliate:product-slug}}\` placeholders and replaces
   them with markdown links; bare URLs matching known partners are wrapped.
5. **Click tracking:** the frontend POSTs to a click-recording endpoint on
   anchor click → row in \`affiliate_clicks\`.

---

## 3. Where the placeholder ("Atlas Live Test Partner") came from

**Unresolved.** The string is not present anywhere in current code or
migrations. Three plausible explanations:

- A test row existed in the live \`affiliate_partners\` table (matching
  \`category='b2b_software'\` or similar) and has since been deleted from the
  DB. Posts published while it existed kept the snapshot.
- An earlier version of the generator hardcoded a fallback placeholder
  during development; that code has been removed but the published
  artifacts remain.
- Manual injection via the API for testing the end-to-end pipeline.

**The placeholder data is gone from the 27 affected posts** (PR #617 strips
it). But there's **no generator-side guard** preventing the same shape of
data from being injected again if a placeholder partner row is recreated in
the DB.

---

## 4. Current vs. intended state

**Current state: half-built with data-integrity gaps.**

- ✓ Schema is production-shaped (enabled flag, commission tracking, click
  analytics).
- ✓ API is complete (full CRUD).
- ✓ Generator hook is functional.
- ✗ **Seeding is minimal** — only Amazon Associates is in migrations. Every
  other partner exists only in live DB rows, not version-controlled.
- ✗ **No editorial validation.** A "CRM" category match returns the first
  CRM affiliate regardless of whether the post actually discusses that
  vendor.
- ✗ **No category enforcement at the schema level.** \`category\` is plain
  TEXT — a typo (\`crm\` vs \`CRM\`) silently breaks matching.
- ✗ **No admin UI.** Partners can only be managed via direct API calls.
- ✗ **No disclosure surface.** Affiliate links render as ordinary links in
  the blog without an FTC-compliant disclosure badge or footer notice.

---

## 5. The HubSpot affiliate on \`top-complaint-every-crm-2026-04\`

The post analyzes 8 CRM vendors: Salesforce, Copper, Zoho CRM, Close,
Pipedrive, Freshsales, Insightly, Nutshell. **HubSpot is not analyzed.**

But the post's \`data_context\` has:
\`\`\`
affiliate_url: "https://hubspot.com/?ref=atlas"
affiliate_partner: { name: "HubSpot Partner", product_name: "HubSpot", slug: "hubspot" }
\`\`\`

**Why this happened:** the generator's category-fallback path matched the
post's \`category='CRM'\` against the live \`affiliate_partners\` table and
returned the first enabled CRM-category row. That row is HubSpot Partner.

**Root issue:** the row exists in the live database but is NOT in any
migration. It was created manually via the API at some point. The
relationship between "what the post discusses" and "what affiliate gets
injected" is purely category-based — no check that the affiliate vendor
appears in the post's vendor set.

---

## 6. Risks to flag

1. **FTC / regulatory compliance:** affiliate links render without a visible
   disclosure. Posts ship with no "this post contains affiliate links"
   notice, no rel="sponsored" on the anchor tags. This is a real legal
   risk depending on jurisdiction and FTC enforcement priorities.

2. **Editorial mismatch:** HubSpot Partner is injected into a post that
   doesn't discuss HubSpot. Readers see "Here are the top complaints about
   8 CRMs (Salesforce, Copper, Zoho, ...)" followed by an affiliate CTA for
   HubSpot. Reads as bait-and-switch.

3. **Database-only partner records:** non-Amazon partners exist only in
   live DB. No git history, no code review, no audit trail. If the DB is
   restored from a stale backup, partner data is lost or mismatched. If
   the DB row is mistakenly edited (wrong URL, wrong category), no
   reviewer catches it.

4. **No category enforcement:** \`category\` is plain TEXT. Typos and
   inconsistent casing silently break matching. The fallback when no
   match is found is unclear from the generator code — does the post
   ship with empty \`data_context\`, or fall through to a default?

5. **\`LIMIT 1\` with no ordering:** if multiple affiliates match a
   category, the order is undefined (insertion order in Postgres,
   typically). No prioritization, no rotation, no relevance scoring.

6. **No placeholder validation:** no schema check or generator-side test
   rejects placeholder values. \`example.com\` URLs, "test" partner names,
   and empty-but-truthy \`affiliate_url\` strings all pass through.

7. **Click tracking decoupled from revenue:** \`affiliate_clicks\` records
   clicks but there's no linkage to commission tracking. No way to
   correlate "we drove 412 clicks to HubSpot this month" with "HubSpot
   paid us $X." Affiliate-program reconciliation is manual.

---

## Recommended sequencing if you build on this

If the user wants to make the affiliate system production-grade:

1. **First, an inventory of live DB rows.** What partners exist in the
   current \`affiliate_partners\` table? Are they real, legal, and intended?
   Dump and review before anything else.

2. **Move partner definitions into migrations.** One partner row = one
   versioned migration. Stop creating partners via raw API calls in prod.

3. **Add editorial validation to the generator.** Before injecting an
   affiliate partner, confirm the partner's \`product_name\` (or aliases)
   appears in the post's vendor set / target_keyword / content.

4. **Add disclosure rendering.** Either a per-post \`<aside>\` disclosure
   block or a footer notice. \`rel="sponsored"\` on the affiliate anchor.

5. **Add placeholder guards.** Schema-level: \`CHECK (affiliate_url NOT LIKE
   'https://example.com/%')\`. Generator-level: refuse to inject if URL
   matches placeholder patterns or partner name contains "test".

6. **Add an admin UI.** Atlas already has an admin-ui; affiliate partner
   management belongs there.

7. **Wire click tracking to commission reports.** Either pull commission
   data via affiliate-network APIs (Amazon Associates, Impact, etc.) or
   require manual entry but at least surface it next to click counts.

None of these are blocking the safety net or placeholder strip — but they
ARE blocking any responsible re-introduction of affiliate links to the
blog.
