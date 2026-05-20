# PR-Affiliate-Partner-Migrations

## Why this slice exists

The affiliate-system investigation (`reports/affiliate-system-investigation.md`,
item #2 / #3) flagged that only Amazon Associates is version-controlled
(`atlas_brain/storage/migrations/088_seed_amazon_affiliate.sql`). The five
B2B affiliate partners that actually drive revenue -- HubSpot, Pipedrive,
Shopify, HelpDesk, Monday.com -- existed solely as live `affiliate_partners`
rows created via the `/b2b/tenant/affiliates` API. That left:

- **No git history / no review** for partner URLs, commission terms, or the
  `product_aliases` arrays that decide which posts get an affiliate link.
- **No disaster-recovery path.** A restore from a stale backup would silently
  lose or mismatch partner data; nothing reconstructs these rows.
- **No audit trail.** A wrong URL or category edited directly in the DB has no
  reviewer in the loop.

This slice closes that gap by seeding the five existing partners from a
migration, so the partner definitions live in version control like every
other schema fact.

## Scope (this PR)

1. Add `326_seed_b2b_affiliate_partners.sql` -- five idempotent
   `INSERT ... ON CONFLICT ((lower(product_name))) DO NOTHING` statements,
   one per partner, mirroring the live DB rows dumped 2026-05-19.
2. Reproduce `product_aliases` verbatim, because the arrays are load-bearing
   in two query paths (see Mechanism).
3. No code change. The generator, API, and frontend already read these rows;
   this PR only moves the row *definitions* into version control.

### Files touched

- `atlas_brain/storage/migrations/326_seed_b2b_affiliate_partners.sql`
- `plans/PR-Affiliate-Partner-Migrations.md`

## Mechanism

The migration runner (`atlas_brain/storage/migrations/__init__.py`) discovers
`*.sql` files in sorted (numeric-prefix) order, tracks applied migrations by
filename in `schema_migrations`, and runs each pending file exactly once via
`pool.execute(sql)`. The next free prefix after `325_ticket_faq_markdown.sql`
is 326.

`ON CONFLICT ((lower(product_name)))` targets the expression unique index
`idx_affiliate_partners_product ON affiliate_partners (LOWER(product_name))`
from migration 063, matching the 088 precedent. `DO NOTHING` makes the
migration a no-op on the live DB where the rows already exist, and a seed on
a fresh database -- safe under any re-application.

`product_aliases` is reproduced verbatim because two query paths match
competitor/vendor names against `product_name` OR any alias:

- the blog generator's vendor matcher
  `_pick_affiliate_partner_for_vendors` (e.g. vendor "monday" resolves to
  Monday.com via the `monday` alias), and
- the `/b2b/tenant/affiliates/opportunities` competitor JOIN
  (`LOWER(rc.competitor_name) = ANY(SELECT LOWER(unnest(ap.product_aliases)))`).

Altering or dropping aliases would change which posts receive an affiliate
link on a rebuilt database, so the arrays are copied exactly --
`{shopify plus, shopify basic, shopify advanced}`,
`{helpdesk.com}`, `{monday, monday CRM, monday work OS}`, and empty for
HubSpot / Pipedrive.

## Intentional

- **One batch migration, not five.** The report's "one partner = one
  migration" rule applies going *forward*, when a partner is added in
  isolation and wants its own reviewable diff. The five existing partners are
  a single bounded back-fill event, so one migration matches the slice.
- **`DO NOTHING`, not `DO UPDATE`.** The runner tracks by filename and runs
  once, but `DO NOTHING` is the safe choice if the file is ever re-applied: it
  never clobbers a live row that may have diverged intentionally. Matches 088.
- **Notes kept verbatim, including program IDs.** The Shopify Publisher ID
  (`7062841`) and HelpDesk Affiliate ID (`OWvKUHFvg`) are already shipped
  publicly inside the affiliate URLs in the published blog HTML, so
  version-controlling them in `notes` is no new exposure.
- **`created_at` left to `DEFAULT NOW()`.** The matcher orders partners by
  `created_at` only as a tiebreaker when two partners match the *same* vendor,
  which does not happen across the five distinct vendors here. Preserving the
  original timestamps would add noise without changing behavior.

## Deferred

- **Per-partner migrations for future additions.** New partners after this
  back-fill should each ship their own `NNN_seed_<partner>.sql` so the
  addition is independently reviewable.
- **Delete the dead `_fetch_affiliate_partner_by_category` stub.** It was
  deprecated to `return None` (b2b_blog_post_generation.py ~6608) after the
  editorial-mismatch fix; it is intentionally dead, kept only because some
  tests still mock the name. A follow-up can remove it once those mocks are
  retired.
- **Schema-level placeholder guard** (`CHECK (affiliate_url NOT LIKE
  'https://example.com/%')`). The generator already filters placeholder rows
  at match time (`_is_placeholder_partner`); a DB-level constraint is
  defense-in-depth, not blocking.
- **Admin UI for partner management** (report item #6) and click-to-commission
  reconciliation (item #7) remain out of scope.

## Verification

- Idempotent no-op against the live DB (rows already present), inside a
  rolled-back transaction:
  `psql ... <<'SQL' BEGIN; \i .../326_seed_b2b_affiliate_partners.sql;
  SELECT COUNT(*) ...; ROLLBACK; SQL`
  -> `INSERT 0 0` x5, `count` `6` before and after, `ROLLBACK`. Live count
  re-checked after: `6` (untouched).
- Fresh-insert + fidelity, inside a rolled-back transaction:
  `BEGIN; DELETE FROM affiliate_partners WHERE product_name <> 'Amazon';`
  (`DELETE 5`, `after_delete = 1`) -> `\i .../326...` (`INSERT 0 1` x5,
  `after_reinsert = 6`) -> `SELECT product_name, product_aliases, category,
  commission_type, commission_value, notes, enabled ...` returned all five
  rows matching the 2026-05-19 dump exactly (aliases, categories, commission
  terms, and notes verbatim) -> `ROLLBACK`. Live count after: `6`.
- `git diff --check` -> passed.

## Estimated diff size

| Area | Estimated LOC |
|---|---:|
| `326_seed_b2b_affiliate_partners.sql` (5 inserts + header) | ~95 |
| Plan doc | ~120 |
| **Total** | **~215** |
