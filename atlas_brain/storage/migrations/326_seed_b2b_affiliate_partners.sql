-- Version-control the B2B affiliate partner rows.
--
-- Before this migration, only Amazon Associates (088) was seeded from a
-- migration; the five B2B partners below existed solely as live
-- affiliate_partners rows created via the /b2b/tenant/affiliates API. That
-- left no git history, no code review, and no disaster-recovery path -- a
-- restore from a stale backup would silently lose or mismatch partner data.
--
-- Each row reproduces the live DB business columns (dumped 2026-05-19):
-- name, product_name, product_aliases, category, affiliate_url,
-- commission_type, commission_value, notes, enabled. The identity/audit
-- columns (id, created_at, updated_at) intentionally use schema DEFAULTs, so
-- a fresh insert is row-equivalent but not byte-for-byte identical to a dump.
-- product_aliases is reproduced verbatim because it is load-bearing in two
-- places: the blog generator's vendor matcher
-- (_pick_affiliate_partner_for_vendors) and the /opportunities competitor
-- JOIN -- both match competitor/vendor names against product_name OR any
-- alias. Dropping or altering aliases would change which posts get an
-- affiliate link on a rebuilt database.
--
-- ON CONFLICT ((lower(product_name))) DO NOTHING matches the expression
-- unique index idx_affiliate_partners_product and the 088 precedent: on the
-- live DB where these rows already exist the migration is a no-op; on a
-- fresh database it seeds them. Going forward, each NEW partner should get
-- its own migration so the addition is independently reviewable.

INSERT INTO affiliate_partners (
    name, product_name, product_aliases, category, affiliate_url,
    commission_type, commission_value, notes, enabled
) VALUES (
    'HubSpot Partner',
    'HubSpot',
    '{}'::text[],
    'CRM',
    'https://hubspot.com/?ref=atlas',
    'cpa',
    '$150/signup',
    NULL,
    true
) ON CONFLICT ((lower(product_name))) DO NOTHING;

INSERT INTO affiliate_partners (
    name, product_name, product_aliases, category, affiliate_url,
    commission_type, commission_value, notes, enabled
) VALUES (
    'Pipedrive Partner',
    'Pipedrive',
    '{}'::text[],
    'CRM',
    'https://pipedrive.com/?ref=atlas',
    'recurring',
    '20% first year',
    NULL,
    true
) ON CONFLICT ((lower(product_name))) DO NOTHING;

INSERT INTO affiliate_partners (
    name, product_name, product_aliases, category, affiliate_url,
    commission_type, commission_value, notes, enabled
) VALUES (
    'Shopify Affiliates',
    'Shopify',
    ARRAY['shopify plus', 'shopify basic', 'shopify advanced']::text[],
    'E-commerce',
    'https://shopify.pxf.io/c/7062841/1424184/13624',
    'cpa',
    'up to $150/merchant',
    'Impact Radius program. Publisher ID: 7062841. Username: canfieldjuan.',
    true
) ON CONFLICT ((lower(product_name))) DO NOTHING;

INSERT INTO affiliate_partners (
    name, product_name, product_aliases, category, affiliate_url,
    commission_type, commission_value, notes, enabled
) VALUES (
    'HelpDesk',
    'HelpDesk',
    ARRAY['helpdesk.com']::text[],
    'Helpdesk',
    'https://www.helpdesk.com/?a=OWvKUHFvg&utm_campaign=pp_helpdesk-default&utm_source=PP',
    'recurring',
    '20% recurring',
    'PartnerStack program. Affiliate ID: OWvKUHFvg.',
    true
) ON CONFLICT ((lower(product_name))) DO NOTHING;

INSERT INTO affiliate_partners (
    name, product_name, product_aliases, category, affiliate_url,
    commission_type, commission_value, notes, enabled
) VALUES (
    'Monday.com',
    'Monday.com',
    ARRAY['monday', 'monday CRM', 'monday work OS']::text[],
    'Project Management',
    'https://try.monday.com/1p7bntdd5bui',
    'rev_share',
    '$100/signup',
    NULL,
    true
) ON CONFLICT ((lower(product_name))) DO NOTHING;
