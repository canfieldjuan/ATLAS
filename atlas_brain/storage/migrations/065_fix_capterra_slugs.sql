-- Fix Capterra product slugs: all 13 had wrong product IDs from seed migration 064.
-- Verified correct IDs from capterra.com search results (Feb 2026).
--
-- The unique constraint is (source, product_slug) so we update product_slug directly.
-- Idempotent: WHERE clause matches old slugs only.

UPDATE b2b_scrape_targets SET product_slug = '61368/Salesforce'
WHERE source = 'capterra' AND product_slug = '15/salesforce';

UPDATE b2b_scrape_targets SET product_slug = '155928/Zoho-CRM'
WHERE source = 'capterra' AND product_slug = '104024/zoho-crm';

UPDATE b2b_scrape_targets SET product_slug = '155563/Freshsales'
WHERE source = 'capterra' AND product_slug = '168020/freshsales';

UPDATE b2b_scrape_targets SET product_slug = '141642/Copper'
WHERE source = 'capterra' AND product_slug = '166248/copper';

UPDATE b2b_scrape_targets SET product_slug = '184581/Asana-PM'
WHERE source = 'capterra' AND product_slug = '148682/asana';

UPDATE b2b_scrape_targets SET product_slug = '158833/ClickUp'
WHERE source = 'capterra' AND product_slug = '212498/clickup';

UPDATE b2b_scrape_targets SET product_slug = '76113/Wrike'
WHERE source = 'capterra' AND product_slug = '137632/wrike';

UPDATE b2b_scrape_targets SET product_slug = '56808/Basecamp'
WHERE source = 'capterra' AND product_slug = '131795/basecamp';

UPDATE b2b_scrape_targets SET product_slug = '164283/Zendesk'
WHERE source = 'capterra' AND product_slug = '154008/zendesk';

UPDATE b2b_scrape_targets SET product_slug = '124981/Freshdesk'
WHERE source = 'capterra' AND product_slug = '134038/freshdesk';

UPDATE b2b_scrape_targets SET product_slug = '134347/Intercom'
WHERE source = 'capterra' AND product_slug = '143231/intercom';

UPDATE b2b_scrape_targets SET product_slug = '110228/MailChimp'
WHERE source = 'capterra' AND product_slug = '152514/mailchimp';

UPDATE b2b_scrape_targets SET product_slug = '79367/ActiveCampaign'
WHERE source = 'capterra' AND product_slug = '155584/activecampaign';

-- Also reset last_scrape_status for Salesforce/capterra so it gets retried
-- (it was marked 'blocked' from an attempt with the wrong slug)
UPDATE b2b_scrape_targets
SET last_scrape_status = NULL, last_scraped_at = NULL, last_scrape_reviews = 0
WHERE source = 'capterra' AND product_slug = '61368/Salesforce';
