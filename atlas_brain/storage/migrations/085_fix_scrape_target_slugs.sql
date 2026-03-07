-- Fix incorrect product_slugs across all sources.
-- Migration 085
--
-- Corrects fabricated/guessed slugs seeded in migrations 064, 082, and
-- seed_phase2_sources.sql.  Verified against live site URLs (Mar 2026).
--
-- Safe: each UPDATE only fires when old slug still exists AND new slug
-- does not already exist (avoids UNIQUE constraint violations).

-- =====================================================================
-- G2: 17 broken slugs (vendor renames, missing suffixes)
-- URL: g2.com/products/{slug}/reviews
-- =====================================================================

UPDATE b2b_scrape_targets SET product_slug = 'salesforce-salesforce-sales-cloud', updated_at = NOW()
WHERE source = 'g2' AND product_slug = 'salesforce-crm'
AND NOT EXISTS (SELECT 1 FROM b2b_scrape_targets WHERE source = 'g2' AND product_slug = 'salesforce-salesforce-sales-cloud');

UPDATE b2b_scrape_targets SET product_slug = 'zendesk-support-suite', updated_at = NOW()
WHERE source = 'g2' AND product_slug = 'zendesk'
AND NOT EXISTS (SELECT 1 FROM b2b_scrape_targets WHERE source = 'g2' AND product_slug = 'zendesk-support-suite');

UPDATE b2b_scrape_targets SET product_slug = 'intuit-mailchimp-email-marketing', updated_at = NOW()
WHERE source = 'g2' AND product_slug = 'mailchimp'
AND NOT EXISTS (SELECT 1 FROM b2b_scrape_targets WHERE source = 'g2' AND product_slug = 'intuit-mailchimp-email-marketing');

UPDATE b2b_scrape_targets SET product_slug = 'insightly-crm', updated_at = NOW()
WHERE source = 'g2' AND product_slug = 'insightly'
AND NOT EXISTS (SELECT 1 FROM b2b_scrape_targets WHERE source = 'g2' AND product_slug = 'insightly-crm');

UPDATE b2b_scrape_targets SET product_slug = 'teamwork-com', updated_at = NOW()
WHERE source = 'g2' AND product_slug = 'teamwork'
AND NOT EXISTS (SELECT 1 FROM b2b_scrape_targets WHERE source = 'g2' AND product_slug = 'teamwork-com');

UPDATE b2b_scrape_targets SET product_slug = 'happyfox-help-desk', updated_at = NOW()
WHERE source = 'g2' AND product_slug = 'happyfox'
AND NOT EXISTS (SELECT 1 FROM b2b_scrape_targets WHERE source = 'g2' AND product_slug = 'happyfox-help-desk');

UPDATE b2b_scrape_targets SET product_slug = 'amazon-aws-platform', updated_at = NOW()
WHERE source = 'g2' AND product_slug = 'amazon-web-services'
AND NOT EXISTS (SELECT 1 FROM b2b_scrape_targets WHERE source = 'g2' AND product_slug = 'amazon-aws-platform');

UPDATE b2b_scrape_targets SET product_slug = 'microsoft-microsoft-azure', updated_at = NOW()
WHERE source = 'g2' AND product_slug = 'microsoft-azure'
AND NOT EXISTS (SELECT 1 FROM b2b_scrape_targets WHERE source = 'g2' AND product_slug = 'microsoft-microsoft-azure');

UPDATE b2b_scrape_targets SET product_slug = 'google-cloud', updated_at = NOW()
WHERE source = 'g2' AND product_slug = 'google-cloud-platform'
AND NOT EXISTS (SELECT 1 FROM b2b_scrape_targets WHERE source = 'g2' AND product_slug = 'google-cloud');

UPDATE b2b_scrape_targets SET product_slug = 'crowdstrike-falcon-endpoint-protection-platform', updated_at = NOW()
WHERE source = 'g2' AND product_slug = 'crowdstrike-falcon'
AND NOT EXISTS (SELECT 1 FROM b2b_scrape_targets WHERE source = 'g2' AND product_slug = 'crowdstrike-falcon-endpoint-protection-platform');

UPDATE b2b_scrape_targets SET product_slug = 'sentinelone-singularity-endpoint', updated_at = NOW()
WHERE source = 'g2' AND product_slug = 'sentinelone'
AND NOT EXISTS (SELECT 1 FROM b2b_scrape_targets WHERE source = 'g2' AND product_slug = 'sentinelone-singularity-endpoint');

UPDATE b2b_scrape_targets SET product_slug = 'fortinet-firewalls', updated_at = NOW()
WHERE source = 'g2' AND product_slug = 'fortinet-fortigate'
AND NOT EXISTS (SELECT 1 FROM b2b_scrape_targets WHERE source = 'g2' AND product_slug = 'fortinet-firewalls');

UPDATE b2b_scrape_targets SET product_slug = 'workday-hcm', updated_at = NOW()
WHERE source = 'g2' AND product_slug = 'workday'
AND NOT EXISTS (SELECT 1 FROM b2b_scrape_targets WHERE source = 'g2' AND product_slug = 'workday-hcm');

UPDATE b2b_scrape_targets SET product_slug = 'microsoft-microsoft-power-bi', updated_at = NOW()
WHERE source = 'g2' AND product_slug = 'microsoft-power-bi'
AND NOT EXISTS (SELECT 1 FROM b2b_scrape_targets WHERE source = 'g2' AND product_slug = 'microsoft-microsoft-power-bi');

UPDATE b2b_scrape_targets SET product_slug = 'zoom-workplace', updated_at = NOW()
WHERE source = 'g2' AND product_slug = 'zoom'
AND NOT EXISTS (SELECT 1 FROM b2b_scrape_targets WHERE source = 'g2' AND product_slug = 'zoom-workplace');

UPDATE b2b_scrape_targets SET product_slug = 'ringex', updated_at = NOW()
WHERE source = 'g2' AND product_slug = 'ringcentral'
AND NOT EXISTS (SELECT 1 FROM b2b_scrape_targets WHERE source = 'g2' AND product_slug = 'ringex');

UPDATE b2b_scrape_targets SET product_slug = 'magento-open-source', updated_at = NOW()
WHERE source = 'g2' AND product_slug = 'magento'
AND NOT EXISTS (SELECT 1 FROM b2b_scrape_targets WHERE source = 'g2' AND product_slug = 'magento-open-source');

-- =====================================================================
-- CAPTERRA: 13 broken slugs in migration 064 (wrong numeric IDs)
-- URL: capterra.com/p/{id}/{name}/reviews/
-- Migration 082 IDs are correct; only 064 needs fixing.
-- =====================================================================

UPDATE b2b_scrape_targets SET product_slug = '61368/Salesforce', updated_at = NOW()
WHERE source = 'capterra' AND product_slug = '15/salesforce'
AND NOT EXISTS (SELECT 1 FROM b2b_scrape_targets WHERE source = 'capterra' AND product_slug = '61368/Salesforce');

UPDATE b2b_scrape_targets SET product_slug = '155928/Zoho-CRM', updated_at = NOW()
WHERE source = 'capterra' AND product_slug = '104024/zoho-crm'
AND NOT EXISTS (SELECT 1 FROM b2b_scrape_targets WHERE source = 'capterra' AND product_slug = '155928/Zoho-CRM');

UPDATE b2b_scrape_targets SET product_slug = '155563/Freshsales', updated_at = NOW()
WHERE source = 'capterra' AND product_slug = '168020/freshsales'
AND NOT EXISTS (SELECT 1 FROM b2b_scrape_targets WHERE source = 'capterra' AND product_slug = '155563/Freshsales');

UPDATE b2b_scrape_targets SET product_slug = '141642/Copper', updated_at = NOW()
WHERE source = 'capterra' AND product_slug = '166248/copper'
AND NOT EXISTS (SELECT 1 FROM b2b_scrape_targets WHERE source = 'capterra' AND product_slug = '141642/Copper');

UPDATE b2b_scrape_targets SET product_slug = '184581/Asana-PM', updated_at = NOW()
WHERE source = 'capterra' AND product_slug = '148682/asana'
AND NOT EXISTS (SELECT 1 FROM b2b_scrape_targets WHERE source = 'capterra' AND product_slug = '184581/Asana-PM');

UPDATE b2b_scrape_targets SET product_slug = '158833/ClickUp', updated_at = NOW()
WHERE source = 'capterra' AND product_slug = '212498/clickup'
AND NOT EXISTS (SELECT 1 FROM b2b_scrape_targets WHERE source = 'capterra' AND product_slug = '158833/ClickUp');

UPDATE b2b_scrape_targets SET product_slug = '76113/Wrike', updated_at = NOW()
WHERE source = 'capterra' AND product_slug = '137632/wrike'
AND NOT EXISTS (SELECT 1 FROM b2b_scrape_targets WHERE source = 'capterra' AND product_slug = '76113/Wrike');

UPDATE b2b_scrape_targets SET product_slug = '56808/Basecamp', updated_at = NOW()
WHERE source = 'capterra' AND product_slug = '131795/basecamp'
AND NOT EXISTS (SELECT 1 FROM b2b_scrape_targets WHERE source = 'capterra' AND product_slug = '56808/Basecamp');

UPDATE b2b_scrape_targets SET product_slug = '164283/Zendesk', updated_at = NOW()
WHERE source = 'capterra' AND product_slug = '154008/zendesk'
AND NOT EXISTS (SELECT 1 FROM b2b_scrape_targets WHERE source = 'capterra' AND product_slug = '164283/Zendesk');

UPDATE b2b_scrape_targets SET product_slug = '124981/Freshdesk', updated_at = NOW()
WHERE source = 'capterra' AND product_slug = '134038/freshdesk'
AND NOT EXISTS (SELECT 1 FROM b2b_scrape_targets WHERE source = 'capterra' AND product_slug = '124981/Freshdesk');

UPDATE b2b_scrape_targets SET product_slug = '134347/Intercom', updated_at = NOW()
WHERE source = 'capterra' AND product_slug = '143231/intercom'
AND NOT EXISTS (SELECT 1 FROM b2b_scrape_targets WHERE source = 'capterra' AND product_slug = '134347/Intercom');

UPDATE b2b_scrape_targets SET product_slug = '110228/MailChimp', updated_at = NOW()
WHERE source = 'capterra' AND product_slug = '152514/mailchimp'
AND NOT EXISTS (SELECT 1 FROM b2b_scrape_targets WHERE source = 'capterra' AND product_slug = '110228/MailChimp');

UPDATE b2b_scrape_targets SET product_slug = '79367/ActiveCampaign', updated_at = NOW()
WHERE source = 'capterra' AND product_slug = '155584/activecampaign'
AND NOT EXISTS (SELECT 1 FROM b2b_scrape_targets WHERE source = 'capterra' AND product_slug = '79367/ActiveCampaign');

-- =====================================================================
-- TRUSTRADIUS: 3 broken slugs (wrong product slug format)
-- URL: trustradius.com/products/{slug}/reviews
-- =====================================================================

UPDATE b2b_scrape_targets SET product_slug = 'notion', updated_at = NOW()
WHERE source = 'trustradius' AND product_slug = 'notion-so'
AND NOT EXISTS (SELECT 1 FROM b2b_scrape_targets WHERE source = 'trustradius' AND product_slug = 'notion');

UPDATE b2b_scrape_targets SET product_slug = 'monday', updated_at = NOW()
WHERE source = 'trustradius' AND product_slug = 'monday-com'
AND NOT EXISTS (SELECT 1 FROM b2b_scrape_targets WHERE source = 'trustradius' AND product_slug = 'monday');

UPDATE b2b_scrape_targets SET product_slug = 'palo-alto-networks-cortex-xdr', updated_at = NOW()
WHERE source = 'trustradius' AND product_slug = 'palo-alto-networks-cortex'
AND NOT EXISTS (SELECT 1 FROM b2b_scrape_targets WHERE source = 'trustradius' AND product_slug = 'palo-alto-networks-cortex-xdr');

-- =====================================================================
-- GARTNER: Fix market category slugs (7/10 wrong)
-- URL: gartner.com/reviews/market/{market}/vendor/{vendor}/reviews
-- Slug format: {market}/{vendor}
-- =====================================================================

-- CRM: crm -> sales-force-automation-platforms
UPDATE b2b_scrape_targets SET product_slug = 'sales-force-automation-platforms/' || split_part(product_slug, '/', 2), updated_at = NOW()
WHERE source = 'gartner' AND product_slug LIKE 'crm/%';

-- Helpdesk: it-service-management-tools -> crm-customer-engagement-center
UPDATE b2b_scrape_targets SET product_slug = 'crm-customer-engagement-center/' || split_part(product_slug, '/', 2), updated_at = NOW()
WHERE source = 'gartner' AND product_slug LIKE 'it-service-management-tools/%';

-- Project Management: project-and-portfolio-management -> collaborative-work-management
UPDATE b2b_scrape_targets SET product_slug = 'collaborative-work-management/' || split_part(product_slug, '/', 2), updated_at = NOW()
WHERE source = 'gartner' AND product_slug LIKE 'project-and-portfolio-management/%';

-- Marketing Automation: multichannel-marketing-hubs -> email-marketing
UPDATE b2b_scrape_targets SET product_slug = 'email-marketing/' || split_part(product_slug, '/', 2), updated_at = NOW()
WHERE source = 'gartner' AND product_slug LIKE 'multichannel-marketing-hubs/%';

-- Cloud Infrastructure: OK as-is (cloud-infrastructure-and-platform-services is correct)

-- HR/HCM: cloud-hcm-suites-for-midmarket-and-large-enterprises -> cloud-hcm-suites-for-1000-employees
UPDATE b2b_scrape_targets SET product_slug = 'cloud-hcm-suites-for-1000-employees/' || split_part(product_slug, '/', 2), updated_at = NOW()
WHERE source = 'gartner' AND product_slug LIKE 'cloud-hcm-suites-for-midmarket-and-large-enterprises/%';

-- Data/Analytics: analytics-and-business-intelligence-platforms -> analytics-business-intelligence-platforms
UPDATE b2b_scrape_targets SET product_slug = 'analytics-business-intelligence-platforms/' || split_part(product_slug, '/', 2), updated_at = NOW()
WHERE source = 'gartner' AND product_slug LIKE 'analytics-and-business-intelligence-platforms/%';

-- Cybersecurity, E-commerce, Communication: already correct market slugs

-- Fix individual Gartner vendor slugs that differ from G2 slugs
UPDATE b2b_scrape_targets SET product_slug = 'sales-force-automation-platforms/salesforce', updated_at = NOW()
WHERE source = 'gartner' AND product_slug = 'sales-force-automation-platforms/salesforce-crm';

-- =====================================================================
-- GETAPP: Fix category slugs (9/10 wrong)
-- URL: getapp.com/software/{category}/a/{product}/reviews/
-- Slug format: {category}/a/{product}
-- Note: products in same vertical may use different categories on GetApp
-- =====================================================================

-- CRM: crm-software -> customer-management-software
UPDATE b2b_scrape_targets SET product_slug = regexp_replace(product_slug, '^crm-software/', 'customer-management-software/'), updated_at = NOW()
WHERE source = 'getapp' AND product_slug LIKE 'crm-software/%';

-- Helpdesk: help-desk-software -> customer-service-support-software
UPDATE b2b_scrape_targets SET product_slug = regexp_replace(product_slug, '^help-desk-software/', 'customer-service-support-software/'), updated_at = NOW()
WHERE source = 'getapp' AND product_slug LIKE 'help-desk-software/%';

-- Project Management: project-management-software -> project-management-planning-software
UPDATE b2b_scrape_targets SET product_slug = regexp_replace(product_slug, '^project-management-software/', 'project-management-planning-software/'), updated_at = NOW()
WHERE source = 'getapp' AND product_slug LIKE 'project-management-software/%';

-- Marketing Automation: marketing-automation-software -> marketing-software
UPDATE b2b_scrape_targets SET product_slug = regexp_replace(product_slug, '^marketing-automation-software/', 'marketing-software/'), updated_at = NOW()
WHERE source = 'getapp' AND product_slug LIKE 'marketing-automation-software/%';

-- Cloud Infrastructure: cloud-management-software -> it-management-software
UPDATE b2b_scrape_targets SET product_slug = regexp_replace(product_slug, '^cloud-management-software/', 'it-management-software/'), updated_at = NOW()
WHERE source = 'getapp' AND product_slug LIKE 'cloud-management-software/%';

-- E-commerce: e-commerce-software -> website-ecommerce-software
UPDATE b2b_scrape_targets SET product_slug = regexp_replace(product_slug, '^e-commerce-software/', 'website-ecommerce-software/'), updated_at = NOW()
WHERE source = 'getapp' AND product_slug LIKE 'e-commerce-software/%';

-- HR/HCM: human-resource-software -> hr-employee-management-software
UPDATE b2b_scrape_targets SET product_slug = regexp_replace(product_slug, '^human-resource-software/', 'hr-employee-management-software/'), updated_at = NOW()
WHERE source = 'getapp' AND product_slug LIKE 'human-resource-software/%';

-- Cybersecurity: endpoint-protection-software -> security-software
UPDATE b2b_scrape_targets SET product_slug = regexp_replace(product_slug, '^endpoint-protection-software/', 'security-software/'), updated_at = NOW()
WHERE source = 'getapp' AND product_slug LIKE 'endpoint-protection-software/%';

-- Communication: collaboration-software -> already correct for Slack/Teams
-- But Zoom needs: it-communications-software
UPDATE b2b_scrape_targets SET product_slug = 'it-communications-software/a/zoom', updated_at = NOW()
WHERE source = 'getapp' AND product_slug = 'collaboration-software/a/zoom';

-- Data/Analytics: business-intelligence-software -> business-intelligence-analytics-software
UPDATE b2b_scrape_targets SET product_slug = regexp_replace(product_slug, '^business-intelligence-software/', 'business-intelligence-analytics-software/'), updated_at = NOW()
WHERE source = 'getapp' AND product_slug LIKE 'business-intelligence-software/%';

-- Fix individual GetApp product slugs
UPDATE b2b_scrape_targets SET product_slug = regexp_replace(product_slug, '/a/salesforce-crm$', '/a/salesforce'), updated_at = NOW()
WHERE source = 'getapp' AND product_slug LIKE '%/a/salesforce-crm';

UPDATE b2b_scrape_targets SET product_slug = regexp_replace(product_slug, '/a/microsoft-power-bi$', '/a/power-bi'), updated_at = NOW()
WHERE source = 'getapp' AND product_slug LIKE '%/a/microsoft-power-bi';

-- =====================================================================
-- PEERSPOT: 2 broken slugs
-- URL: peerspot.com/products/{slug}-reviews
-- =====================================================================

UPDATE b2b_scrape_targets SET product_slug = 'salesforce-sales-cloud', updated_at = NOW()
WHERE source = 'peerspot' AND product_slug = 'salesforce-crm'
AND NOT EXISTS (SELECT 1 FROM b2b_scrape_targets WHERE source = 'peerspot' AND product_slug = 'salesforce-sales-cloud');

UPDATE b2b_scrape_targets SET product_slug = 'cortex-xdr-by-palo-alto-networks', updated_at = NOW()
WHERE source = 'peerspot' AND product_slug = 'palo-alto-networks-cortex-xdr'
AND NOT EXISTS (SELECT 1 FROM b2b_scrape_targets WHERE source = 'peerspot' AND product_slug = 'cortex-xdr-by-palo-alto-networks');

-- =====================================================================
-- TRUSTPILOT: 4 broken slugs (path-based slugs 404 on Trustpilot)
-- URL: trustpilot.com/review/{domain}
-- Trustpilot profiles are per-company-domain, not per-product.
-- Sub-paths like zoho.com/desk don't work.
-- =====================================================================

-- Trustpilot is per-company-domain. Sub-paths (zoho.com/desk) 404.
-- If the parent domain already exists as a row, disable the sub-product row.
-- If the parent domain does NOT exist, update the slug to the parent domain.

-- zoho.com/desk -> zoho.com (Zoho CRM row already has zoho.com)
UPDATE b2b_scrape_targets SET enabled = false, updated_at = NOW()
WHERE source = 'trustpilot' AND product_slug = 'zoho.com/desk'
AND EXISTS (SELECT 1 FROM b2b_scrape_targets WHERE source = 'trustpilot' AND product_slug = 'zoho.com');

UPDATE b2b_scrape_targets SET product_slug = 'zoho.com', updated_at = NOW()
WHERE source = 'trustpilot' AND product_slug = 'zoho.com/desk';

-- hubspot.com/marketing -> hubspot.com (HubSpot Service Hub row already has hubspot.com)
UPDATE b2b_scrape_targets SET enabled = false, updated_at = NOW()
WHERE source = 'trustpilot' AND product_slug = 'hubspot.com/marketing'
AND EXISTS (SELECT 1 FROM b2b_scrape_targets WHERE source = 'trustpilot' AND product_slug = 'hubspot.com');

UPDATE b2b_scrape_targets SET product_slug = 'hubspot.com', updated_at = NOW()
WHERE source = 'trustpilot' AND product_slug = 'hubspot.com/marketing';

-- microsoft.com/teams -> microsoft.com
UPDATE b2b_scrape_targets SET product_slug = 'microsoft.com', updated_at = NOW()
WHERE source = 'trustpilot' AND product_slug = 'microsoft.com/teams'
AND NOT EXISTS (SELECT 1 FROM b2b_scrape_targets WHERE source = 'trustpilot' AND product_slug = 'microsoft.com');

UPDATE b2b_scrape_targets SET enabled = false, updated_at = NOW()
WHERE source = 'trustpilot' AND product_slug = 'microsoft.com/teams';

-- powerbi.microsoft.com -> microsoft.com (may already exist from Teams fix above)
UPDATE b2b_scrape_targets SET enabled = false, updated_at = NOW()
WHERE source = 'trustpilot' AND product_slug = 'powerbi.microsoft.com';

-- =====================================================================
-- PRODUCTHUNT: 3 broken slugs
-- URL: producthunt.com/products/{slug}/reviews
-- =====================================================================

UPDATE b2b_scrape_targets SET product_slug = 'salesforce-sfdc', updated_at = NOW()
WHERE source = 'producthunt' AND product_slug = 'salesforce'
AND NOT EXISTS (SELECT 1 FROM b2b_scrape_targets WHERE source = 'producthunt' AND product_slug = 'salesforce-sfdc');

UPDATE b2b_scrape_targets SET product_slug = 'monday-com', updated_at = NOW()
WHERE source = 'producthunt' AND product_slug = 'monday'
AND NOT EXISTS (SELECT 1 FROM b2b_scrape_targets WHERE source = 'producthunt' AND product_slug = 'monday-com');

UPDATE b2b_scrape_targets SET product_slug = 'zoho', updated_at = NOW()
WHERE source = 'producthunt' AND product_slug = 'zoho-crm'
AND NOT EXISTS (SELECT 1 FROM b2b_scrape_targets WHERE source = 'producthunt' AND product_slug = 'zoho');
