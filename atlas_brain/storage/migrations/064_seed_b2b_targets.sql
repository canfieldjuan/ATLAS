-- Seed multi-vendor B2B scrape targets
-- Migration 064
--
-- Populates b2b_scrape_targets for structured review sites (G2, Capterra,
-- TrustRadius) across CRM, Project Management, Helpdesk, and Marketing
-- Automation categories.  Social media sources seeded disabled until
-- relevance filter is verified working.
--
-- Idempotent: ON CONFLICT (source, product_slug) DO NOTHING.

-- =====================================================================
-- CRM (priority 10)
-- =====================================================================

-- Salesforce
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, priority, max_pages, metadata)
VALUES
    ('g2',          'Salesforce', 'Sales Cloud',  'salesforce-crm',       'CRM', 10, 5, '{}'),
    ('capterra',    'Salesforce', 'Sales Cloud',  '15/salesforce',        'CRM', 10, 5, '{}'),
    ('trustradius', 'Salesforce', 'Sales Cloud',  'salesforce',           'CRM', 10, 5, '{}'),
    ('reddit',      'Salesforce', 'Sales Cloud',  'salesforce-reddit',    'CRM', 10, 3, '{"subreddits": "salesforce,sysadmin"}'),
    ('hackernews',  'Salesforce', 'Sales Cloud',  'salesforce-hn',        'CRM', 10, 3, '{}'),
    ('github',      'Salesforce', 'Sales Cloud',  'salesforce-github',    'CRM', 10, 3, '{}'),
    ('rss',         'Salesforce', 'Sales Cloud',  'salesforce-rss',       'CRM', 10, 3, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;

-- Disable social media sources for Salesforce (relevance filter must be verified first)
UPDATE b2b_scrape_targets
SET enabled = false
WHERE vendor_name = 'Salesforce'
  AND source IN ('reddit', 'hackernews', 'github', 'rss');

-- Zoho CRM
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, priority, max_pages, metadata)
VALUES
    ('g2',          'Zoho CRM', 'Zoho CRM', 'zoho-crm',           'CRM', 10, 5, '{}'),
    ('capterra',    'Zoho CRM', 'Zoho CRM', '104024/zoho-crm',    'CRM', 10, 5, '{}'),
    ('trustradius', 'Zoho CRM', 'Zoho CRM', 'zoho-crm',           'CRM', 10, 5, '{}'),
    ('reddit',      'Zoho CRM', 'Zoho CRM', 'zoho-crm-reddit',    'CRM', 10, 3, '{"subreddits": "zoho,smallbusiness"}'),
    ('hackernews',  'Zoho CRM', 'Zoho CRM', 'zoho-crm-hn',        'CRM', 10, 3, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;

UPDATE b2b_scrape_targets
SET enabled = false
WHERE vendor_name = 'Zoho CRM'
  AND source IN ('reddit', 'hackernews');

-- Freshsales
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, priority, max_pages, metadata)
VALUES
    ('g2',          'Freshsales', 'Freshsales', 'freshsales',           'CRM', 10, 5, '{}'),
    ('capterra',    'Freshsales', 'Freshsales', '168020/freshsales',    'CRM', 10, 5, '{}'),
    ('trustradius', 'Freshsales', 'Freshsales', 'freshsales',           'CRM', 10, 5, '{}'),
    ('reddit',      'Freshsales', 'Freshsales', 'freshsales-reddit',    'CRM', 10, 3, '{"subreddits": "sysadmin,smallbusiness"}')
ON CONFLICT (source, product_slug) DO NOTHING;

UPDATE b2b_scrape_targets
SET enabled = false
WHERE vendor_name = 'Freshsales'
  AND source IN ('reddit');

-- Copper
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, priority, max_pages, metadata)
VALUES
    ('g2',          'Copper', 'Copper CRM', 'copper',           'CRM', 10, 5, '{}'),
    ('capterra',    'Copper', 'Copper CRM', '166248/copper',    'CRM', 10, 5, '{}'),
    ('trustradius', 'Copper', 'Copper CRM', 'copper',            'CRM', 10, 5, '{}'),
    ('reddit',      'Copper', 'Copper CRM', 'copper-reddit',    'CRM', 10, 3, '{"subreddits": "sysadmin,smallbusiness"}')
ON CONFLICT (source, product_slug) DO NOTHING;

UPDATE b2b_scrape_targets
SET enabled = false
WHERE vendor_name = 'Copper'
  AND source IN ('reddit');

-- =====================================================================
-- Project Management (priority 5)
-- =====================================================================

-- Asana
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, priority, max_pages, metadata)
VALUES
    ('g2',          'Asana', 'Asana', 'asana',           'Project Management', 5, 5, '{}'),
    ('capterra',    'Asana', 'Asana', '148682/asana',    'Project Management', 5, 5, '{}'),
    ('trustradius', 'Asana', 'Asana', 'asana',           'Project Management', 5, 5, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;

-- ClickUp
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, priority, max_pages, metadata)
VALUES
    ('g2',          'ClickUp', 'ClickUp', 'clickup',           'Project Management', 5, 5, '{}'),
    ('capterra',    'ClickUp', 'ClickUp', '212498/clickup',    'Project Management', 5, 5, '{}'),
    ('trustradius', 'ClickUp', 'ClickUp', 'clickup',           'Project Management', 5, 5, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;

-- Wrike
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, priority, max_pages, metadata)
VALUES
    ('g2',          'Wrike', 'Wrike', 'wrike',           'Project Management', 5, 5, '{}'),
    ('capterra',    'Wrike', 'Wrike', '137632/wrike',    'Project Management', 5, 5, '{}'),
    ('trustradius', 'Wrike', 'Wrike', 'wrike',           'Project Management', 5, 5, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;

-- Basecamp
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, priority, max_pages, metadata)
VALUES
    ('g2',          'Basecamp', 'Basecamp', 'basecamp',           'Project Management', 5, 5, '{}'),
    ('capterra',    'Basecamp', 'Basecamp', '131795/basecamp',    'Project Management', 5, 5, '{}'),
    ('trustradius', 'Basecamp', 'Basecamp', 'basecamp',           'Project Management', 5, 5, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;

-- =====================================================================
-- Helpdesk (priority 5)
-- =====================================================================

-- Zendesk
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, priority, max_pages, metadata)
VALUES
    ('g2',          'Zendesk', 'Zendesk', 'zendesk',           'Helpdesk', 5, 5, '{}'),
    ('capterra',    'Zendesk', 'Zendesk', '154008/zendesk',    'Helpdesk', 5, 5, '{}'),
    ('trustradius', 'Zendesk', 'Zendesk', 'zendesk',           'Helpdesk', 5, 5, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;

-- Freshdesk
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, priority, max_pages, metadata)
VALUES
    ('g2',          'Freshdesk', 'Freshdesk', 'freshdesk',           'Helpdesk', 5, 5, '{}'),
    ('capterra',    'Freshdesk', 'Freshdesk', '134038/freshdesk',    'Helpdesk', 5, 5, '{}'),
    ('trustradius', 'Freshdesk', 'Freshdesk', 'freshdesk',           'Helpdesk', 5, 5, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;

-- Intercom
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, priority, max_pages, metadata)
VALUES
    ('g2',          'Intercom', 'Intercom', 'intercom',           'Helpdesk', 5, 5, '{}'),
    ('capterra',    'Intercom', 'Intercom', '143231/intercom',    'Helpdesk', 5, 5, '{}'),
    ('trustradius', 'Intercom', 'Intercom', 'intercom',           'Helpdesk', 5, 5, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;

-- =====================================================================
-- Marketing Automation (priority 3)
-- =====================================================================

-- Mailchimp
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, priority, max_pages, metadata)
VALUES
    ('g2',          'Mailchimp', 'Mailchimp', 'mailchimp',           'Marketing Automation', 3, 5, '{}'),
    ('capterra',    'Mailchimp', 'Mailchimp', '152514/mailchimp',    'Marketing Automation', 3, 5, '{}'),
    ('trustradius', 'Mailchimp', 'Mailchimp', 'mailchimp',           'Marketing Automation', 3, 5, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;

-- ActiveCampaign
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, priority, max_pages, metadata)
VALUES
    ('g2',          'ActiveCampaign', 'ActiveCampaign', 'activecampaign',           'Marketing Automation', 3, 5, '{}'),
    ('capterra',    'ActiveCampaign', 'ActiveCampaign', '155584/activecampaign',    'Marketing Automation', 3, 5, '{}'),
    ('trustradius', 'ActiveCampaign', 'ActiveCampaign', 'activecampaign',           'Marketing Automation', 3, 5, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;
