-- Expand B2B scrape target coverage: 42 new vendors across 10 categories
-- + backfill social sources for 9 existing vendors that lack them.
-- Migration 082
--
-- Grows from 13 vendors / 4 categories to 55 vendors / 10 categories.
-- All social sources (reddit, hackernews, github, rss) seeded DISABLED
-- until relevance filter is verified working (same pattern as migration 064).
-- Capterra IDs verified from capterra.com (Mar 2026).
--
-- Idempotent: ON CONFLICT (source, product_slug) DO NOTHING.

-- =====================================================================
-- BACKFILL: social sources for existing vendors that only have
-- g2 + capterra + trustradius (PM, Helpdesk, Marketing Automation)
-- =====================================================================

-- Asana (social backfill)
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, priority, max_pages, metadata)
VALUES
    ('reddit',     'Asana', 'Asana', 'asana-reddit',     'Project Management', 5, 3, '{"subreddits": "asana,projectmanagement"}'),
    ('hackernews', 'Asana', 'Asana', 'asana-hn',         'Project Management', 5, 3, '{}'),
    ('github',     'Asana', 'Asana', 'asana-github',     'Project Management', 5, 3, '{}'),
    ('rss',        'Asana', 'Asana', 'asana-rss',        'Project Management', 5, 3, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;

UPDATE b2b_scrape_targets
SET enabled = false
WHERE vendor_name = 'Asana'
  AND source IN ('reddit', 'hackernews', 'github', 'rss');

-- ClickUp (social backfill)
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, priority, max_pages, metadata)
VALUES
    ('reddit',     'ClickUp', 'ClickUp', 'clickup-reddit',     'Project Management', 5, 3, '{"subreddits": "clickup,projectmanagement"}'),
    ('hackernews', 'ClickUp', 'ClickUp', 'clickup-hn',         'Project Management', 5, 3, '{}'),
    ('github',     'ClickUp', 'ClickUp', 'clickup-github',     'Project Management', 5, 3, '{}'),
    ('rss',        'ClickUp', 'ClickUp', 'clickup-rss',        'Project Management', 5, 3, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;

UPDATE b2b_scrape_targets
SET enabled = false
WHERE vendor_name = 'ClickUp'
  AND source IN ('reddit', 'hackernews', 'github', 'rss');

-- Wrike (social backfill)
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, priority, max_pages, metadata)
VALUES
    ('reddit',     'Wrike', 'Wrike', 'wrike-reddit',     'Project Management', 5, 3, '{"subreddits": "projectmanagement,sysadmin"}'),
    ('hackernews', 'Wrike', 'Wrike', 'wrike-hn',         'Project Management', 5, 3, '{}'),
    ('github',     'Wrike', 'Wrike', 'wrike-github',     'Project Management', 5, 3, '{}'),
    ('rss',        'Wrike', 'Wrike', 'wrike-rss',        'Project Management', 5, 3, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;

UPDATE b2b_scrape_targets
SET enabled = false
WHERE vendor_name = 'Wrike'
  AND source IN ('reddit', 'hackernews', 'github', 'rss');

-- Basecamp (social backfill)
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, priority, max_pages, metadata)
VALUES
    ('reddit',     'Basecamp', 'Basecamp', 'basecamp-reddit',     'Project Management', 5, 3, '{"subreddits": "projectmanagement,smallbusiness"}'),
    ('hackernews', 'Basecamp', 'Basecamp', 'basecamp-hn',         'Project Management', 5, 3, '{}'),
    ('github',     'Basecamp', 'Basecamp', 'basecamp-github',     'Project Management', 5, 3, '{}'),
    ('rss',        'Basecamp', 'Basecamp', 'basecamp-rss',        'Project Management', 5, 3, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;

UPDATE b2b_scrape_targets
SET enabled = false
WHERE vendor_name = 'Basecamp'
  AND source IN ('reddit', 'hackernews', 'github', 'rss');

-- Zendesk (social backfill)
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, priority, max_pages, metadata)
VALUES
    ('reddit',     'Zendesk', 'Zendesk', 'zendesk-reddit',     'Helpdesk', 5, 3, '{"subreddits": "zendesk,sysadmin"}'),
    ('hackernews', 'Zendesk', 'Zendesk', 'zendesk-hn',         'Helpdesk', 5, 3, '{}'),
    ('github',     'Zendesk', 'Zendesk', 'zendesk-github',     'Helpdesk', 5, 3, '{}'),
    ('rss',        'Zendesk', 'Zendesk', 'zendesk-rss',        'Helpdesk', 5, 3, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;

UPDATE b2b_scrape_targets
SET enabled = false
WHERE vendor_name = 'Zendesk'
  AND source IN ('reddit', 'hackernews', 'github', 'rss');

-- Freshdesk (social backfill)
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, priority, max_pages, metadata)
VALUES
    ('reddit',     'Freshdesk', 'Freshdesk', 'freshdesk-reddit',     'Helpdesk', 5, 3, '{"subreddits": "sysadmin,helpdesk"}'),
    ('hackernews', 'Freshdesk', 'Freshdesk', 'freshdesk-hn',         'Helpdesk', 5, 3, '{}'),
    ('github',     'Freshdesk', 'Freshdesk', 'freshdesk-github',     'Helpdesk', 5, 3, '{}'),
    ('rss',        'Freshdesk', 'Freshdesk', 'freshdesk-rss',        'Helpdesk', 5, 3, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;

UPDATE b2b_scrape_targets
SET enabled = false
WHERE vendor_name = 'Freshdesk'
  AND source IN ('reddit', 'hackernews', 'github', 'rss');

-- Intercom (social backfill)
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, priority, max_pages, metadata)
VALUES
    ('reddit',     'Intercom', 'Intercom', 'intercom-reddit',     'Helpdesk', 5, 3, '{"subreddits": "sysadmin,startups"}'),
    ('hackernews', 'Intercom', 'Intercom', 'intercom-hn',         'Helpdesk', 5, 3, '{}'),
    ('github',     'Intercom', 'Intercom', 'intercom-github',     'Helpdesk', 5, 3, '{}'),
    ('rss',        'Intercom', 'Intercom', 'intercom-rss',        'Helpdesk', 5, 3, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;

UPDATE b2b_scrape_targets
SET enabled = false
WHERE vendor_name = 'Intercom'
  AND source IN ('reddit', 'hackernews', 'github', 'rss');

-- Mailchimp (social backfill)
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, priority, max_pages, metadata)
VALUES
    ('reddit',     'Mailchimp', 'Mailchimp', 'mailchimp-reddit',     'Marketing Automation', 3, 3, '{"subreddits": "emailmarketing,marketing"}'),
    ('hackernews', 'Mailchimp', 'Mailchimp', 'mailchimp-hn',         'Marketing Automation', 3, 3, '{}'),
    ('github',     'Mailchimp', 'Mailchimp', 'mailchimp-github',     'Marketing Automation', 3, 3, '{}'),
    ('rss',        'Mailchimp', 'Mailchimp', 'mailchimp-rss',        'Marketing Automation', 3, 3, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;

UPDATE b2b_scrape_targets
SET enabled = false
WHERE vendor_name = 'Mailchimp'
  AND source IN ('reddit', 'hackernews', 'github', 'rss');

-- ActiveCampaign (social backfill)
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, priority, max_pages, metadata)
VALUES
    ('reddit',     'ActiveCampaign', 'ActiveCampaign', 'activecampaign-reddit',     'Marketing Automation', 3, 3, '{"subreddits": "emailmarketing,marketing"}'),
    ('hackernews', 'ActiveCampaign', 'ActiveCampaign', 'activecampaign-hn',         'Marketing Automation', 3, 3, '{}'),
    ('github',     'ActiveCampaign', 'ActiveCampaign', 'activecampaign-github',     'Marketing Automation', 3, 3, '{}'),
    ('rss',        'ActiveCampaign', 'ActiveCampaign', 'activecampaign-rss',        'Marketing Automation', 3, 3, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;

UPDATE b2b_scrape_targets
SET enabled = false
WHERE vendor_name = 'ActiveCampaign'
  AND source IN ('reddit', 'hackernews', 'github', 'rss');


-- =====================================================================
-- NEW VENDORS: CRM (priority 10)
-- =====================================================================

-- Pipedrive
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, priority, max_pages, metadata)
VALUES
    ('g2',          'Pipedrive', 'Pipedrive', 'pipedrive',                'CRM', 10, 5, '{}'),
    ('capterra',    'Pipedrive', 'Pipedrive', '132666/Pipedrive',         'CRM', 10, 5, '{}'),
    ('trustradius', 'Pipedrive', 'Pipedrive', 'pipedrive',               'CRM', 10, 5, '{}'),
    ('reddit',      'Pipedrive', 'Pipedrive', 'pipedrive-reddit',        'CRM', 10, 3, '{"subreddits": "sales,smallbusiness"}'),
    ('hackernews',  'Pipedrive', 'Pipedrive', 'pipedrive-hn',            'CRM', 10, 3, '{}'),
    ('github',      'Pipedrive', 'Pipedrive', 'pipedrive-github',        'CRM', 10, 3, '{}'),
    ('rss',         'Pipedrive', 'Pipedrive', 'pipedrive-rss',           'CRM', 10, 3, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;

UPDATE b2b_scrape_targets
SET enabled = false
WHERE vendor_name = 'Pipedrive'
  AND source IN ('reddit', 'hackernews', 'github', 'rss');

-- Close
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, priority, max_pages, metadata)
VALUES
    ('g2',          'Close', 'Close CRM', 'close',                  'CRM', 10, 5, '{}'),
    ('capterra',    'Close', 'Close CRM', '132667/Close-io',        'CRM', 10, 5, '{}'),
    ('trustradius', 'Close', 'Close CRM', 'close',                  'CRM', 10, 5, '{}'),
    ('reddit',      'Close', 'Close CRM', 'close-reddit',           'CRM', 10, 3, '{"subreddits": "sales,startups"}'),
    ('hackernews',  'Close', 'Close CRM', 'close-hn',               'CRM', 10, 3, '{}'),
    ('github',      'Close', 'Close CRM', 'close-github',           'CRM', 10, 3, '{}'),
    ('rss',         'Close', 'Close CRM', 'close-rss',              'CRM', 10, 3, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;

UPDATE b2b_scrape_targets
SET enabled = false
WHERE vendor_name = 'Close'
  AND source IN ('reddit', 'hackernews', 'github', 'rss');

-- Insightly
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, priority, max_pages, metadata)
VALUES
    ('g2',          'Insightly', 'Insightly CRM', 'insightly',                'CRM', 10, 5, '{}'),
    ('capterra',    'Insightly', 'Insightly CRM', '130671/Insightly',         'CRM', 10, 5, '{}'),
    ('trustradius', 'Insightly', 'Insightly CRM', 'insightly',               'CRM', 10, 5, '{}'),
    ('reddit',      'Insightly', 'Insightly CRM', 'insightly-reddit',        'CRM', 10, 3, '{"subreddits": "smallbusiness,sysadmin"}'),
    ('hackernews',  'Insightly', 'Insightly CRM', 'insightly-hn',            'CRM', 10, 3, '{}'),
    ('github',      'Insightly', 'Insightly CRM', 'insightly-github',        'CRM', 10, 3, '{}'),
    ('rss',         'Insightly', 'Insightly CRM', 'insightly-rss',           'CRM', 10, 3, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;

UPDATE b2b_scrape_targets
SET enabled = false
WHERE vendor_name = 'Insightly'
  AND source IN ('reddit', 'hackernews', 'github', 'rss');

-- Nutshell
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, priority, max_pages, metadata)
VALUES
    ('g2',          'Nutshell', 'Nutshell CRM', 'nutshell',                'CRM', 10, 5, '{}'),
    ('capterra',    'Nutshell', 'Nutshell CRM', '144340/Nutshell',         'CRM', 10, 5, '{}'),
    ('trustradius', 'Nutshell', 'Nutshell CRM', 'nutshell',               'CRM', 10, 5, '{}'),
    ('reddit',      'Nutshell', 'Nutshell CRM', 'nutshell-reddit',        'CRM', 10, 3, '{"subreddits": "sales,smallbusiness"}'),
    ('hackernews',  'Nutshell', 'Nutshell CRM', 'nutshell-hn',            'CRM', 10, 3, '{}'),
    ('github',      'Nutshell', 'Nutshell CRM', 'nutshell-github',        'CRM', 10, 3, '{}'),
    ('rss',         'Nutshell', 'Nutshell CRM', 'nutshell-rss',           'CRM', 10, 3, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;

UPDATE b2b_scrape_targets
SET enabled = false
WHERE vendor_name = 'Nutshell'
  AND source IN ('reddit', 'hackernews', 'github', 'rss');


-- =====================================================================
-- NEW VENDORS: Project Management (priority 5)
-- =====================================================================

-- Monday.com
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, priority, max_pages, metadata)
VALUES
    ('g2',          'Monday.com', 'monday.com', 'monday-com',                    'Project Management', 5, 5, '{}'),
    ('capterra',    'Monday.com', 'monday.com', '147657/monday-com',             'Project Management', 5, 5, '{}'),
    ('trustradius', 'Monday.com', 'monday.com', 'monday-com',                    'Project Management', 5, 5, '{}'),
    ('reddit',      'Monday.com', 'monday.com', 'monday-com-reddit',            'Project Management', 5, 3, '{"subreddits": "mondaydotcom,projectmanagement"}'),
    ('hackernews',  'Monday.com', 'monday.com', 'monday-com-hn',               'Project Management', 5, 3, '{}'),
    ('github',      'Monday.com', 'monday.com', 'monday-com-github',           'Project Management', 5, 3, '{}'),
    ('rss',         'Monday.com', 'monday.com', 'monday-com-rss',              'Project Management', 5, 3, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;

UPDATE b2b_scrape_targets
SET enabled = false
WHERE vendor_name = 'Monday.com'
  AND source IN ('reddit', 'hackernews', 'github', 'rss');

-- Smartsheet
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, priority, max_pages, metadata)
VALUES
    ('g2',          'Smartsheet', 'Smartsheet', 'smartsheet',                'Project Management', 5, 5, '{}'),
    ('capterra',    'Smartsheet', 'Smartsheet', '79104/Smartsheet',          'Project Management', 5, 5, '{}'),
    ('trustradius', 'Smartsheet', 'Smartsheet', 'smartsheet',               'Project Management', 5, 5, '{}'),
    ('reddit',      'Smartsheet', 'Smartsheet', 'smartsheet-reddit',        'Project Management', 5, 3, '{"subreddits": "smartsheet,projectmanagement"}'),
    ('hackernews',  'Smartsheet', 'Smartsheet', 'smartsheet-hn',            'Project Management', 5, 3, '{}'),
    ('github',      'Smartsheet', 'Smartsheet', 'smartsheet-github',        'Project Management', 5, 3, '{}'),
    ('rss',         'Smartsheet', 'Smartsheet', 'smartsheet-rss',           'Project Management', 5, 3, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;

UPDATE b2b_scrape_targets
SET enabled = false
WHERE vendor_name = 'Smartsheet'
  AND source IN ('reddit', 'hackernews', 'github', 'rss');

-- Notion
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, priority, max_pages, metadata)
VALUES
    ('g2',          'Notion', 'Notion', 'notion',                'Project Management', 5, 5, '{}'),
    ('capterra',    'Notion', 'Notion', '186596/Notion',         'Project Management', 5, 5, '{}'),
    ('trustradius', 'Notion', 'Notion', 'notion-so',             'Project Management', 5, 5, '{}'),
    ('reddit',      'Notion', 'Notion', 'notion-reddit',         'Project Management', 5, 3, '{"subreddits": "notion,productivity"}'),
    ('hackernews',  'Notion', 'Notion', 'notion-hn',             'Project Management', 5, 3, '{}'),
    ('github',      'Notion', 'Notion', 'notion-github',         'Project Management', 5, 3, '{}'),
    ('rss',         'Notion', 'Notion', 'notion-rss',            'Project Management', 5, 3, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;

UPDATE b2b_scrape_targets
SET enabled = false
WHERE vendor_name = 'Notion'
  AND source IN ('reddit', 'hackernews', 'github', 'rss');

-- Teamwork
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, priority, max_pages, metadata)
VALUES
    ('g2',          'Teamwork', 'Teamwork', 'teamwork',                      'Project Management', 5, 5, '{}'),
    ('capterra',    'Teamwork', 'Teamwork', '120390/Teamwork-Projects',      'Project Management', 5, 5, '{}'),
    ('trustradius', 'Teamwork', 'Teamwork', 'teamwork',                      'Project Management', 5, 5, '{}'),
    ('reddit',      'Teamwork', 'Teamwork', 'teamwork-reddit',              'Project Management', 5, 3, '{"subreddits": "projectmanagement,msp"}'),
    ('hackernews',  'Teamwork', 'Teamwork', 'teamwork-hn',                  'Project Management', 5, 3, '{}'),
    ('github',      'Teamwork', 'Teamwork', 'teamwork-github',              'Project Management', 5, 3, '{}'),
    ('rss',         'Teamwork', 'Teamwork', 'teamwork-rss',                 'Project Management', 5, 3, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;

UPDATE b2b_scrape_targets
SET enabled = false
WHERE vendor_name = 'Teamwork'
  AND source IN ('reddit', 'hackernews', 'github', 'rss');


-- =====================================================================
-- NEW VENDORS: Helpdesk (priority 5)
-- =====================================================================

-- HubSpot Service Hub
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, priority, max_pages, metadata)
VALUES
    ('g2',          'HubSpot Service Hub', 'HubSpot Service Hub', 'hubspot-service-hub',              'Helpdesk', 5, 5, '{}'),
    ('capterra',    'HubSpot Service Hub', 'HubSpot Service Hub', '182476/HubSpot-Service-Hub',       'Helpdesk', 5, 5, '{}'),
    ('trustradius', 'HubSpot Service Hub', 'HubSpot Service Hub', 'hubspot-service-hub',              'Helpdesk', 5, 5, '{}'),
    ('reddit',      'HubSpot Service Hub', 'HubSpot Service Hub', 'hubspot-service-hub-reddit',       'Helpdesk', 5, 3, '{"subreddits": "hubspot,sysadmin"}'),
    ('hackernews',  'HubSpot Service Hub', 'HubSpot Service Hub', 'hubspot-service-hub-hn',           'Helpdesk', 5, 3, '{}'),
    ('github',      'HubSpot Service Hub', 'HubSpot Service Hub', 'hubspot-service-hub-github',       'Helpdesk', 5, 3, '{}'),
    ('rss',         'HubSpot Service Hub', 'HubSpot Service Hub', 'hubspot-service-hub-rss',          'Helpdesk', 5, 3, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;

UPDATE b2b_scrape_targets
SET enabled = false
WHERE vendor_name = 'HubSpot Service Hub'
  AND source IN ('reddit', 'hackernews', 'github', 'rss');

-- Zoho Desk
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, priority, max_pages, metadata)
VALUES
    ('g2',          'Zoho Desk', 'Zoho Desk', 'zoho-desk',                'Helpdesk', 5, 5, '{}'),
    ('capterra',    'Zoho Desk', 'Zoho Desk', '169505/Zoho-Desk',         'Helpdesk', 5, 5, '{}'),
    ('trustradius', 'Zoho Desk', 'Zoho Desk', 'zoho-desk',               'Helpdesk', 5, 5, '{}'),
    ('reddit',      'Zoho Desk', 'Zoho Desk', 'zoho-desk-reddit',        'Helpdesk', 5, 3, '{"subreddits": "zoho,sysadmin"}'),
    ('hackernews',  'Zoho Desk', 'Zoho Desk', 'zoho-desk-hn',            'Helpdesk', 5, 3, '{}'),
    ('github',      'Zoho Desk', 'Zoho Desk', 'zoho-desk-github',        'Helpdesk', 5, 3, '{}'),
    ('rss',         'Zoho Desk', 'Zoho Desk', 'zoho-desk-rss',           'Helpdesk', 5, 3, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;

UPDATE b2b_scrape_targets
SET enabled = false
WHERE vendor_name = 'Zoho Desk'
  AND source IN ('reddit', 'hackernews', 'github', 'rss');

-- Help Scout
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, priority, max_pages, metadata)
VALUES
    ('g2',          'Help Scout', 'Help Scout', 'help-scout',                'Helpdesk', 5, 5, '{}'),
    ('capterra',    'Help Scout', 'Help Scout', '136909/Help-Scout',         'Helpdesk', 5, 5, '{}'),
    ('trustradius', 'Help Scout', 'Help Scout', 'help-scout',               'Helpdesk', 5, 5, '{}'),
    ('reddit',      'Help Scout', 'Help Scout', 'help-scout-reddit',        'Helpdesk', 5, 3, '{"subreddits": "helpdesk,sysadmin"}'),
    ('hackernews',  'Help Scout', 'Help Scout', 'help-scout-hn',            'Helpdesk', 5, 3, '{}'),
    ('github',      'Help Scout', 'Help Scout', 'help-scout-github',        'Helpdesk', 5, 3, '{}'),
    ('rss',         'Help Scout', 'Help Scout', 'help-scout-rss',           'Helpdesk', 5, 3, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;

UPDATE b2b_scrape_targets
SET enabled = false
WHERE vendor_name = 'Help Scout'
  AND source IN ('reddit', 'hackernews', 'github', 'rss');

-- HappyFox
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, priority, max_pages, metadata)
VALUES
    ('g2',          'HappyFox', 'HappyFox', 'happyfox',                'Helpdesk', 5, 5, '{}'),
    ('capterra',    'HappyFox', 'HappyFox', '83211/HappyFox',          'Helpdesk', 5, 5, '{}'),
    ('trustradius', 'HappyFox', 'HappyFox', 'happyfox',               'Helpdesk', 5, 5, '{}'),
    ('reddit',      'HappyFox', 'HappyFox', 'happyfox-reddit',        'Helpdesk', 5, 3, '{"subreddits": "helpdesk,sysadmin"}'),
    ('hackernews',  'HappyFox', 'HappyFox', 'happyfox-hn',            'Helpdesk', 5, 3, '{}'),
    ('github',      'HappyFox', 'HappyFox', 'happyfox-github',        'Helpdesk', 5, 3, '{}'),
    ('rss',         'HappyFox', 'HappyFox', 'happyfox-rss',           'Helpdesk', 5, 3, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;

UPDATE b2b_scrape_targets
SET enabled = false
WHERE vendor_name = 'HappyFox'
  AND source IN ('reddit', 'hackernews', 'github', 'rss');


-- =====================================================================
-- NEW VENDORS: Marketing Automation (priority 3)
-- =====================================================================

-- HubSpot Marketing Hub
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, priority, max_pages, metadata)
VALUES
    ('g2',          'HubSpot Marketing Hub', 'HubSpot Marketing Hub', 'hubspot-marketing-hub',              'Marketing Automation', 3, 5, '{}'),
    ('capterra',    'HubSpot Marketing Hub', 'HubSpot Marketing Hub', '171840/HubSpot-Marketing',           'Marketing Automation', 3, 5, '{}'),
    ('trustradius', 'HubSpot Marketing Hub', 'HubSpot Marketing Hub', 'hubspot-marketing-hub',              'Marketing Automation', 3, 5, '{}'),
    ('reddit',      'HubSpot Marketing Hub', 'HubSpot Marketing Hub', 'hubspot-marketing-hub-reddit',       'Marketing Automation', 3, 3, '{"subreddits": "hubspot,marketing"}'),
    ('hackernews',  'HubSpot Marketing Hub', 'HubSpot Marketing Hub', 'hubspot-marketing-hub-hn',           'Marketing Automation', 3, 3, '{}'),
    ('github',      'HubSpot Marketing Hub', 'HubSpot Marketing Hub', 'hubspot-marketing-hub-github',       'Marketing Automation', 3, 3, '{}'),
    ('rss',         'HubSpot Marketing Hub', 'HubSpot Marketing Hub', 'hubspot-marketing-hub-rss',          'Marketing Automation', 3, 3, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;

UPDATE b2b_scrape_targets
SET enabled = false
WHERE vendor_name = 'HubSpot Marketing Hub'
  AND source IN ('reddit', 'hackernews', 'github', 'rss');

-- Brevo (formerly Sendinblue)
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, priority, max_pages, metadata)
VALUES
    ('g2',          'Brevo', 'Brevo', 'brevo',                'Marketing Automation', 3, 5, '{}'),
    ('capterra',    'Brevo', 'Brevo', '132996/brevo',         'Marketing Automation', 3, 5, '{}'),
    ('trustradius', 'Brevo', 'Brevo', 'brevo',                'Marketing Automation', 3, 5, '{}'),
    ('reddit',      'Brevo', 'Brevo', 'brevo-reddit',         'Marketing Automation', 3, 3, '{"subreddits": "emailmarketing,marketing"}'),
    ('hackernews',  'Brevo', 'Brevo', 'brevo-hn',             'Marketing Automation', 3, 3, '{}'),
    ('github',      'Brevo', 'Brevo', 'brevo-github',         'Marketing Automation', 3, 3, '{}'),
    ('rss',         'Brevo', 'Brevo', 'brevo-rss',            'Marketing Automation', 3, 3, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;

UPDATE b2b_scrape_targets
SET enabled = false
WHERE vendor_name = 'Brevo'
  AND source IN ('reddit', 'hackernews', 'github', 'rss');

-- Klaviyo
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, priority, max_pages, metadata)
VALUES
    ('g2',          'Klaviyo', 'Klaviyo', 'klaviyo',                'Marketing Automation', 3, 5, '{}'),
    ('capterra',    'Klaviyo', 'Klaviyo', '156699/Klaviyo',         'Marketing Automation', 3, 5, '{}'),
    ('trustradius', 'Klaviyo', 'Klaviyo', 'klaviyo',               'Marketing Automation', 3, 5, '{}'),
    ('reddit',      'Klaviyo', 'Klaviyo', 'klaviyo-reddit',        'Marketing Automation', 3, 3, '{"subreddits": "emailmarketing,ecommerce"}'),
    ('hackernews',  'Klaviyo', 'Klaviyo', 'klaviyo-hn',            'Marketing Automation', 3, 3, '{}'),
    ('github',      'Klaviyo', 'Klaviyo', 'klaviyo-github',        'Marketing Automation', 3, 3, '{}'),
    ('rss',         'Klaviyo', 'Klaviyo', 'klaviyo-rss',           'Marketing Automation', 3, 3, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;

UPDATE b2b_scrape_targets
SET enabled = false
WHERE vendor_name = 'Klaviyo'
  AND source IN ('reddit', 'hackernews', 'github', 'rss');

-- GetResponse
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, priority, max_pages, metadata)
VALUES
    ('g2',          'GetResponse', 'GetResponse', 'getresponse',                'Marketing Automation', 3, 5, '{}'),
    ('capterra',    'GetResponse', 'GetResponse', '153948/GetResponse',         'Marketing Automation', 3, 5, '{}'),
    ('trustradius', 'GetResponse', 'GetResponse', 'getresponse',               'Marketing Automation', 3, 5, '{}'),
    ('reddit',      'GetResponse', 'GetResponse', 'getresponse-reddit',        'Marketing Automation', 3, 3, '{"subreddits": "emailmarketing,marketing"}'),
    ('hackernews',  'GetResponse', 'GetResponse', 'getresponse-hn',            'Marketing Automation', 3, 3, '{}'),
    ('github',      'GetResponse', 'GetResponse', 'getresponse-github',        'Marketing Automation', 3, 3, '{}'),
    ('rss',         'GetResponse', 'GetResponse', 'getresponse-rss',           'Marketing Automation', 3, 3, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;

UPDATE b2b_scrape_targets
SET enabled = false
WHERE vendor_name = 'GetResponse'
  AND source IN ('reddit', 'hackernews', 'github', 'rss');


-- =====================================================================
-- NEW VENDORS: Cloud Infrastructure (priority 8)
-- =====================================================================

-- AWS
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, priority, max_pages, metadata)
VALUES
    ('g2',          'AWS', 'Amazon Web Services', 'amazon-web-services',                    'Cloud Infrastructure', 8, 5, '{}'),
    ('capterra',    'AWS', 'Amazon Web Services', '229593/AWS-Management-Console',           'Cloud Infrastructure', 8, 5, '{}'),
    ('trustradius', 'AWS', 'Amazon Web Services', 'amazon-web-services',                    'Cloud Infrastructure', 8, 5, '{}'),
    ('reddit',      'AWS', 'Amazon Web Services', 'aws-reddit',                             'Cloud Infrastructure', 8, 3, '{"subreddits": "aws,devops"}'),
    ('hackernews',  'AWS', 'Amazon Web Services', 'aws-hn',                                 'Cloud Infrastructure', 8, 3, '{}'),
    ('github',      'AWS', 'Amazon Web Services', 'aws-github',                             'Cloud Infrastructure', 8, 3, '{}'),
    ('rss',         'AWS', 'Amazon Web Services', 'aws-rss',                                'Cloud Infrastructure', 8, 3, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;

UPDATE b2b_scrape_targets
SET enabled = false
WHERE vendor_name = 'AWS'
  AND source IN ('reddit', 'hackernews', 'github', 'rss');

-- Azure
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, priority, max_pages, metadata)
VALUES
    ('g2',          'Azure', 'Microsoft Azure', 'microsoft-azure',              'Cloud Infrastructure', 8, 5, '{}'),
    ('capterra',    'Azure', 'Microsoft Azure', '16365/Azure',                  'Cloud Infrastructure', 8, 5, '{}'),
    ('trustradius', 'Azure', 'Microsoft Azure', 'microsoft-azure',             'Cloud Infrastructure', 8, 5, '{}'),
    ('reddit',      'Azure', 'Microsoft Azure', 'azure-reddit',               'Cloud Infrastructure', 8, 3, '{"subreddits": "azure,devops"}'),
    ('hackernews',  'Azure', 'Microsoft Azure', 'azure-hn',                   'Cloud Infrastructure', 8, 3, '{}'),
    ('github',      'Azure', 'Microsoft Azure', 'azure-github',               'Cloud Infrastructure', 8, 3, '{}'),
    ('rss',         'Azure', 'Microsoft Azure', 'azure-rss',                  'Cloud Infrastructure', 8, 3, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;

UPDATE b2b_scrape_targets
SET enabled = false
WHERE vendor_name = 'Azure'
  AND source IN ('reddit', 'hackernews', 'github', 'rss');

-- Google Cloud
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, priority, max_pages, metadata)
VALUES
    ('g2',          'Google Cloud', 'Google Cloud Platform', 'google-cloud-platform',                  'Cloud Infrastructure', 8, 5, '{}'),
    ('capterra',    'Google Cloud', 'Google Cloud Platform', '268690/Google-Cloud-Platform',            'Cloud Infrastructure', 8, 5, '{}'),
    ('trustradius', 'Google Cloud', 'Google Cloud Platform', 'google-cloud-platform',                  'Cloud Infrastructure', 8, 5, '{}'),
    ('reddit',      'Google Cloud', 'Google Cloud Platform', 'google-cloud-reddit',                    'Cloud Infrastructure', 8, 3, '{"subreddits": "googlecloud,devops"}'),
    ('hackernews',  'Google Cloud', 'Google Cloud Platform', 'google-cloud-hn',                        'Cloud Infrastructure', 8, 3, '{}'),
    ('github',      'Google Cloud', 'Google Cloud Platform', 'google-cloud-github',                    'Cloud Infrastructure', 8, 3, '{}'),
    ('rss',         'Google Cloud', 'Google Cloud Platform', 'google-cloud-rss',                       'Cloud Infrastructure', 8, 3, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;

UPDATE b2b_scrape_targets
SET enabled = false
WHERE vendor_name = 'Google Cloud'
  AND source IN ('reddit', 'hackernews', 'github', 'rss');

-- DigitalOcean
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, priority, max_pages, metadata)
VALUES
    ('g2',          'DigitalOcean', 'DigitalOcean', 'digitalocean',                'Cloud Infrastructure', 8, 5, '{}'),
    ('capterra',    'DigitalOcean', 'DigitalOcean', '205055/DigitalOcean',         'Cloud Infrastructure', 8, 5, '{}'),
    ('trustradius', 'DigitalOcean', 'DigitalOcean', 'digitalocean',               'Cloud Infrastructure', 8, 5, '{}'),
    ('reddit',      'DigitalOcean', 'DigitalOcean', 'digitalocean-reddit',        'Cloud Infrastructure', 8, 3, '{"subreddits": "digitalocean,devops"}'),
    ('hackernews',  'DigitalOcean', 'DigitalOcean', 'digitalocean-hn',            'Cloud Infrastructure', 8, 3, '{}'),
    ('github',      'DigitalOcean', 'DigitalOcean', 'digitalocean-github',        'Cloud Infrastructure', 8, 3, '{}'),
    ('rss',         'DigitalOcean', 'DigitalOcean', 'digitalocean-rss',           'Cloud Infrastructure', 8, 3, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;

UPDATE b2b_scrape_targets
SET enabled = false
WHERE vendor_name = 'DigitalOcean'
  AND source IN ('reddit', 'hackernews', 'github', 'rss');

-- Linode
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, priority, max_pages, metadata)
VALUES
    ('g2',          'Linode', 'Linode', 'linode',                'Cloud Infrastructure', 8, 5, '{}'),
    ('capterra',    'Linode', 'Linode', '210618/Linode',         'Cloud Infrastructure', 8, 5, '{}'),
    ('trustradius', 'Linode', 'Linode', 'linode',               'Cloud Infrastructure', 8, 5, '{}'),
    ('reddit',      'Linode', 'Linode', 'linode-reddit',        'Cloud Infrastructure', 8, 3, '{"subreddits": "linode,selfhosted"}'),
    ('hackernews',  'Linode', 'Linode', 'linode-hn',            'Cloud Infrastructure', 8, 3, '{}'),
    ('github',      'Linode', 'Linode', 'linode-github',        'Cloud Infrastructure', 8, 3, '{}'),
    ('rss',         'Linode', 'Linode', 'linode-rss',           'Cloud Infrastructure', 8, 3, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;

UPDATE b2b_scrape_targets
SET enabled = false
WHERE vendor_name = 'Linode'
  AND source IN ('reddit', 'hackernews', 'github', 'rss');


-- =====================================================================
-- NEW VENDORS: Cybersecurity (priority 7)
-- =====================================================================

-- CrowdStrike
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, priority, max_pages, metadata)
VALUES
    ('g2',          'CrowdStrike', 'CrowdStrike Falcon', 'crowdstrike-falcon',                  'Cybersecurity', 7, 5, '{}'),
    ('capterra',    'CrowdStrike', 'CrowdStrike Falcon', '147662/CrowdStrike-Falcon',           'Cybersecurity', 7, 5, '{}'),
    ('trustradius', 'CrowdStrike', 'CrowdStrike Falcon', 'crowdstrike-falcon',                  'Cybersecurity', 7, 5, '{}'),
    ('reddit',      'CrowdStrike', 'CrowdStrike Falcon', 'crowdstrike-reddit',                  'Cybersecurity', 7, 3, '{"subreddits": "crowdstrike,sysadmin"}'),
    ('hackernews',  'CrowdStrike', 'CrowdStrike Falcon', 'crowdstrike-hn',                      'Cybersecurity', 7, 3, '{}'),
    ('github',      'CrowdStrike', 'CrowdStrike Falcon', 'crowdstrike-github',                  'Cybersecurity', 7, 3, '{}'),
    ('rss',         'CrowdStrike', 'CrowdStrike Falcon', 'crowdstrike-rss',                     'Cybersecurity', 7, 3, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;

UPDATE b2b_scrape_targets
SET enabled = false
WHERE vendor_name = 'CrowdStrike'
  AND source IN ('reddit', 'hackernews', 'github', 'rss');

-- SentinelOne
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, priority, max_pages, metadata)
VALUES
    ('g2',          'SentinelOne', 'SentinelOne Singularity', 'sentinelone',                           'Cybersecurity', 7, 5, '{}'),
    ('capterra',    'SentinelOne', 'SentinelOne Singularity', '152564/Endpoint-Protection-Platform',   'Cybersecurity', 7, 5, '{}'),
    ('trustradius', 'SentinelOne', 'SentinelOne Singularity', 'sentinelone',                           'Cybersecurity', 7, 5, '{}'),
    ('reddit',      'SentinelOne', 'SentinelOne Singularity', 'sentinelone-reddit',                    'Cybersecurity', 7, 3, '{"subreddits": "sysadmin,cybersecurity"}'),
    ('hackernews',  'SentinelOne', 'SentinelOne Singularity', 'sentinelone-hn',                        'Cybersecurity', 7, 3, '{}'),
    ('github',      'SentinelOne', 'SentinelOne Singularity', 'sentinelone-github',                    'Cybersecurity', 7, 3, '{}'),
    ('rss',         'SentinelOne', 'SentinelOne Singularity', 'sentinelone-rss',                       'Cybersecurity', 7, 3, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;

UPDATE b2b_scrape_targets
SET enabled = false
WHERE vendor_name = 'SentinelOne'
  AND source IN ('reddit', 'hackernews', 'github', 'rss');

-- Palo Alto Networks
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, priority, max_pages, metadata)
VALUES
    ('g2',          'Palo Alto Networks', 'Cortex XDR', 'palo-alto-networks-cortex-xdr',        'Cybersecurity', 7, 5, '{}'),
    ('capterra',    'Palo Alto Networks', 'Cortex XDR', '175139/Traps',                         'Cybersecurity', 7, 5, '{}'),
    ('trustradius', 'Palo Alto Networks', 'Cortex XDR', 'palo-alto-networks-cortex',            'Cybersecurity', 7, 5, '{}'),
    ('reddit',      'Palo Alto Networks', 'Cortex XDR', 'palo-alto-reddit',                     'Cybersecurity', 7, 3, '{"subreddits": "paloaltonetworks,sysadmin"}'),
    ('hackernews',  'Palo Alto Networks', 'Cortex XDR', 'palo-alto-hn',                         'Cybersecurity', 7, 3, '{}'),
    ('github',      'Palo Alto Networks', 'Cortex XDR', 'palo-alto-github',                     'Cybersecurity', 7, 3, '{}'),
    ('rss',         'Palo Alto Networks', 'Cortex XDR', 'palo-alto-rss',                        'Cybersecurity', 7, 3, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;

UPDATE b2b_scrape_targets
SET enabled = false
WHERE vendor_name = 'Palo Alto Networks'
  AND source IN ('reddit', 'hackernews', 'github', 'rss');

-- Fortinet
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, priority, max_pages, metadata)
VALUES
    ('g2',          'Fortinet', 'FortiGate', 'fortinet-fortigate',                              'Cybersecurity', 7, 5, '{}'),
    ('capterra',    'Fortinet', 'FortiGate', '231835/FortiGate-Next-Generation-Firewalls',      'Cybersecurity', 7, 5, '{}'),
    ('trustradius', 'Fortinet', 'FortiGate', 'fortinet-fortigate',                              'Cybersecurity', 7, 5, '{}'),
    ('reddit',      'Fortinet', 'FortiGate', 'fortinet-reddit',                                 'Cybersecurity', 7, 3, '{"subreddits": "fortinet,sysadmin"}'),
    ('hackernews',  'Fortinet', 'FortiGate', 'fortinet-hn',                                     'Cybersecurity', 7, 3, '{}'),
    ('github',      'Fortinet', 'FortiGate', 'fortinet-github',                                 'Cybersecurity', 7, 3, '{}'),
    ('rss',         'Fortinet', 'FortiGate', 'fortinet-rss',                                    'Cybersecurity', 7, 3, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;

UPDATE b2b_scrape_targets
SET enabled = false
WHERE vendor_name = 'Fortinet'
  AND source IN ('reddit', 'hackernews', 'github', 'rss');


-- =====================================================================
-- NEW VENDORS: HR / HCM (priority 4)
-- =====================================================================

-- BambooHR
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, priority, max_pages, metadata)
VALUES
    ('g2',          'BambooHR', 'BambooHR', 'bamboohr',                'HR / HCM', 4, 5, '{}'),
    ('capterra',    'BambooHR', 'BambooHR', '110968/BambooHR',         'HR / HCM', 4, 5, '{}'),
    ('trustradius', 'BambooHR', 'BambooHR', 'bamboohr',               'HR / HCM', 4, 5, '{}'),
    ('reddit',      'BambooHR', 'BambooHR', 'bamboohr-reddit',        'HR / HCM', 4, 3, '{"subreddits": "humanresources,sysadmin"}'),
    ('hackernews',  'BambooHR', 'BambooHR', 'bamboohr-hn',            'HR / HCM', 4, 3, '{}'),
    ('github',      'BambooHR', 'BambooHR', 'bamboohr-github',        'HR / HCM', 4, 3, '{}'),
    ('rss',         'BambooHR', 'BambooHR', 'bamboohr-rss',           'HR / HCM', 4, 3, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;

UPDATE b2b_scrape_targets
SET enabled = false
WHERE vendor_name = 'BambooHR'
  AND source IN ('reddit', 'hackernews', 'github', 'rss');

-- Gusto
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, priority, max_pages, metadata)
VALUES
    ('g2',          'Gusto', 'Gusto', 'gusto',                'HR / HCM', 4, 5, '{}'),
    ('capterra',    'Gusto', 'Gusto', '131882/Gusto',         'HR / HCM', 4, 5, '{}'),
    ('trustradius', 'Gusto', 'Gusto', 'gusto',               'HR / HCM', 4, 5, '{}'),
    ('reddit',      'Gusto', 'Gusto', 'gusto-reddit',        'HR / HCM', 4, 3, '{"subreddits": "smallbusiness,accounting"}'),
    ('hackernews',  'Gusto', 'Gusto', 'gusto-hn',            'HR / HCM', 4, 3, '{}'),
    ('github',      'Gusto', 'Gusto', 'gusto-github',        'HR / HCM', 4, 3, '{}'),
    ('rss',         'Gusto', 'Gusto', 'gusto-rss',           'HR / HCM', 4, 3, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;

UPDATE b2b_scrape_targets
SET enabled = false
WHERE vendor_name = 'Gusto'
  AND source IN ('reddit', 'hackernews', 'github', 'rss');

-- Rippling
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, priority, max_pages, metadata)
VALUES
    ('g2',          'Rippling', 'Rippling', 'rippling',                'HR / HCM', 4, 5, '{}'),
    ('capterra',    'Rippling', 'Rippling', '172127/Rippling',         'HR / HCM', 4, 5, '{}'),
    ('trustradius', 'Rippling', 'Rippling', 'rippling',               'HR / HCM', 4, 5, '{}'),
    ('reddit',      'Rippling', 'Rippling', 'rippling-reddit',        'HR / HCM', 4, 3, '{"subreddits": "humanresources,startups"}'),
    ('hackernews',  'Rippling', 'Rippling', 'rippling-hn',            'HR / HCM', 4, 3, '{}'),
    ('github',      'Rippling', 'Rippling', 'rippling-github',        'HR / HCM', 4, 3, '{}'),
    ('rss',         'Rippling', 'Rippling', 'rippling-rss',           'HR / HCM', 4, 3, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;

UPDATE b2b_scrape_targets
SET enabled = false
WHERE vendor_name = 'Rippling'
  AND source IN ('reddit', 'hackernews', 'github', 'rss');

-- Workday
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, priority, max_pages, metadata)
VALUES
    ('g2',          'Workday', 'Workday HCM', 'workday',                'HR / HCM', 4, 5, '{}'),
    ('capterra',    'Workday', 'Workday HCM', '66908/Workday-HCM',     'HR / HCM', 4, 5, '{}'),
    ('trustradius', 'Workday', 'Workday HCM', 'workday',               'HR / HCM', 4, 5, '{}'),
    ('reddit',      'Workday', 'Workday HCM', 'workday-reddit',        'HR / HCM', 4, 3, '{"subreddits": "humanresources,sysadmin"}'),
    ('hackernews',  'Workday', 'Workday HCM', 'workday-hn',            'HR / HCM', 4, 3, '{}'),
    ('github',      'Workday', 'Workday HCM', 'workday-github',        'HR / HCM', 4, 3, '{}'),
    ('rss',         'Workday', 'Workday HCM', 'workday-rss',           'HR / HCM', 4, 3, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;

UPDATE b2b_scrape_targets
SET enabled = false
WHERE vendor_name = 'Workday'
  AND source IN ('reddit', 'hackernews', 'github', 'rss');


-- =====================================================================
-- NEW VENDORS: Data & Analytics (priority 4)
-- =====================================================================

-- Tableau
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, priority, max_pages, metadata)
VALUES
    ('g2',          'Tableau', 'Tableau', 'tableau',                'Data & Analytics', 4, 5, '{}'),
    ('capterra',    'Tableau', 'Tableau', '208764/Tableau',         'Data & Analytics', 4, 5, '{}'),
    ('trustradius', 'Tableau', 'Tableau', 'tableau',               'Data & Analytics', 4, 5, '{}'),
    ('reddit',      'Tableau', 'Tableau', 'tableau-reddit',        'Data & Analytics', 4, 3, '{"subreddits": "tableau,datascience"}'),
    ('hackernews',  'Tableau', 'Tableau', 'tableau-hn',            'Data & Analytics', 4, 3, '{}'),
    ('github',      'Tableau', 'Tableau', 'tableau-github',        'Data & Analytics', 4, 3, '{}'),
    ('rss',         'Tableau', 'Tableau', 'tableau-rss',           'Data & Analytics', 4, 3, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;

UPDATE b2b_scrape_targets
SET enabled = false
WHERE vendor_name = 'Tableau'
  AND source IN ('reddit', 'hackernews', 'github', 'rss');

-- Looker
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, priority, max_pages, metadata)
VALUES
    ('g2',          'Looker', 'Looker', 'looker',                'Data & Analytics', 4, 5, '{}'),
    ('capterra',    'Looker', 'Looker', '169053/Looker',         'Data & Analytics', 4, 5, '{}'),
    ('trustradius', 'Looker', 'Looker', 'looker',               'Data & Analytics', 4, 5, '{}'),
    ('reddit',      'Looker', 'Looker', 'looker-reddit',        'Data & Analytics', 4, 3, '{"subreddits": "looker,datascience"}'),
    ('hackernews',  'Looker', 'Looker', 'looker-hn',            'Data & Analytics', 4, 3, '{}'),
    ('github',      'Looker', 'Looker', 'looker-github',        'Data & Analytics', 4, 3, '{}'),
    ('rss',         'Looker', 'Looker', 'looker-rss',           'Data & Analytics', 4, 3, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;

UPDATE b2b_scrape_targets
SET enabled = false
WHERE vendor_name = 'Looker'
  AND source IN ('reddit', 'hackernews', 'github', 'rss');

-- Power BI
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, priority, max_pages, metadata)
VALUES
    ('g2',          'Power BI', 'Microsoft Power BI', 'microsoft-power-bi',        'Data & Analytics', 4, 5, '{}'),
    ('capterra',    'Power BI', 'Microsoft Power BI', '176586/Power-BI',           'Data & Analytics', 4, 5, '{}'),
    ('trustradius', 'Power BI', 'Microsoft Power BI', 'microsoft-power-bi',        'Data & Analytics', 4, 5, '{}'),
    ('reddit',      'Power BI', 'Microsoft Power BI', 'power-bi-reddit',           'Data & Analytics', 4, 3, '{"subreddits": "powerbi,datascience"}'),
    ('hackernews',  'Power BI', 'Microsoft Power BI', 'power-bi-hn',              'Data & Analytics', 4, 3, '{}'),
    ('github',      'Power BI', 'Microsoft Power BI', 'power-bi-github',          'Data & Analytics', 4, 3, '{}'),
    ('rss',         'Power BI', 'Microsoft Power BI', 'power-bi-rss',             'Data & Analytics', 4, 3, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;

UPDATE b2b_scrape_targets
SET enabled = false
WHERE vendor_name = 'Power BI'
  AND source IN ('reddit', 'hackernews', 'github', 'rss');

-- Metabase
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, priority, max_pages, metadata)
VALUES
    ('g2',          'Metabase', 'Metabase', 'metabase',                'Data & Analytics', 4, 5, '{}'),
    ('capterra',    'Metabase', 'Metabase', '176651/Metabase',         'Data & Analytics', 4, 5, '{}'),
    ('trustradius', 'Metabase', 'Metabase', 'metabase',               'Data & Analytics', 4, 5, '{}'),
    ('reddit',      'Metabase', 'Metabase', 'metabase-reddit',        'Data & Analytics', 4, 3, '{"subreddits": "metabase,datascience"}'),
    ('hackernews',  'Metabase', 'Metabase', 'metabase-hn',            'Data & Analytics', 4, 3, '{}'),
    ('github',      'Metabase', 'Metabase', 'metabase-github',        'Data & Analytics', 4, 3, '{}'),
    ('rss',         'Metabase', 'Metabase', 'metabase-rss',           'Data & Analytics', 4, 3, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;

UPDATE b2b_scrape_targets
SET enabled = false
WHERE vendor_name = 'Metabase'
  AND source IN ('reddit', 'hackernews', 'github', 'rss');


-- =====================================================================
-- NEW VENDORS: Communication (priority 3)
-- =====================================================================

-- Slack
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, priority, max_pages, metadata)
VALUES
    ('g2',          'Slack', 'Slack', 'slack',                'Communication', 3, 5, '{}'),
    ('capterra',    'Slack', 'Slack', '135003/Slack',         'Communication', 3, 5, '{}'),
    ('trustradius', 'Slack', 'Slack', 'slack',               'Communication', 3, 5, '{}'),
    ('reddit',      'Slack', 'Slack', 'slack-reddit',        'Communication', 3, 3, '{"subreddits": "slack,sysadmin"}'),
    ('hackernews',  'Slack', 'Slack', 'slack-hn',            'Communication', 3, 3, '{}'),
    ('github',      'Slack', 'Slack', 'slack-github',        'Communication', 3, 3, '{}'),
    ('rss',         'Slack', 'Slack', 'slack-rss',           'Communication', 3, 3, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;

UPDATE b2b_scrape_targets
SET enabled = false
WHERE vendor_name = 'Slack'
  AND source IN ('reddit', 'hackernews', 'github', 'rss');

-- Microsoft Teams
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, priority, max_pages, metadata)
VALUES
    ('g2',          'Microsoft Teams', 'Microsoft Teams', 'microsoft-teams',                'Communication', 3, 5, '{}'),
    ('capterra',    'Microsoft Teams', 'Microsoft Teams', '168668/Microsoft-Teams',         'Communication', 3, 5, '{}'),
    ('trustradius', 'Microsoft Teams', 'Microsoft Teams', 'microsoft-teams',               'Communication', 3, 5, '{}'),
    ('reddit',      'Microsoft Teams', 'Microsoft Teams', 'microsoft-teams-reddit',        'Communication', 3, 3, '{"subreddits": "microsoftteams,sysadmin"}'),
    ('hackernews',  'Microsoft Teams', 'Microsoft Teams', 'microsoft-teams-hn',            'Communication', 3, 3, '{}'),
    ('github',      'Microsoft Teams', 'Microsoft Teams', 'microsoft-teams-github',        'Communication', 3, 3, '{}'),
    ('rss',         'Microsoft Teams', 'Microsoft Teams', 'microsoft-teams-rss',           'Communication', 3, 3, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;

UPDATE b2b_scrape_targets
SET enabled = false
WHERE vendor_name = 'Microsoft Teams'
  AND source IN ('reddit', 'hackernews', 'github', 'rss');

-- Zoom
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, priority, max_pages, metadata)
VALUES
    ('g2',          'Zoom', 'Zoom', 'zoom',                                    'Communication', 3, 5, '{}'),
    ('capterra',    'Zoom', 'Zoom', '144037/Zoom-Video-Conferencing',           'Communication', 3, 5, '{}'),
    ('trustradius', 'Zoom', 'Zoom', 'zoom',                                    'Communication', 3, 5, '{}'),
    ('reddit',      'Zoom', 'Zoom', 'zoom-reddit',                             'Communication', 3, 3, '{"subreddits": "zoom,sysadmin"}'),
    ('hackernews',  'Zoom', 'Zoom', 'zoom-hn',                                 'Communication', 3, 3, '{}'),
    ('github',      'Zoom', 'Zoom', 'zoom-github',                             'Communication', 3, 3, '{}'),
    ('rss',         'Zoom', 'Zoom', 'zoom-rss',                                'Communication', 3, 3, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;

UPDATE b2b_scrape_targets
SET enabled = false
WHERE vendor_name = 'Zoom'
  AND source IN ('reddit', 'hackernews', 'github', 'rss');

-- RingCentral
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, priority, max_pages, metadata)
VALUES
    ('g2',          'RingCentral', 'RingCentral', 'ringcentral',                    'Communication', 3, 5, '{}'),
    ('capterra',    'RingCentral', 'RingCentral', '242188/RingCentral-MVP',         'Communication', 3, 5, '{}'),
    ('trustradius', 'RingCentral', 'RingCentral', 'ringcentral',                    'Communication', 3, 5, '{}'),
    ('reddit',      'RingCentral', 'RingCentral', 'ringcentral-reddit',             'Communication', 3, 3, '{"subreddits": "voip,sysadmin"}'),
    ('hackernews',  'RingCentral', 'RingCentral', 'ringcentral-hn',                 'Communication', 3, 3, '{}'),
    ('github',      'RingCentral', 'RingCentral', 'ringcentral-github',             'Communication', 3, 3, '{}'),
    ('rss',         'RingCentral', 'RingCentral', 'ringcentral-rss',                'Communication', 3, 3, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;

UPDATE b2b_scrape_targets
SET enabled = false
WHERE vendor_name = 'RingCentral'
  AND source IN ('reddit', 'hackernews', 'github', 'rss');


-- =====================================================================
-- NEW VENDORS: E-commerce (priority 3)
-- =====================================================================

-- Shopify
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, priority, max_pages, metadata)
VALUES
    ('g2',          'Shopify', 'Shopify', 'shopify',                'E-commerce', 3, 5, '{}'),
    ('capterra',    'Shopify', 'Shopify', '83891/Shopify',          'E-commerce', 3, 5, '{}'),
    ('trustradius', 'Shopify', 'Shopify', 'shopify',               'E-commerce', 3, 5, '{}'),
    ('reddit',      'Shopify', 'Shopify', 'shopify-reddit',        'E-commerce', 3, 3, '{"subreddits": "shopify,ecommerce"}'),
    ('hackernews',  'Shopify', 'Shopify', 'shopify-hn',            'E-commerce', 3, 3, '{}'),
    ('github',      'Shopify', 'Shopify', 'shopify-github',        'E-commerce', 3, 3, '{}'),
    ('rss',         'Shopify', 'Shopify', 'shopify-rss',           'E-commerce', 3, 3, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;

UPDATE b2b_scrape_targets
SET enabled = false
WHERE vendor_name = 'Shopify'
  AND source IN ('reddit', 'hackernews', 'github', 'rss');

-- BigCommerce
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, priority, max_pages, metadata)
VALUES
    ('g2',          'BigCommerce', 'BigCommerce', 'bigcommerce',                'E-commerce', 3, 5, '{}'),
    ('capterra',    'BigCommerce', 'BigCommerce', '131883/Bigcommerce',         'E-commerce', 3, 5, '{}'),
    ('trustradius', 'BigCommerce', 'BigCommerce', 'bigcommerce',               'E-commerce', 3, 5, '{}'),
    ('reddit',      'BigCommerce', 'BigCommerce', 'bigcommerce-reddit',        'E-commerce', 3, 3, '{"subreddits": "ecommerce,bigcommerce"}'),
    ('hackernews',  'BigCommerce', 'BigCommerce', 'bigcommerce-hn',            'E-commerce', 3, 3, '{}'),
    ('github',      'BigCommerce', 'BigCommerce', 'bigcommerce-github',        'E-commerce', 3, 3, '{}'),
    ('rss',         'BigCommerce', 'BigCommerce', 'bigcommerce-rss',           'E-commerce', 3, 3, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;

UPDATE b2b_scrape_targets
SET enabled = false
WHERE vendor_name = 'BigCommerce'
  AND source IN ('reddit', 'hackernews', 'github', 'rss');

-- WooCommerce
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, priority, max_pages, metadata)
VALUES
    ('g2',          'WooCommerce', 'WooCommerce', 'woocommerce',                'E-commerce', 3, 5, '{}'),
    ('capterra',    'WooCommerce', 'WooCommerce', '225601/WooCommerce',         'E-commerce', 3, 5, '{}'),
    ('trustradius', 'WooCommerce', 'WooCommerce', 'woocommerce',               'E-commerce', 3, 5, '{}'),
    ('reddit',      'WooCommerce', 'WooCommerce', 'woocommerce-reddit',        'E-commerce', 3, 3, '{"subreddits": "woocommerce,ecommerce"}'),
    ('hackernews',  'WooCommerce', 'WooCommerce', 'woocommerce-hn',            'E-commerce', 3, 3, '{}'),
    ('github',      'WooCommerce', 'WooCommerce', 'woocommerce-github',        'E-commerce', 3, 3, '{}'),
    ('rss',         'WooCommerce', 'WooCommerce', 'woocommerce-rss',           'E-commerce', 3, 3, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;

UPDATE b2b_scrape_targets
SET enabled = false
WHERE vendor_name = 'WooCommerce'
  AND source IN ('reddit', 'hackernews', 'github', 'rss');

-- Magento
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, priority, max_pages, metadata)
VALUES
    ('g2',          'Magento', 'Adobe Commerce', 'magento',                    'E-commerce', 3, 5, '{}'),
    ('capterra',    'Magento', 'Adobe Commerce', '227129/Magento-Commerce',    'E-commerce', 3, 5, '{}'),
    ('trustradius', 'Magento', 'Adobe Commerce', 'magento',                    'E-commerce', 3, 5, '{}'),
    ('reddit',      'Magento', 'Adobe Commerce', 'magento-reddit',            'E-commerce', 3, 3, '{"subreddits": "magento,ecommerce"}'),
    ('hackernews',  'Magento', 'Adobe Commerce', 'magento-hn',                'E-commerce', 3, 3, '{}'),
    ('github',      'Magento', 'Adobe Commerce', 'magento-github',            'E-commerce', 3, 3, '{}'),
    ('rss',         'Magento', 'Adobe Commerce', 'magento-rss',               'E-commerce', 3, 3, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;

UPDATE b2b_scrape_targets
SET enabled = false
WHERE vendor_name = 'Magento'
  AND source IN ('reddit', 'hackernews', 'github', 'rss');
