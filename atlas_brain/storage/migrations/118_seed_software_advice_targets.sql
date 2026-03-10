-- Seed Software Advice scrape targets for all existing vendors.
-- Migration 118
--
-- Software Advice uses two URL patterns:
--   1. /{category}/{slug}-profile/reviews/  (most products)
--   2. /product/{id}-{Name}/reviews/        (some products)
--
-- All slugs verified via web search against softwareadvice.com (Mar 2026).
-- Idempotent: ON CONFLICT (source, product_slug) DO NOTHING.

-- =====================================================================
-- CRM
-- =====================================================================
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, priority, max_pages, metadata)
VALUES
    ('software_advice', 'Salesforce',   'Salesforce Sales Cloud', 'product/2764-Salesforce',            'CRM', 15, 3, '{}'),
    ('software_advice', 'Pipedrive',    'Pipedrive',              'crm/pipedrive-profile',              'CRM', 15, 3, '{}'),
    ('software_advice', 'Zoho CRM',     'Zoho CRM',              'crm/zoho-crm-profile',               'CRM', 15, 3, '{}'),
    ('software_advice', 'Copper',       'Copper',                 'product/63221-Copper',               'CRM', 15, 3, '{}'),
    ('software_advice', 'Close',        'Close',                  'crm/close-io-profile',               'CRM', 15, 3, '{}'),
    ('software_advice', 'Insightly',    'Insightly',              'product/2667-Insightly',             'CRM', 15, 3, '{}'),
    ('software_advice', 'Nutshell',     'Nutshell',               'crm/nutshell-profile',               'CRM', 15, 3, '{}'),
    ('software_advice', 'Freshsales',   'Freshsales',             'product/113796-Freshsales',          'CRM', 15, 3, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;

-- =====================================================================
-- Project Management
-- =====================================================================
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, priority, max_pages, metadata)
VALUES
    ('software_advice', 'Asana',       'Asana',       'project-management/asana-profile',       'Project Management', 15, 3, '{}'),
    ('software_advice', 'ClickUp',     'ClickUp',     'project-management/clickup-profile',     'Project Management', 15, 3, '{}'),
    ('software_advice', 'Monday.com',  'Monday.com',  'marketing/monday-com-profile',           'Project Management', 15, 3, '{}'),
    ('software_advice', 'Notion',      'Notion',      'project-management/notion-profile',      'Project Management', 15, 3, '{}'),
    ('software_advice', 'Smartsheet',  'Smartsheet',  'project-management/smartsheet-profile',  'Project Management', 15, 3, '{}'),
    ('software_advice', 'Wrike',       'Wrike',       'project-management/wrike-profile',       'Project Management', 15, 3, '{}'),
    ('software_advice', 'Basecamp',    'Basecamp',    'project-management/basecamp-profile',    'Project Management', 15, 3, '{}'),
    ('software_advice', 'Teamwork',    'Teamwork',    'project-management/teamwork-pm-profile', 'Project Management', 15, 3, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;

-- =====================================================================
-- Helpdesk / Customer Service
-- =====================================================================
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, priority, max_pages, metadata)
VALUES
    ('software_advice', 'Zendesk',           'Zendesk Suite',       'product/26892-Zendesk',                'Helpdesk', 15, 3, '{}'),
    ('software_advice', 'Freshdesk',         'Freshdesk',           'crm/freshdesk-profile',                'Helpdesk', 15, 3, '{}'),
    ('software_advice', 'Intercom',          'Intercom',            'crm/intercom-profile',                 'Helpdesk', 15, 3, '{}'),
    ('software_advice', 'HubSpot Service Hub','HubSpot Service Hub','crm/hubspot-service-hub-profile',      'Helpdesk', 15, 3, '{}'),
    ('software_advice', 'Zoho Desk',         'Zoho Desk',           'crm/zoho-desk-profile',                'Helpdesk', 15, 3, '{}'),
    ('software_advice', 'Help Scout',        'Help Scout',          'crm/help-scout-profile',               'Helpdesk', 15, 3, '{}'),
    ('software_advice', 'HappyFox',          'HappyFox Help Desk',  'crm/happyfox-profile',                'Helpdesk', 15, 3, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;

-- =====================================================================
-- Marketing Automation
-- =====================================================================
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, priority, max_pages, metadata)
VALUES
    ('software_advice', 'HubSpot Marketing Hub', 'HubSpot Marketing Hub', 'product/119437-HubSpot-Marketing', 'Marketing Automation', 15, 3, '{}'),
    ('software_advice', 'Mailchimp',             'Mailchimp',             'crm/mailchimp-profile',             'Marketing Automation', 15, 3, '{}'),
    ('software_advice', 'ActiveCampaign',        'ActiveCampaign',        'crm/activecampaign-profile',        'Marketing Automation', 15, 3, '{}'),
    ('software_advice', 'Brevo',                 'Brevo',                 'scheduling/brevo-profile',          'Marketing Automation', 15, 3, '{}'),
    ('software_advice', 'Klaviyo',               'Klaviyo',               'marketing/klaviyo-profile',         'Marketing Automation', 15, 3, '{}'),
    ('software_advice', 'GetResponse',           'GetResponse',           'marketing/getresponse-profile',     'Marketing Automation', 15, 3, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;

-- =====================================================================
-- HR / HCM
-- =====================================================================
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, priority, max_pages, metadata)
VALUES
    ('software_advice', 'BambooHR',  'BambooHR',    'hr/bamboohr-profile',   'HR / HCM', 15, 3, '{}'),
    ('software_advice', 'Gusto',     'Gusto',       'product/20428-Gusto',   'HR / HCM', 15, 3, '{}'),
    ('software_advice', 'Rippling',  'Rippling',    'hr/rippling-profile',   'HR / HCM', 15, 3, '{}'),
    ('software_advice', 'Workday',   'Workday HCM', 'psa/workday-hcm-profile', 'HR / HCM', 15, 3, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;

-- =====================================================================
-- E-commerce
-- =====================================================================
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, priority, max_pages, metadata)
VALUES
    ('software_advice', 'Shopify',      'Shopify',      'retail/shopify-profile',                          'E-commerce', 15, 3, '{}'),
    ('software_advice', 'BigCommerce',  'BigCommerce',  'ecommerce/bigcommerce-profile',                   'E-commerce', 15, 3, '{}'),
    ('software_advice', 'Magento',      'Adobe Commerce','inventory-management/magento-commerce-profile',  'E-commerce', 15, 3, '{}'),
    ('software_advice', 'WooCommerce',  'WooCommerce',  'ecommerce/woocommerce-profile',                   'E-commerce', 15, 3, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;

-- =====================================================================
-- Data & Analytics
-- =====================================================================
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, priority, max_pages, metadata)
VALUES
    ('software_advice', 'Tableau',  'Tableau',          'bi/tableau-profile',             'Data & Analytics', 15, 3, '{}'),
    ('software_advice', 'Power BI', 'Microsoft Power BI','bi/microsoft-power-bi-profile', 'Data & Analytics', 15, 3, '{}'),
    ('software_advice', 'Looker',   'Looker',           'bi/looker-profile',              'Data & Analytics', 15, 3, '{}'),
    ('software_advice', 'Metabase', 'Metabase',         'bi/metabase-profile',            'Data & Analytics', 15, 3, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;

-- =====================================================================
-- Communication
-- =====================================================================
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, priority, max_pages, metadata)
VALUES
    ('software_advice', 'Slack',            'Slack',            'remote-support/slack-profile',        'Communication', 15, 3, '{}'),
    ('software_advice', 'Microsoft Teams',  'Microsoft Teams',  'voip/microsoft-teams-profile',        'Communication', 15, 3, '{}'),
    ('software_advice', 'Zoom',             'Zoom Workplace',   'product/101384-Zoom-Video-Conferencing', 'Communication', 15, 3, '{}'),
    ('software_advice', 'RingCentral',      'RingEX',           'call-center/office-profile',          'Communication', 15, 3, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;

-- =====================================================================
-- Cloud Infrastructure
-- =====================================================================
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, priority, max_pages, metadata)
VALUES
    ('software_advice', 'Azure',                  'Microsoft Azure',  'website-builder/microsoft-azure-profile',     'Cloud Infrastructure', 15, 3, '{}'),
    ('software_advice', 'Google Cloud Platform',  'Google Cloud',     'compliance/google-cloud-platform-profile',    'Cloud Infrastructure', 15, 3, '{}'),
    ('software_advice', 'DigitalOcean',           'DigitalOcean',     'virtual-machine/digitalocean-profile',        'Cloud Infrastructure', 15, 3, '{}'),
    ('software_advice', 'Linode',                 'Linode',           'cloud-management/linode-profile',             'Cloud Infrastructure', 15, 3, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;

-- NOTE: Amazon Web Services has no single umbrella review page on Software Advice.
-- Individual service pages exist (amazon-ec2-profile, amazon-s3-profile, etc.)
-- but do not map cleanly to a single vendor target. Skipped for now.

-- =====================================================================
-- Cybersecurity
-- =====================================================================
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, priority, max_pages, metadata)
VALUES
    ('software_advice', 'CrowdStrike',         'CrowdStrike Falcon',        'product/135499-CrowdStrike-Falcon',                    'Cybersecurity', 15, 3, '{}'),
    ('software_advice', 'Palo Alto Networks',   'Palo Alto Cortex XDR',      'security/traps-profile',                               'Cybersecurity', 15, 3, '{}'),
    ('software_advice', 'SentinelOne',          'SentinelOne',               'container-security/sentinelone-profile',                'Cybersecurity', 15, 3, '{}'),
    ('software_advice', 'Fortinet',             'FortiGate NGFW',            'firewall/fortigate-next-generation-firewalls-profile',  'Cybersecurity', 15, 3, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;

