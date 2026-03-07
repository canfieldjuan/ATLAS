-- Seed phase 2 scrape sources: 8 new source types across 54 vendors
-- Sources: gartner, trustpilot, getapp, producthunt, youtube, quora, stackoverflow, peerspot
--
-- 54 vendors x 8 sources = 432 new targets.
-- Idempotent: ON CONFLICT (source, product_slug) DO NOTHING.

-- =====================================================================
-- GARTNER (max_pages=3, priority=5)
-- Slug format: {gartner-market}/{g2_slug}
-- =====================================================================

-- CRM vendors
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, max_pages, priority, metadata)
VALUES
    ('gartner', 'Close',       'Close',       'crm/close',           'CRM', 3, 5, '{}'),
    ('gartner', 'Copper',      'Copper',      'crm/copper',          'CRM', 3, 5, '{}'),
    ('gartner', 'Freshsales',  'Freshsales',  'crm/freshsales',      'CRM', 3, 5, '{}'),
    ('gartner', 'Insightly',   'Insightly',   'crm/insightly',       'CRM', 3, 5, '{}'),
    ('gartner', 'Nutshell',    'Nutshell',    'crm/nutshell',        'CRM', 3, 5, '{}'),
    ('gartner', 'Pipedrive',   'Pipedrive',   'crm/pipedrive',       'CRM', 3, 5, '{}'),
    ('gartner', 'Salesforce',  'Salesforce',  'crm/salesforce-crm',  'CRM', 3, 5, '{}'),
    ('gartner', 'Zoho CRM',   'Zoho CRM',    'crm/zoho-crm',        'CRM', 3, 5, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;

-- Helpdesk vendors
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, max_pages, priority, metadata)
VALUES
    ('gartner', 'Freshdesk',           'Freshdesk',           'it-service-management-tools/freshdesk',           'Helpdesk', 3, 5, '{}'),
    ('gartner', 'HappyFox',            'HappyFox',            'it-service-management-tools/happyfox',            'Helpdesk', 3, 5, '{}'),
    ('gartner', 'Help Scout',          'Help Scout',          'it-service-management-tools/help-scout',          'Helpdesk', 3, 5, '{}'),
    ('gartner', 'HubSpot Service Hub', 'HubSpot Service Hub', 'it-service-management-tools/hubspot-service-hub', 'Helpdesk', 3, 5, '{}'),
    ('gartner', 'Intercom',            'Intercom',            'it-service-management-tools/intercom',            'Helpdesk', 3, 5, '{}'),
    ('gartner', 'Zendesk',             'Zendesk',             'it-service-management-tools/zendesk',             'Helpdesk', 3, 5, '{}'),
    ('gartner', 'Zoho Desk',           'Zoho Desk',           'it-service-management-tools/zoho-desk',           'Helpdesk', 3, 5, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;

-- Project Management vendors
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, max_pages, priority, metadata)
VALUES
    ('gartner', 'Asana',       'Asana',       'project-and-portfolio-management/asana',       'Project Management', 3, 5, '{}'),
    ('gartner', 'Basecamp',    'Basecamp',    'project-and-portfolio-management/basecamp',    'Project Management', 3, 5, '{}'),
    ('gartner', 'ClickUp',     'ClickUp',     'project-and-portfolio-management/clickup',     'Project Management', 3, 5, '{}'),
    ('gartner', 'Monday.com',  'Monday.com',  'project-and-portfolio-management/monday-com',  'Project Management', 3, 5, '{}'),
    ('gartner', 'Notion',      'Notion',      'project-and-portfolio-management/notion',      'Project Management', 3, 5, '{}'),
    ('gartner', 'Smartsheet',  'Smartsheet',  'project-and-portfolio-management/smartsheet',  'Project Management', 3, 5, '{}'),
    ('gartner', 'Teamwork',    'Teamwork',    'project-and-portfolio-management/teamwork',    'Project Management', 3, 5, '{}'),
    ('gartner', 'Wrike',       'Wrike',       'project-and-portfolio-management/wrike',       'Project Management', 3, 5, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;

-- Marketing Automation vendors
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, max_pages, priority, metadata)
VALUES
    ('gartner', 'ActiveCampaign',       'ActiveCampaign',       'multichannel-marketing-hubs/activecampaign',       'Marketing Automation', 3, 5, '{}'),
    ('gartner', 'Brevo',                'Brevo',                'multichannel-marketing-hubs/brevo',                'Marketing Automation', 3, 5, '{}'),
    ('gartner', 'GetResponse',          'GetResponse',          'multichannel-marketing-hubs/getresponse',          'Marketing Automation', 3, 5, '{}'),
    ('gartner', 'HubSpot Marketing Hub','HubSpot Marketing Hub','multichannel-marketing-hubs/hubspot-marketing-hub','Marketing Automation', 3, 5, '{}'),
    ('gartner', 'Klaviyo',              'Klaviyo',              'multichannel-marketing-hubs/klaviyo',              'Marketing Automation', 3, 5, '{}'),
    ('gartner', 'Mailchimp',            'Mailchimp',            'multichannel-marketing-hubs/mailchimp',            'Marketing Automation', 3, 5, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;

-- Cloud Infrastructure vendors
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, max_pages, priority, metadata)
VALUES
    ('gartner', 'AWS',          'AWS',          'cloud-infrastructure-and-platform-services/amazon-web-services',  'Cloud Infrastructure', 3, 5, '{}'),
    ('gartner', 'Azure',        'Azure',        'cloud-infrastructure-and-platform-services/microsoft-azure',      'Cloud Infrastructure', 3, 5, '{}'),
    ('gartner', 'DigitalOcean', 'DigitalOcean', 'cloud-infrastructure-and-platform-services/digitalocean',         'Cloud Infrastructure', 3, 5, '{}'),
    ('gartner', 'Google Cloud', 'Google Cloud', 'cloud-infrastructure-and-platform-services/google-cloud-platform','Cloud Infrastructure', 3, 5, '{}'),
    ('gartner', 'Linode',       'Linode',       'cloud-infrastructure-and-platform-services/linode',               'Cloud Infrastructure', 3, 5, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;

-- E-commerce vendors
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, max_pages, priority, metadata)
VALUES
    ('gartner', 'BigCommerce', 'BigCommerce', 'digital-commerce/bigcommerce', 'E-commerce', 3, 5, '{}'),
    ('gartner', 'Magento',     'Magento',     'digital-commerce/magento',     'E-commerce', 3, 5, '{}'),
    ('gartner', 'Shopify',     'Shopify',     'digital-commerce/shopify',     'E-commerce', 3, 5, '{}'),
    ('gartner', 'WooCommerce', 'WooCommerce', 'digital-commerce/woocommerce', 'E-commerce', 3, 5, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;

-- HR / HCM vendors
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, max_pages, priority, metadata)
VALUES
    ('gartner', 'BambooHR', 'BambooHR', 'cloud-hcm-suites-for-midmarket-and-large-enterprises/bamboohr',  'HR / HCM', 3, 5, '{}'),
    ('gartner', 'Gusto',    'Gusto',    'cloud-hcm-suites-for-midmarket-and-large-enterprises/gusto',     'HR / HCM', 3, 5, '{}'),
    ('gartner', 'Rippling', 'Rippling', 'cloud-hcm-suites-for-midmarket-and-large-enterprises/rippling',  'HR / HCM', 3, 5, '{}'),
    ('gartner', 'Workday',  'Workday',  'cloud-hcm-suites-for-midmarket-and-large-enterprises/workday',   'HR / HCM', 3, 5, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;

-- Cybersecurity vendors
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, max_pages, priority, metadata)
VALUES
    ('gartner', 'CrowdStrike',          'CrowdStrike Falcon',          'endpoint-protection-platforms/crowdstrike-falcon',                'Cybersecurity', 3, 5, '{}'),
    ('gartner', 'Fortinet',             'Fortinet FortiGate',          'endpoint-protection-platforms/fortinet-fortigate',                'Cybersecurity', 3, 5, '{}'),
    ('gartner', 'Palo Alto Networks',   'Palo Alto Cortex XDR',        'endpoint-protection-platforms/palo-alto-networks-cortex-xdr',    'Cybersecurity', 3, 5, '{}'),
    ('gartner', 'SentinelOne',          'SentinelOne',                 'endpoint-protection-platforms/sentinelone',                       'Cybersecurity', 3, 5, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;

-- Communication vendors
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, max_pages, priority, metadata)
VALUES
    ('gartner', 'Microsoft Teams', 'Microsoft Teams', 'unified-communications-as-a-service/microsoft-teams',  'Communication', 3, 5, '{}'),
    ('gartner', 'RingCentral',     'RingCentral',     'unified-communications-as-a-service/ringcentral',      'Communication', 3, 5, '{}'),
    ('gartner', 'Slack',           'Slack',           'unified-communications-as-a-service/slack',             'Communication', 3, 5, '{}'),
    ('gartner', 'Zoom',            'Zoom',            'unified-communications-as-a-service/zoom',              'Communication', 3, 5, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;

-- Data & Analytics vendors
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, max_pages, priority, metadata)
VALUES
    ('gartner', 'Looker',    'Looker',    'analytics-and-business-intelligence-platforms/looker',              'Data & Analytics', 3, 5, '{}'),
    ('gartner', 'Metabase',  'Metabase',  'analytics-and-business-intelligence-platforms/metabase',            'Data & Analytics', 3, 5, '{}'),
    ('gartner', 'Power BI',  'Power BI',  'analytics-and-business-intelligence-platforms/microsoft-power-bi',  'Data & Analytics', 3, 5, '{}'),
    ('gartner', 'Tableau',   'Tableau',   'analytics-and-business-intelligence-platforms/tableau',             'Data & Analytics', 3, 5, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;


-- =====================================================================
-- TRUSTPILOT (max_pages=5, priority=5)
-- Slug = company domain
-- =====================================================================

INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, max_pages, priority, metadata)
VALUES
    -- CRM
    ('trustpilot', 'Close',       'Close',       'close.com',        'CRM', 5, 5, '{}'),
    ('trustpilot', 'Copper',      'Copper',      'copper.com',       'CRM', 5, 5, '{}'),
    ('trustpilot', 'Freshsales',  'Freshsales',  'freshworks.com',   'CRM', 5, 5, '{}'),
    ('trustpilot', 'Insightly',   'Insightly',   'insightly.com',    'CRM', 5, 5, '{}'),
    ('trustpilot', 'Nutshell',    'Nutshell',    'nutshell.com',     'CRM', 5, 5, '{}'),
    ('trustpilot', 'Pipedrive',   'Pipedrive',   'pipedrive.com',    'CRM', 5, 5, '{}'),
    ('trustpilot', 'Salesforce',  'Salesforce',  'salesforce.com',   'CRM', 5, 5, '{}'),
    ('trustpilot', 'Zoho CRM',   'Zoho CRM',    'zoho.com',         'CRM', 5, 5, '{}'),
    -- Helpdesk
    ('trustpilot', 'Freshdesk',           'Freshdesk',           'freshdesk.com',    'Helpdesk', 5, 5, '{}'),
    ('trustpilot', 'HappyFox',            'HappyFox',            'happyfox.com',     'Helpdesk', 5, 5, '{}'),
    ('trustpilot', 'Help Scout',          'Help Scout',          'helpscout.com',    'Helpdesk', 5, 5, '{}'),
    ('trustpilot', 'HubSpot Service Hub', 'HubSpot Service Hub', 'hubspot.com',      'Helpdesk', 5, 5, '{}'),
    ('trustpilot', 'Intercom',            'Intercom',            'intercom.com',     'Helpdesk', 5, 5, '{}'),
    ('trustpilot', 'Zendesk',             'Zendesk',             'zendesk.com',      'Helpdesk', 5, 5, '{}'),
    ('trustpilot', 'Zoho Desk',           'Zoho Desk',           'zoho.com/desk',    'Helpdesk', 5, 5, '{}'),
    -- Project Management
    ('trustpilot', 'Asana',       'Asana',       'asana.com',        'Project Management', 5, 5, '{}'),
    ('trustpilot', 'Basecamp',    'Basecamp',    'basecamp.com',     'Project Management', 5, 5, '{}'),
    ('trustpilot', 'ClickUp',     'ClickUp',     'clickup.com',      'Project Management', 5, 5, '{}'),
    ('trustpilot', 'Monday.com',  'Monday.com',  'monday.com',       'Project Management', 5, 5, '{}'),
    ('trustpilot', 'Notion',      'Notion',      'notion.so',        'Project Management', 5, 5, '{}'),
    ('trustpilot', 'Smartsheet',  'Smartsheet',  'smartsheet.com',   'Project Management', 5, 5, '{}'),
    ('trustpilot', 'Teamwork',    'Teamwork',    'teamwork.com',     'Project Management', 5, 5, '{}'),
    ('trustpilot', 'Wrike',       'Wrike',       'wrike.com',        'Project Management', 5, 5, '{}'),
    -- Marketing Automation
    ('trustpilot', 'ActiveCampaign',        'ActiveCampaign',        'activecampaign.com',  'Marketing Automation', 5, 5, '{}'),
    ('trustpilot', 'Brevo',                 'Brevo',                 'brevo.com',           'Marketing Automation', 5, 5, '{}'),
    ('trustpilot', 'GetResponse',           'GetResponse',           'getresponse.com',     'Marketing Automation', 5, 5, '{}'),
    ('trustpilot', 'HubSpot Marketing Hub', 'HubSpot Marketing Hub', 'hubspot.com/marketing','Marketing Automation', 5, 5, '{}'),
    ('trustpilot', 'Klaviyo',               'Klaviyo',               'klaviyo.com',         'Marketing Automation', 5, 5, '{}'),
    ('trustpilot', 'Mailchimp',             'Mailchimp',             'mailchimp.com',       'Marketing Automation', 5, 5, '{}'),
    -- Cloud Infrastructure
    ('trustpilot', 'AWS',          'AWS',          'aws.amazon.com',       'Cloud Infrastructure', 5, 5, '{}'),
    ('trustpilot', 'Azure',        'Azure',        'azure.microsoft.com',  'Cloud Infrastructure', 5, 5, '{}'),
    ('trustpilot', 'DigitalOcean', 'DigitalOcean', 'digitalocean.com',     'Cloud Infrastructure', 5, 5, '{}'),
    ('trustpilot', 'Google Cloud', 'Google Cloud', 'cloud.google.com',     'Cloud Infrastructure', 5, 5, '{}'),
    ('trustpilot', 'Linode',       'Linode',       'linode.com',           'Cloud Infrastructure', 5, 5, '{}'),
    -- E-commerce
    ('trustpilot', 'BigCommerce', 'BigCommerce', 'bigcommerce.com',  'E-commerce', 5, 5, '{}'),
    ('trustpilot', 'Magento',     'Magento',     'magento.com',      'E-commerce', 5, 5, '{}'),
    ('trustpilot', 'Shopify',     'Shopify',     'shopify.com',      'E-commerce', 5, 5, '{}'),
    ('trustpilot', 'WooCommerce', 'WooCommerce', 'woocommerce.com',  'E-commerce', 5, 5, '{}'),
    -- HR / HCM
    ('trustpilot', 'BambooHR', 'BambooHR', 'bamboohr.com',  'HR / HCM', 5, 5, '{}'),
    ('trustpilot', 'Gusto',    'Gusto',    'gusto.com',     'HR / HCM', 5, 5, '{}'),
    ('trustpilot', 'Rippling', 'Rippling', 'rippling.com',  'HR / HCM', 5, 5, '{}'),
    ('trustpilot', 'Workday',  'Workday',  'workday.com',   'HR / HCM', 5, 5, '{}'),
    -- Cybersecurity
    ('trustpilot', 'CrowdStrike',        'CrowdStrike Falcon',   'crowdstrike.com',       'Cybersecurity', 5, 5, '{}'),
    ('trustpilot', 'Fortinet',           'Fortinet FortiGate',   'fortinet.com',          'Cybersecurity', 5, 5, '{}'),
    ('trustpilot', 'Palo Alto Networks', 'Palo Alto Cortex XDR', 'paloaltonetworks.com',  'Cybersecurity', 5, 5, '{}'),
    ('trustpilot', 'SentinelOne',        'SentinelOne',          'sentinelone.com',       'Cybersecurity', 5, 5, '{}'),
    -- Communication
    ('trustpilot', 'Microsoft Teams', 'Microsoft Teams', 'microsoft.com/teams',  'Communication', 5, 5, '{}'),
    ('trustpilot', 'RingCentral',     'RingCentral',     'ringcentral.com',      'Communication', 5, 5, '{}'),
    ('trustpilot', 'Slack',           'Slack',           'slack.com',            'Communication', 5, 5, '{}'),
    ('trustpilot', 'Zoom',            'Zoom',            'zoom.us',             'Communication', 5, 5, '{}'),
    -- Data & Analytics
    ('trustpilot', 'Looker',   'Looker',   'looker.com',               'Data & Analytics', 5, 5, '{}'),
    ('trustpilot', 'Metabase', 'Metabase', 'metabase.com',             'Data & Analytics', 5, 5, '{}'),
    ('trustpilot', 'Power BI', 'Power BI', 'powerbi.microsoft.com',    'Data & Analytics', 5, 5, '{}'),
    ('trustpilot', 'Tableau',  'Tableau',  'tableau.com',              'Data & Analytics', 5, 5, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;


-- =====================================================================
-- GETAPP (max_pages=3, priority=5)
-- Slug format: {getapp-category}/a/{product-slug}
-- =====================================================================

INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, max_pages, priority, metadata)
VALUES
    -- CRM
    ('getapp', 'Close',       'Close',       'crm-software/a/close',       'CRM', 3, 5, '{}'),
    ('getapp', 'Copper',      'Copper',      'crm-software/a/copper',      'CRM', 3, 5, '{}'),
    ('getapp', 'Freshsales',  'Freshsales',  'crm-software/a/freshsales',  'CRM', 3, 5, '{}'),
    ('getapp', 'Insightly',   'Insightly',   'crm-software/a/insightly',   'CRM', 3, 5, '{}'),
    ('getapp', 'Nutshell',    'Nutshell',    'crm-software/a/nutshell',    'CRM', 3, 5, '{}'),
    ('getapp', 'Pipedrive',   'Pipedrive',   'crm-software/a/pipedrive',   'CRM', 3, 5, '{}'),
    ('getapp', 'Salesforce',  'Salesforce',  'crm-software/a/salesforce-crm','CRM', 3, 5, '{}'),
    ('getapp', 'Zoho CRM',   'Zoho CRM',    'crm-software/a/zoho-crm',    'CRM', 3, 5, '{}'),
    -- Helpdesk
    ('getapp', 'Freshdesk',           'Freshdesk',           'help-desk-software/a/freshdesk',           'Helpdesk', 3, 5, '{}'),
    ('getapp', 'HappyFox',            'HappyFox',            'help-desk-software/a/happyfox',            'Helpdesk', 3, 5, '{}'),
    ('getapp', 'Help Scout',          'Help Scout',          'help-desk-software/a/help-scout',          'Helpdesk', 3, 5, '{}'),
    ('getapp', 'HubSpot Service Hub', 'HubSpot Service Hub', 'help-desk-software/a/hubspot-service-hub', 'Helpdesk', 3, 5, '{}'),
    ('getapp', 'Intercom',            'Intercom',            'help-desk-software/a/intercom',            'Helpdesk', 3, 5, '{}'),
    ('getapp', 'Zendesk',             'Zendesk',             'help-desk-software/a/zendesk',             'Helpdesk', 3, 5, '{}'),
    ('getapp', 'Zoho Desk',           'Zoho Desk',           'help-desk-software/a/zoho-desk',           'Helpdesk', 3, 5, '{}'),
    -- Project Management
    ('getapp', 'Asana',       'Asana',       'project-management-software/a/asana',       'Project Management', 3, 5, '{}'),
    ('getapp', 'Basecamp',    'Basecamp',    'project-management-software/a/basecamp',    'Project Management', 3, 5, '{}'),
    ('getapp', 'ClickUp',     'ClickUp',     'project-management-software/a/clickup',     'Project Management', 3, 5, '{}'),
    ('getapp', 'Monday.com',  'Monday.com',  'project-management-software/a/monday-com',  'Project Management', 3, 5, '{}'),
    ('getapp', 'Notion',      'Notion',      'project-management-software/a/notion',      'Project Management', 3, 5, '{}'),
    ('getapp', 'Smartsheet',  'Smartsheet',  'project-management-software/a/smartsheet',  'Project Management', 3, 5, '{}'),
    ('getapp', 'Teamwork',    'Teamwork',    'project-management-software/a/teamwork',    'Project Management', 3, 5, '{}'),
    ('getapp', 'Wrike',       'Wrike',       'project-management-software/a/wrike',       'Project Management', 3, 5, '{}'),
    -- Marketing Automation
    ('getapp', 'ActiveCampaign',        'ActiveCampaign',        'marketing-automation-software/a/activecampaign',        'Marketing Automation', 3, 5, '{}'),
    ('getapp', 'Brevo',                 'Brevo',                 'marketing-automation-software/a/brevo',                 'Marketing Automation', 3, 5, '{}'),
    ('getapp', 'GetResponse',           'GetResponse',           'marketing-automation-software/a/getresponse',           'Marketing Automation', 3, 5, '{}'),
    ('getapp', 'HubSpot Marketing Hub', 'HubSpot Marketing Hub', 'marketing-automation-software/a/hubspot-marketing-hub', 'Marketing Automation', 3, 5, '{}'),
    ('getapp', 'Klaviyo',               'Klaviyo',               'marketing-automation-software/a/klaviyo',               'Marketing Automation', 3, 5, '{}'),
    ('getapp', 'Mailchimp',             'Mailchimp',             'marketing-automation-software/a/mailchimp',             'Marketing Automation', 3, 5, '{}'),
    -- Cloud Infrastructure
    ('getapp', 'AWS',          'AWS',          'cloud-management-software/a/amazon-web-services',   'Cloud Infrastructure', 3, 5, '{}'),
    ('getapp', 'Azure',        'Azure',        'cloud-management-software/a/microsoft-azure',       'Cloud Infrastructure', 3, 5, '{}'),
    ('getapp', 'DigitalOcean', 'DigitalOcean', 'cloud-management-software/a/digitalocean',          'Cloud Infrastructure', 3, 5, '{}'),
    ('getapp', 'Google Cloud', 'Google Cloud', 'cloud-management-software/a/google-cloud-platform', 'Cloud Infrastructure', 3, 5, '{}'),
    ('getapp', 'Linode',       'Linode',       'cloud-management-software/a/linode',                'Cloud Infrastructure', 3, 5, '{}'),
    -- E-commerce
    ('getapp', 'BigCommerce', 'BigCommerce', 'e-commerce-software/a/bigcommerce', 'E-commerce', 3, 5, '{}'),
    ('getapp', 'Magento',     'Magento',     'e-commerce-software/a/magento',     'E-commerce', 3, 5, '{}'),
    ('getapp', 'Shopify',     'Shopify',     'e-commerce-software/a/shopify',     'E-commerce', 3, 5, '{}'),
    ('getapp', 'WooCommerce', 'WooCommerce', 'e-commerce-software/a/woocommerce', 'E-commerce', 3, 5, '{}'),
    -- HR / HCM
    ('getapp', 'BambooHR', 'BambooHR', 'human-resource-software/a/bamboohr',  'HR / HCM', 3, 5, '{}'),
    ('getapp', 'Gusto',    'Gusto',    'human-resource-software/a/gusto',     'HR / HCM', 3, 5, '{}'),
    ('getapp', 'Rippling', 'Rippling', 'human-resource-software/a/rippling',  'HR / HCM', 3, 5, '{}'),
    ('getapp', 'Workday',  'Workday',  'human-resource-software/a/workday',   'HR / HCM', 3, 5, '{}'),
    -- Cybersecurity
    ('getapp', 'CrowdStrike',        'CrowdStrike Falcon',   'endpoint-protection-software/a/crowdstrike-falcon',             'Cybersecurity', 3, 5, '{}'),
    ('getapp', 'Fortinet',           'Fortinet FortiGate',   'endpoint-protection-software/a/fortinet-fortigate',             'Cybersecurity', 3, 5, '{}'),
    ('getapp', 'Palo Alto Networks', 'Palo Alto Cortex XDR', 'endpoint-protection-software/a/palo-alto-networks-cortex-xdr',  'Cybersecurity', 3, 5, '{}'),
    ('getapp', 'SentinelOne',        'SentinelOne',          'endpoint-protection-software/a/sentinelone',                    'Cybersecurity', 3, 5, '{}'),
    -- Communication
    ('getapp', 'Microsoft Teams', 'Microsoft Teams', 'collaboration-software/a/microsoft-teams',  'Communication', 3, 5, '{}'),
    ('getapp', 'RingCentral',     'RingCentral',     'collaboration-software/a/ringcentral',      'Communication', 3, 5, '{}'),
    ('getapp', 'Slack',           'Slack',           'collaboration-software/a/slack',             'Communication', 3, 5, '{}'),
    ('getapp', 'Zoom',            'Zoom',            'collaboration-software/a/zoom',              'Communication', 3, 5, '{}'),
    -- Data & Analytics
    ('getapp', 'Looker',   'Looker',   'business-intelligence-software/a/looker',              'Data & Analytics', 3, 5, '{}'),
    ('getapp', 'Metabase', 'Metabase', 'business-intelligence-software/a/metabase',            'Data & Analytics', 3, 5, '{}'),
    ('getapp', 'Power BI', 'Power BI', 'business-intelligence-software/a/microsoft-power-bi',  'Data & Analytics', 3, 5, '{}'),
    ('getapp', 'Tableau',  'Tableau',  'business-intelligence-software/a/tableau',             'Data & Analytics', 3, 5, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;


-- =====================================================================
-- PRODUCTHUNT (max_pages=3, priority=4)
-- Slug = Product Hunt product slug (simplified from g2_slug)
-- =====================================================================

INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, max_pages, priority, metadata)
VALUES
    -- CRM
    ('producthunt', 'Close',       'Close',       'close',       'CRM', 3, 4, '{}'),
    ('producthunt', 'Copper',      'Copper',      'copper',      'CRM', 3, 4, '{}'),
    ('producthunt', 'Freshsales',  'Freshsales',  'freshsales',  'CRM', 3, 4, '{}'),
    ('producthunt', 'Insightly',   'Insightly',   'insightly',   'CRM', 3, 4, '{}'),
    ('producthunt', 'Nutshell',    'Nutshell',    'nutshell',    'CRM', 3, 4, '{}'),
    ('producthunt', 'Pipedrive',   'Pipedrive',   'pipedrive',   'CRM', 3, 4, '{}'),
    ('producthunt', 'Salesforce',  'Salesforce',  'salesforce',  'CRM', 3, 4, '{}'),
    ('producthunt', 'Zoho CRM',   'Zoho CRM',    'zoho-crm',    'CRM', 3, 4, '{}'),
    -- Helpdesk
    ('producthunt', 'Freshdesk',           'Freshdesk',           'freshdesk',           'Helpdesk', 3, 4, '{}'),
    ('producthunt', 'HappyFox',            'HappyFox',            'happyfox',            'Helpdesk', 3, 4, '{}'),
    ('producthunt', 'Help Scout',          'Help Scout',          'help-scout',          'Helpdesk', 3, 4, '{}'),
    ('producthunt', 'HubSpot Service Hub', 'HubSpot Service Hub', 'hubspot-service-hub', 'Helpdesk', 3, 4, '{}'),
    ('producthunt', 'Intercom',            'Intercom',            'intercom',            'Helpdesk', 3, 4, '{}'),
    ('producthunt', 'Zendesk',             'Zendesk',             'zendesk',             'Helpdesk', 3, 4, '{}'),
    ('producthunt', 'Zoho Desk',           'Zoho Desk',           'zoho-desk',           'Helpdesk', 3, 4, '{}'),
    -- Project Management
    ('producthunt', 'Asana',       'Asana',       'asana',       'Project Management', 3, 4, '{}'),
    ('producthunt', 'Basecamp',    'Basecamp',    'basecamp',    'Project Management', 3, 4, '{}'),
    ('producthunt', 'ClickUp',     'ClickUp',     'clickup',     'Project Management', 3, 4, '{}'),
    ('producthunt', 'Monday.com',  'Monday.com',  'monday',      'Project Management', 3, 4, '{}'),
    ('producthunt', 'Notion',      'Notion',      'notion',      'Project Management', 3, 4, '{}'),
    ('producthunt', 'Smartsheet',  'Smartsheet',  'smartsheet',  'Project Management', 3, 4, '{}'),
    ('producthunt', 'Teamwork',    'Teamwork',    'teamwork',    'Project Management', 3, 4, '{}'),
    ('producthunt', 'Wrike',       'Wrike',       'wrike',       'Project Management', 3, 4, '{}'),
    -- Marketing Automation
    ('producthunt', 'ActiveCampaign',        'ActiveCampaign',        'activecampaign',        'Marketing Automation', 3, 4, '{}'),
    ('producthunt', 'Brevo',                 'Brevo',                 'brevo',                 'Marketing Automation', 3, 4, '{}'),
    ('producthunt', 'GetResponse',           'GetResponse',           'getresponse',           'Marketing Automation', 3, 4, '{}'),
    ('producthunt', 'HubSpot Marketing Hub', 'HubSpot Marketing Hub', 'hubspot-marketing-hub', 'Marketing Automation', 3, 4, '{}'),
    ('producthunt', 'Klaviyo',               'Klaviyo',               'klaviyo',               'Marketing Automation', 3, 4, '{}'),
    ('producthunt', 'Mailchimp',             'Mailchimp',             'mailchimp',             'Marketing Automation', 3, 4, '{}'),
    -- Cloud Infrastructure
    ('producthunt', 'AWS',          'AWS',          'amazon-web-services',   'Cloud Infrastructure', 3, 4, '{}'),
    ('producthunt', 'Azure',        'Azure',        'microsoft-azure',       'Cloud Infrastructure', 3, 4, '{}'),
    ('producthunt', 'DigitalOcean', 'DigitalOcean', 'digitalocean',          'Cloud Infrastructure', 3, 4, '{}'),
    ('producthunt', 'Google Cloud', 'Google Cloud', 'google-cloud-platform', 'Cloud Infrastructure', 3, 4, '{}'),
    ('producthunt', 'Linode',       'Linode',       'linode',                'Cloud Infrastructure', 3, 4, '{}'),
    -- E-commerce
    ('producthunt', 'BigCommerce', 'BigCommerce', 'bigcommerce', 'E-commerce', 3, 4, '{}'),
    ('producthunt', 'Magento',     'Magento',     'magento',     'E-commerce', 3, 4, '{}'),
    ('producthunt', 'Shopify',     'Shopify',     'shopify',     'E-commerce', 3, 4, '{}'),
    ('producthunt', 'WooCommerce', 'WooCommerce', 'woocommerce', 'E-commerce', 3, 4, '{}'),
    -- HR / HCM
    ('producthunt', 'BambooHR', 'BambooHR', 'bamboohr',  'HR / HCM', 3, 4, '{}'),
    ('producthunt', 'Gusto',    'Gusto',    'gusto',     'HR / HCM', 3, 4, '{}'),
    ('producthunt', 'Rippling', 'Rippling', 'rippling',  'HR / HCM', 3, 4, '{}'),
    ('producthunt', 'Workday',  'Workday',  'workday',   'HR / HCM', 3, 4, '{}'),
    -- Cybersecurity
    ('producthunt', 'CrowdStrike',        'CrowdStrike Falcon',   'crowdstrike-falcon',             'Cybersecurity', 3, 4, '{}'),
    ('producthunt', 'Fortinet',           'Fortinet FortiGate',   'fortinet',                       'Cybersecurity', 3, 4, '{}'),
    ('producthunt', 'Palo Alto Networks', 'Palo Alto Cortex XDR', 'palo-alto-networks',             'Cybersecurity', 3, 4, '{}'),
    ('producthunt', 'SentinelOne',        'SentinelOne',          'sentinelone',                    'Cybersecurity', 3, 4, '{}'),
    -- Communication
    ('producthunt', 'Microsoft Teams', 'Microsoft Teams', 'microsoft-teams',  'Communication', 3, 4, '{}'),
    ('producthunt', 'RingCentral',     'RingCentral',     'ringcentral',      'Communication', 3, 4, '{}'),
    ('producthunt', 'Slack',           'Slack',           'slack',            'Communication', 3, 4, '{}'),
    ('producthunt', 'Zoom',            'Zoom',            'zoom',             'Communication', 3, 4, '{}'),
    -- Data & Analytics
    ('producthunt', 'Looker',   'Looker',   'looker',              'Data & Analytics', 3, 4, '{}'),
    ('producthunt', 'Metabase', 'Metabase', 'metabase',            'Data & Analytics', 3, 4, '{}'),
    ('producthunt', 'Power BI', 'Power BI', 'microsoft-power-bi',  'Data & Analytics', 3, 4, '{}'),
    ('producthunt', 'Tableau',  'Tableau',  'tableau',             'Data & Analytics', 3, 4, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;


-- =====================================================================
-- YOUTUBE (max_pages=2, priority=3)
-- Slug = vendor name (used as search term)
-- =====================================================================

INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, max_pages, priority, metadata)
VALUES
    -- CRM
    ('youtube', 'Close',       'Close',       'Close',       'CRM', 2, 3, '{}'),
    ('youtube', 'Copper',      'Copper',      'Copper CRM',  'CRM', 2, 3, '{}'),
    ('youtube', 'Freshsales',  'Freshsales',  'Freshsales',  'CRM', 2, 3, '{}'),
    ('youtube', 'Insightly',   'Insightly',   'Insightly',   'CRM', 2, 3, '{}'),
    ('youtube', 'Nutshell',    'Nutshell',    'Nutshell CRM','CRM', 2, 3, '{}'),
    ('youtube', 'Pipedrive',   'Pipedrive',   'Pipedrive',   'CRM', 2, 3, '{}'),
    ('youtube', 'Salesforce',  'Salesforce',  'Salesforce',  'CRM', 2, 3, '{}'),
    ('youtube', 'Zoho CRM',   'Zoho CRM',    'Zoho CRM',    'CRM', 2, 3, '{}'),
    -- Helpdesk
    ('youtube', 'Freshdesk',           'Freshdesk',           'Freshdesk',           'Helpdesk', 2, 3, '{}'),
    ('youtube', 'HappyFox',            'HappyFox',            'HappyFox',            'Helpdesk', 2, 3, '{}'),
    ('youtube', 'Help Scout',          'Help Scout',          'Help Scout',          'Helpdesk', 2, 3, '{}'),
    ('youtube', 'HubSpot Service Hub', 'HubSpot Service Hub', 'HubSpot Service Hub', 'Helpdesk', 2, 3, '{}'),
    ('youtube', 'Intercom',            'Intercom',            'Intercom',            'Helpdesk', 2, 3, '{}'),
    ('youtube', 'Zendesk',             'Zendesk',             'Zendesk',             'Helpdesk', 2, 3, '{}'),
    ('youtube', 'Zoho Desk',           'Zoho Desk',           'Zoho Desk',           'Helpdesk', 2, 3, '{}'),
    -- Project Management
    ('youtube', 'Asana',       'Asana',       'Asana',       'Project Management', 2, 3, '{}'),
    ('youtube', 'Basecamp',    'Basecamp',    'Basecamp',    'Project Management', 2, 3, '{}'),
    ('youtube', 'ClickUp',     'ClickUp',     'ClickUp',     'Project Management', 2, 3, '{}'),
    ('youtube', 'Monday.com',  'Monday.com',  'Monday.com',  'Project Management', 2, 3, '{}'),
    ('youtube', 'Notion',      'Notion',      'Notion',      'Project Management', 2, 3, '{}'),
    ('youtube', 'Smartsheet',  'Smartsheet',  'Smartsheet',  'Project Management', 2, 3, '{}'),
    ('youtube', 'Teamwork',    'Teamwork',    'Teamwork',    'Project Management', 2, 3, '{}'),
    ('youtube', 'Wrike',       'Wrike',       'Wrike',       'Project Management', 2, 3, '{}'),
    -- Marketing Automation
    ('youtube', 'ActiveCampaign',        'ActiveCampaign',        'ActiveCampaign',        'Marketing Automation', 2, 3, '{}'),
    ('youtube', 'Brevo',                 'Brevo',                 'Brevo',                 'Marketing Automation', 2, 3, '{}'),
    ('youtube', 'GetResponse',           'GetResponse',           'GetResponse',           'Marketing Automation', 2, 3, '{}'),
    ('youtube', 'HubSpot Marketing Hub', 'HubSpot Marketing Hub', 'HubSpot Marketing Hub', 'Marketing Automation', 2, 3, '{}'),
    ('youtube', 'Klaviyo',               'Klaviyo',               'Klaviyo',               'Marketing Automation', 2, 3, '{}'),
    ('youtube', 'Mailchimp',             'Mailchimp',             'Mailchimp',             'Marketing Automation', 2, 3, '{}'),
    -- Cloud Infrastructure
    ('youtube', 'AWS',          'AWS',          'AWS',          'Cloud Infrastructure', 2, 3, '{}'),
    ('youtube', 'Azure',        'Azure',        'Azure',        'Cloud Infrastructure', 2, 3, '{}'),
    ('youtube', 'DigitalOcean', 'DigitalOcean', 'DigitalOcean', 'Cloud Infrastructure', 2, 3, '{}'),
    ('youtube', 'Google Cloud', 'Google Cloud', 'Google Cloud', 'Cloud Infrastructure', 2, 3, '{}'),
    ('youtube', 'Linode',       'Linode',       'Linode',       'Cloud Infrastructure', 2, 3, '{}'),
    -- E-commerce
    ('youtube', 'BigCommerce', 'BigCommerce', 'BigCommerce', 'E-commerce', 2, 3, '{}'),
    ('youtube', 'Magento',     'Magento',     'Magento',     'E-commerce', 2, 3, '{}'),
    ('youtube', 'Shopify',     'Shopify',     'Shopify',     'E-commerce', 2, 3, '{}'),
    ('youtube', 'WooCommerce', 'WooCommerce', 'WooCommerce', 'E-commerce', 2, 3, '{}'),
    -- HR / HCM
    ('youtube', 'BambooHR', 'BambooHR', 'BambooHR', 'HR / HCM', 2, 3, '{}'),
    ('youtube', 'Gusto',    'Gusto',    'Gusto',    'HR / HCM', 2, 3, '{}'),
    ('youtube', 'Rippling', 'Rippling', 'Rippling', 'HR / HCM', 2, 3, '{}'),
    ('youtube', 'Workday',  'Workday',  'Workday',  'HR / HCM', 2, 3, '{}'),
    -- Cybersecurity
    ('youtube', 'CrowdStrike',        'CrowdStrike Falcon',   'CrowdStrike',        'Cybersecurity', 2, 3, '{}'),
    ('youtube', 'Fortinet',           'Fortinet FortiGate',   'Fortinet',           'Cybersecurity', 2, 3, '{}'),
    ('youtube', 'Palo Alto Networks', 'Palo Alto Cortex XDR', 'Palo Alto Networks', 'Cybersecurity', 2, 3, '{}'),
    ('youtube', 'SentinelOne',        'SentinelOne',          'SentinelOne',        'Cybersecurity', 2, 3, '{}'),
    -- Communication
    ('youtube', 'Microsoft Teams', 'Microsoft Teams', 'Microsoft Teams', 'Communication', 2, 3, '{}'),
    ('youtube', 'RingCentral',     'RingCentral',     'RingCentral',     'Communication', 2, 3, '{}'),
    ('youtube', 'Slack',           'Slack',           'Slack',            'Communication', 2, 3, '{}'),
    ('youtube', 'Zoom',            'Zoom',            'Zoom',             'Communication', 2, 3, '{}'),
    -- Data & Analytics
    ('youtube', 'Looker',   'Looker',   'Looker',   'Data & Analytics', 2, 3, '{}'),
    ('youtube', 'Metabase', 'Metabase', 'Metabase', 'Data & Analytics', 2, 3, '{}'),
    ('youtube', 'Power BI', 'Power BI', 'Power BI', 'Data & Analytics', 2, 3, '{}'),
    ('youtube', 'Tableau',  'Tableau',  'Tableau',  'Data & Analytics', 2, 3, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;


-- =====================================================================
-- QUORA (max_pages=2, priority=3)
-- Slug = vendor name (used as search term)
-- =====================================================================

INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, max_pages, priority, metadata)
VALUES
    -- CRM
    ('quora', 'Close',       'Close',       'Close CRM',    'CRM', 2, 3, '{}'),
    ('quora', 'Copper',      'Copper',      'Copper CRM',   'CRM', 2, 3, '{}'),
    ('quora', 'Freshsales',  'Freshsales',  'Freshsales',   'CRM', 2, 3, '{}'),
    ('quora', 'Insightly',   'Insightly',   'Insightly',    'CRM', 2, 3, '{}'),
    ('quora', 'Nutshell',    'Nutshell',    'Nutshell CRM', 'CRM', 2, 3, '{}'),
    ('quora', 'Pipedrive',   'Pipedrive',   'Pipedrive',    'CRM', 2, 3, '{}'),
    ('quora', 'Salesforce',  'Salesforce',  'Salesforce',   'CRM', 2, 3, '{}'),
    ('quora', 'Zoho CRM',   'Zoho CRM',    'Zoho CRM',     'CRM', 2, 3, '{}'),
    -- Helpdesk
    ('quora', 'Freshdesk',           'Freshdesk',           'Freshdesk',           'Helpdesk', 2, 3, '{}'),
    ('quora', 'HappyFox',            'HappyFox',            'HappyFox',            'Helpdesk', 2, 3, '{}'),
    ('quora', 'Help Scout',          'Help Scout',          'Help Scout',          'Helpdesk', 2, 3, '{}'),
    ('quora', 'HubSpot Service Hub', 'HubSpot Service Hub', 'HubSpot Service Hub', 'Helpdesk', 2, 3, '{}'),
    ('quora', 'Intercom',            'Intercom',            'Intercom',            'Helpdesk', 2, 3, '{}'),
    ('quora', 'Zendesk',             'Zendesk',             'Zendesk',             'Helpdesk', 2, 3, '{}'),
    ('quora', 'Zoho Desk',           'Zoho Desk',           'Zoho Desk',           'Helpdesk', 2, 3, '{}'),
    -- Project Management
    ('quora', 'Asana',       'Asana',       'Asana',       'Project Management', 2, 3, '{}'),
    ('quora', 'Basecamp',    'Basecamp',    'Basecamp',    'Project Management', 2, 3, '{}'),
    ('quora', 'ClickUp',     'ClickUp',     'ClickUp',     'Project Management', 2, 3, '{}'),
    ('quora', 'Monday.com',  'Monday.com',  'Monday.com',  'Project Management', 2, 3, '{}'),
    ('quora', 'Notion',      'Notion',      'Notion',      'Project Management', 2, 3, '{}'),
    ('quora', 'Smartsheet',  'Smartsheet',  'Smartsheet',  'Project Management', 2, 3, '{}'),
    ('quora', 'Teamwork',    'Teamwork',    'Teamwork',    'Project Management', 2, 3, '{}'),
    ('quora', 'Wrike',       'Wrike',       'Wrike',       'Project Management', 2, 3, '{}'),
    -- Marketing Automation
    ('quora', 'ActiveCampaign',        'ActiveCampaign',        'ActiveCampaign',        'Marketing Automation', 2, 3, '{}'),
    ('quora', 'Brevo',                 'Brevo',                 'Brevo',                 'Marketing Automation', 2, 3, '{}'),
    ('quora', 'GetResponse',           'GetResponse',           'GetResponse',           'Marketing Automation', 2, 3, '{}'),
    ('quora', 'HubSpot Marketing Hub', 'HubSpot Marketing Hub', 'HubSpot Marketing Hub', 'Marketing Automation', 2, 3, '{}'),
    ('quora', 'Klaviyo',               'Klaviyo',               'Klaviyo',               'Marketing Automation', 2, 3, '{}'),
    ('quora', 'Mailchimp',             'Mailchimp',             'Mailchimp',             'Marketing Automation', 2, 3, '{}'),
    -- Cloud Infrastructure
    ('quora', 'AWS',          'AWS',          'AWS',          'Cloud Infrastructure', 2, 3, '{}'),
    ('quora', 'Azure',        'Azure',        'Azure',        'Cloud Infrastructure', 2, 3, '{}'),
    ('quora', 'DigitalOcean', 'DigitalOcean', 'DigitalOcean', 'Cloud Infrastructure', 2, 3, '{}'),
    ('quora', 'Google Cloud', 'Google Cloud', 'Google Cloud', 'Cloud Infrastructure', 2, 3, '{}'),
    ('quora', 'Linode',       'Linode',       'Linode',       'Cloud Infrastructure', 2, 3, '{}'),
    -- E-commerce
    ('quora', 'BigCommerce', 'BigCommerce', 'BigCommerce', 'E-commerce', 2, 3, '{}'),
    ('quora', 'Magento',     'Magento',     'Magento',     'E-commerce', 2, 3, '{}'),
    ('quora', 'Shopify',     'Shopify',     'Shopify',     'E-commerce', 2, 3, '{}'),
    ('quora', 'WooCommerce', 'WooCommerce', 'WooCommerce', 'E-commerce', 2, 3, '{}'),
    -- HR / HCM
    ('quora', 'BambooHR', 'BambooHR', 'BambooHR', 'HR / HCM', 2, 3, '{}'),
    ('quora', 'Gusto',    'Gusto',    'Gusto',    'HR / HCM', 2, 3, '{}'),
    ('quora', 'Rippling', 'Rippling', 'Rippling', 'HR / HCM', 2, 3, '{}'),
    ('quora', 'Workday',  'Workday',  'Workday',  'HR / HCM', 2, 3, '{}'),
    -- Cybersecurity
    ('quora', 'CrowdStrike',        'CrowdStrike Falcon',   'CrowdStrike',        'Cybersecurity', 2, 3, '{}'),
    ('quora', 'Fortinet',           'Fortinet FortiGate',   'Fortinet',           'Cybersecurity', 2, 3, '{}'),
    ('quora', 'Palo Alto Networks', 'Palo Alto Cortex XDR', 'Palo Alto Networks', 'Cybersecurity', 2, 3, '{}'),
    ('quora', 'SentinelOne',        'SentinelOne',          'SentinelOne',        'Cybersecurity', 2, 3, '{}'),
    -- Communication
    ('quora', 'Microsoft Teams', 'Microsoft Teams', 'Microsoft Teams', 'Communication', 2, 3, '{}'),
    ('quora', 'RingCentral',     'RingCentral',     'RingCentral',     'Communication', 2, 3, '{}'),
    ('quora', 'Slack',           'Slack',           'Slack',            'Communication', 2, 3, '{}'),
    ('quora', 'Zoom',            'Zoom',            'Zoom',             'Communication', 2, 3, '{}'),
    -- Data & Analytics
    ('quora', 'Looker',   'Looker',   'Looker',   'Data & Analytics', 2, 3, '{}'),
    ('quora', 'Metabase', 'Metabase', 'Metabase', 'Data & Analytics', 2, 3, '{}'),
    ('quora', 'Power BI', 'Power BI', 'Power BI', 'Data & Analytics', 2, 3, '{}'),
    ('quora', 'Tableau',  'Tableau',  'Tableau',  'Data & Analytics', 2, 3, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;


-- =====================================================================
-- STACKOVERFLOW (max_pages=2, priority=3)
-- Slug = vendor name (used as search term)
-- =====================================================================

INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, max_pages, priority, metadata)
VALUES
    -- CRM
    ('stackoverflow', 'Close',       'Close',       'Close CRM',    'CRM', 2, 3, '{}'),
    ('stackoverflow', 'Copper',      'Copper',      'Copper CRM',   'CRM', 2, 3, '{}'),
    ('stackoverflow', 'Freshsales',  'Freshsales',  'Freshsales',   'CRM', 2, 3, '{}'),
    ('stackoverflow', 'Insightly',   'Insightly',   'Insightly',    'CRM', 2, 3, '{}'),
    ('stackoverflow', 'Nutshell',    'Nutshell',    'Nutshell CRM', 'CRM', 2, 3, '{}'),
    ('stackoverflow', 'Pipedrive',   'Pipedrive',   'Pipedrive',    'CRM', 2, 3, '{}'),
    ('stackoverflow', 'Salesforce',  'Salesforce',  'Salesforce',   'CRM', 2, 3, '{}'),
    ('stackoverflow', 'Zoho CRM',   'Zoho CRM',    'Zoho CRM',     'CRM', 2, 3, '{}'),
    -- Helpdesk
    ('stackoverflow', 'Freshdesk',           'Freshdesk',           'Freshdesk',           'Helpdesk', 2, 3, '{}'),
    ('stackoverflow', 'HappyFox',            'HappyFox',            'HappyFox',            'Helpdesk', 2, 3, '{}'),
    ('stackoverflow', 'Help Scout',          'Help Scout',          'Help Scout',          'Helpdesk', 2, 3, '{}'),
    ('stackoverflow', 'HubSpot Service Hub', 'HubSpot Service Hub', 'HubSpot Service Hub', 'Helpdesk', 2, 3, '{}'),
    ('stackoverflow', 'Intercom',            'Intercom',            'Intercom',            'Helpdesk', 2, 3, '{}'),
    ('stackoverflow', 'Zendesk',             'Zendesk',             'Zendesk',             'Helpdesk', 2, 3, '{}'),
    ('stackoverflow', 'Zoho Desk',           'Zoho Desk',           'Zoho Desk',           'Helpdesk', 2, 3, '{}'),
    -- Project Management
    ('stackoverflow', 'Asana',       'Asana',       'Asana',       'Project Management', 2, 3, '{}'),
    ('stackoverflow', 'Basecamp',    'Basecamp',    'Basecamp',    'Project Management', 2, 3, '{}'),
    ('stackoverflow', 'ClickUp',     'ClickUp',     'ClickUp',     'Project Management', 2, 3, '{}'),
    ('stackoverflow', 'Monday.com',  'Monday.com',  'Monday.com',  'Project Management', 2, 3, '{}'),
    ('stackoverflow', 'Notion',      'Notion',      'Notion',      'Project Management', 2, 3, '{}'),
    ('stackoverflow', 'Smartsheet',  'Smartsheet',  'Smartsheet',  'Project Management', 2, 3, '{}'),
    ('stackoverflow', 'Teamwork',    'Teamwork',    'Teamwork',    'Project Management', 2, 3, '{}'),
    ('stackoverflow', 'Wrike',       'Wrike',       'Wrike',       'Project Management', 2, 3, '{}'),
    -- Marketing Automation
    ('stackoverflow', 'ActiveCampaign',        'ActiveCampaign',        'ActiveCampaign',        'Marketing Automation', 2, 3, '{}'),
    ('stackoverflow', 'Brevo',                 'Brevo',                 'Brevo',                 'Marketing Automation', 2, 3, '{}'),
    ('stackoverflow', 'GetResponse',           'GetResponse',           'GetResponse',           'Marketing Automation', 2, 3, '{}'),
    ('stackoverflow', 'HubSpot Marketing Hub', 'HubSpot Marketing Hub', 'HubSpot Marketing Hub', 'Marketing Automation', 2, 3, '{}'),
    ('stackoverflow', 'Klaviyo',               'Klaviyo',               'Klaviyo',               'Marketing Automation', 2, 3, '{}'),
    ('stackoverflow', 'Mailchimp',             'Mailchimp',             'Mailchimp',             'Marketing Automation', 2, 3, '{}'),
    -- Cloud Infrastructure
    ('stackoverflow', 'AWS',          'AWS',          'AWS',          'Cloud Infrastructure', 2, 3, '{}'),
    ('stackoverflow', 'Azure',        'Azure',        'Azure',        'Cloud Infrastructure', 2, 3, '{}'),
    ('stackoverflow', 'DigitalOcean', 'DigitalOcean', 'DigitalOcean', 'Cloud Infrastructure', 2, 3, '{}'),
    ('stackoverflow', 'Google Cloud', 'Google Cloud', 'Google Cloud', 'Cloud Infrastructure', 2, 3, '{}'),
    ('stackoverflow', 'Linode',       'Linode',       'Linode',       'Cloud Infrastructure', 2, 3, '{}'),
    -- E-commerce
    ('stackoverflow', 'BigCommerce', 'BigCommerce', 'BigCommerce', 'E-commerce', 2, 3, '{}'),
    ('stackoverflow', 'Magento',     'Magento',     'Magento',     'E-commerce', 2, 3, '{}'),
    ('stackoverflow', 'Shopify',     'Shopify',     'Shopify',     'E-commerce', 2, 3, '{}'),
    ('stackoverflow', 'WooCommerce', 'WooCommerce', 'WooCommerce', 'E-commerce', 2, 3, '{}'),
    -- HR / HCM
    ('stackoverflow', 'BambooHR', 'BambooHR', 'BambooHR', 'HR / HCM', 2, 3, '{}'),
    ('stackoverflow', 'Gusto',    'Gusto',    'Gusto',    'HR / HCM', 2, 3, '{}'),
    ('stackoverflow', 'Rippling', 'Rippling', 'Rippling', 'HR / HCM', 2, 3, '{}'),
    ('stackoverflow', 'Workday',  'Workday',  'Workday',  'HR / HCM', 2, 3, '{}'),
    -- Cybersecurity
    ('stackoverflow', 'CrowdStrike',        'CrowdStrike Falcon',   'CrowdStrike',        'Cybersecurity', 2, 3, '{}'),
    ('stackoverflow', 'Fortinet',           'Fortinet FortiGate',   'Fortinet',           'Cybersecurity', 2, 3, '{}'),
    ('stackoverflow', 'Palo Alto Networks', 'Palo Alto Cortex XDR', 'Palo Alto Networks', 'Cybersecurity', 2, 3, '{}'),
    ('stackoverflow', 'SentinelOne',        'SentinelOne',          'SentinelOne',        'Cybersecurity', 2, 3, '{}'),
    -- Communication
    ('stackoverflow', 'Microsoft Teams', 'Microsoft Teams', 'Microsoft Teams', 'Communication', 2, 3, '{}'),
    ('stackoverflow', 'RingCentral',     'RingCentral',     'RingCentral',     'Communication', 2, 3, '{}'),
    ('stackoverflow', 'Slack',           'Slack',           'Slack',            'Communication', 2, 3, '{}'),
    ('stackoverflow', 'Zoom',            'Zoom',            'Zoom',             'Communication', 2, 3, '{}'),
    -- Data & Analytics
    ('stackoverflow', 'Looker',   'Looker',   'Looker',   'Data & Analytics', 2, 3, '{}'),
    ('stackoverflow', 'Metabase', 'Metabase', 'Metabase', 'Data & Analytics', 2, 3, '{}'),
    ('stackoverflow', 'Power BI', 'Power BI', 'Power BI', 'Data & Analytics', 2, 3, '{}'),
    ('stackoverflow', 'Tableau',  'Tableau',  'Tableau',  'Data & Analytics', 2, 3, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;


-- =====================================================================
-- PEERSPOT (max_pages=3, priority=5)
-- Slug = PeerSpot URL slug (defaults to g2_slug)
-- =====================================================================

INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, max_pages, priority, metadata)
VALUES
    -- CRM
    ('peerspot', 'Close',       'Close',       'close',           'CRM', 3, 5, '{}'),
    ('peerspot', 'Copper',      'Copper',      'copper',          'CRM', 3, 5, '{}'),
    ('peerspot', 'Freshsales',  'Freshsales',  'freshsales',      'CRM', 3, 5, '{}'),
    ('peerspot', 'Insightly',   'Insightly',   'insightly',       'CRM', 3, 5, '{}'),
    ('peerspot', 'Nutshell',    'Nutshell',    'nutshell',        'CRM', 3, 5, '{}'),
    ('peerspot', 'Pipedrive',   'Pipedrive',   'pipedrive',       'CRM', 3, 5, '{}'),
    ('peerspot', 'Salesforce',  'Salesforce',  'salesforce-crm',  'CRM', 3, 5, '{}'),
    ('peerspot', 'Zoho CRM',   'Zoho CRM',    'zoho-crm',        'CRM', 3, 5, '{}'),
    -- Helpdesk
    ('peerspot', 'Freshdesk',           'Freshdesk',           'freshdesk',           'Helpdesk', 3, 5, '{}'),
    ('peerspot', 'HappyFox',            'HappyFox',            'happyfox',            'Helpdesk', 3, 5, '{}'),
    ('peerspot', 'Help Scout',          'Help Scout',          'help-scout',          'Helpdesk', 3, 5, '{}'),
    ('peerspot', 'HubSpot Service Hub', 'HubSpot Service Hub', 'hubspot-service-hub', 'Helpdesk', 3, 5, '{}'),
    ('peerspot', 'Intercom',            'Intercom',            'intercom',            'Helpdesk', 3, 5, '{}'),
    ('peerspot', 'Zendesk',             'Zendesk',             'zendesk',             'Helpdesk', 3, 5, '{}'),
    ('peerspot', 'Zoho Desk',           'Zoho Desk',           'zoho-desk',           'Helpdesk', 3, 5, '{}'),
    -- Project Management
    ('peerspot', 'Asana',       'Asana',       'asana',       'Project Management', 3, 5, '{}'),
    ('peerspot', 'Basecamp',    'Basecamp',    'basecamp',    'Project Management', 3, 5, '{}'),
    ('peerspot', 'ClickUp',     'ClickUp',     'clickup',     'Project Management', 3, 5, '{}'),
    ('peerspot', 'Monday.com',  'Monday.com',  'monday-com',  'Project Management', 3, 5, '{}'),
    ('peerspot', 'Notion',      'Notion',      'notion',      'Project Management', 3, 5, '{}'),
    ('peerspot', 'Smartsheet',  'Smartsheet',  'smartsheet',  'Project Management', 3, 5, '{}'),
    ('peerspot', 'Teamwork',    'Teamwork',    'teamwork',    'Project Management', 3, 5, '{}'),
    ('peerspot', 'Wrike',       'Wrike',       'wrike',       'Project Management', 3, 5, '{}'),
    -- Marketing Automation
    ('peerspot', 'ActiveCampaign',        'ActiveCampaign',        'activecampaign',        'Marketing Automation', 3, 5, '{}'),
    ('peerspot', 'Brevo',                 'Brevo',                 'brevo',                 'Marketing Automation', 3, 5, '{}'),
    ('peerspot', 'GetResponse',           'GetResponse',           'getresponse',           'Marketing Automation', 3, 5, '{}'),
    ('peerspot', 'HubSpot Marketing Hub', 'HubSpot Marketing Hub', 'hubspot-marketing-hub', 'Marketing Automation', 3, 5, '{}'),
    ('peerspot', 'Klaviyo',               'Klaviyo',               'klaviyo',               'Marketing Automation', 3, 5, '{}'),
    ('peerspot', 'Mailchimp',             'Mailchimp',             'mailchimp',             'Marketing Automation', 3, 5, '{}'),
    -- Cloud Infrastructure
    ('peerspot', 'AWS',          'AWS',          'amazon-web-services',   'Cloud Infrastructure', 3, 5, '{}'),
    ('peerspot', 'Azure',        'Azure',        'microsoft-azure',       'Cloud Infrastructure', 3, 5, '{}'),
    ('peerspot', 'DigitalOcean', 'DigitalOcean', 'digitalocean',          'Cloud Infrastructure', 3, 5, '{}'),
    ('peerspot', 'Google Cloud', 'Google Cloud', 'google-cloud-platform', 'Cloud Infrastructure', 3, 5, '{}'),
    ('peerspot', 'Linode',       'Linode',       'linode',                'Cloud Infrastructure', 3, 5, '{}'),
    -- E-commerce
    ('peerspot', 'BigCommerce', 'BigCommerce', 'bigcommerce', 'E-commerce', 3, 5, '{}'),
    ('peerspot', 'Magento',     'Magento',     'magento',     'E-commerce', 3, 5, '{}'),
    ('peerspot', 'Shopify',     'Shopify',     'shopify',     'E-commerce', 3, 5, '{}'),
    ('peerspot', 'WooCommerce', 'WooCommerce', 'woocommerce', 'E-commerce', 3, 5, '{}'),
    -- HR / HCM
    ('peerspot', 'BambooHR', 'BambooHR', 'bamboohr',  'HR / HCM', 3, 5, '{}'),
    ('peerspot', 'Gusto',    'Gusto',    'gusto',     'HR / HCM', 3, 5, '{}'),
    ('peerspot', 'Rippling', 'Rippling', 'rippling',  'HR / HCM', 3, 5, '{}'),
    ('peerspot', 'Workday',  'Workday',  'workday',   'HR / HCM', 3, 5, '{}'),
    -- Cybersecurity
    ('peerspot', 'CrowdStrike',        'CrowdStrike Falcon',   'crowdstrike-falcon',             'Cybersecurity', 3, 5, '{}'),
    ('peerspot', 'Fortinet',           'Fortinet FortiGate',   'fortinet-fortigate',             'Cybersecurity', 3, 5, '{}'),
    ('peerspot', 'Palo Alto Networks', 'Palo Alto Cortex XDR', 'palo-alto-networks-cortex-xdr',  'Cybersecurity', 3, 5, '{}'),
    ('peerspot', 'SentinelOne',        'SentinelOne',          'sentinelone',                    'Cybersecurity', 3, 5, '{}'),
    -- Communication
    ('peerspot', 'Microsoft Teams', 'Microsoft Teams', 'microsoft-teams',  'Communication', 3, 5, '{}'),
    ('peerspot', 'RingCentral',     'RingCentral',     'ringcentral',      'Communication', 3, 5, '{}'),
    ('peerspot', 'Slack',           'Slack',           'slack',            'Communication', 3, 5, '{}'),
    ('peerspot', 'Zoom',            'Zoom',            'zoom',             'Communication', 3, 5, '{}'),
    -- Data & Analytics
    ('peerspot', 'Looker',   'Looker',   'looker',              'Data & Analytics', 3, 5, '{}'),
    ('peerspot', 'Metabase', 'Metabase', 'metabase',            'Data & Analytics', 3, 5, '{}'),
    ('peerspot', 'Power BI', 'Power BI', 'microsoft-power-bi',  'Data & Analytics', 3, 5, '{}'),
    ('peerspot', 'Tableau',  'Tableau',  'tableau',             'Data & Analytics', 3, 5, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;
