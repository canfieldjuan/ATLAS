-- 086: Seed Trello and Jira scrape targets across all sources
-- These are key PM competitors needed for Monday.com comparison content

-- ============================================================
-- TRELLO
-- ============================================================

-- URL-slug sources
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, max_pages, metadata)
VALUES
  ('g2',          'Trello', 'Trello', 'trello', 'Project Management', 5, '{}'),
  ('capterra',    'Trello', 'Trello', '211559/Trello', 'Project Management', 5, '{}'),
  ('trustradius', 'Trello', 'Trello', 'trello', 'Project Management', 5, '{}'),
  ('trustpilot',  'Trello', 'Trello', 'trello.com', 'Project Management', 3, '{}'),
  ('gartner',     'Trello', 'Trello', 'collaborative-work-management/atlassian', 'Project Management', 5, '{}'),
  ('getapp',      'Trello', 'Trello', 'project-management-planning-software/a/trello', 'Project Management', 5, '{}'),
  ('peerspot',    'Trello', 'Trello', 'trello', 'Project Management', 3, '{}'),
  ('producthunt', 'Trello', 'Trello', 'trello', 'Project Management', 3, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;

-- Search-based sources (slug is informational, vendor_name is the search term)
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, max_pages, metadata)
VALUES
  ('reddit',        'Trello', 'Trello', 'trello-reddit', 'Project Management', 5, '{}'),
  ('hackernews',    'Trello', 'Trello', 'trello-hn', 'Project Management', 3, '{}'),
  ('github',        'Trello', 'Trello', 'trello-github', 'Project Management', 3, '{}'),
  ('youtube',       'Trello', 'Trello', 'Trello', 'Project Management', 3, '{}'),
  ('stackoverflow', 'Trello', 'Trello', 'Trello', 'Project Management', 10, '{}'),
  ('quora',         'Trello', 'Trello', 'Trello', 'Project Management', 3, '{}'),
  ('twitter',       'Trello', 'Trello', 'trello-twitter', 'Project Management', 3, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;

-- RSS
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, max_pages, metadata)
VALUES
  ('rss', 'Trello', 'Trello',
   'https://news.google.com/rss/search?q=Trello+churn+OR+switching+OR+alternative+OR+migration&hl=en-US&gl=US&ceid=US:en',
   'Project Management', 1, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;

-- ============================================================
-- JIRA
-- ============================================================

-- URL-slug sources
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, max_pages, metadata)
VALUES
  ('g2',          'Jira', 'Jira', 'jira', 'Project Management', 5, '{}'),
  ('capterra',    'Jira', 'Jira', '19319/JIRA', 'Project Management', 5, '{}'),
  ('trustradius', 'Jira', 'Jira', 'atlassian-jira', 'Project Management', 5, '{}'),
  ('trustpilot',  'Jira', 'Jira', 'atlassian.com', 'Project Management', 3, '{}'),
  ('gartner',     'Jira', 'Jira', 'enterprise-agile-planning-tools/atlassian', 'Project Management', 5, '{}'),
  ('getapp',      'Jira', 'Jira', 'project-management-planning-software/a/jira', 'Project Management', 5, '{}'),
  ('peerspot',    'Jira', 'Jira', 'jira', 'Project Management', 3, '{}'),
  ('producthunt', 'Jira', 'Jira', 'jira', 'Project Management', 3, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;

-- Search-based sources
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, max_pages, metadata)
VALUES
  ('reddit',        'Jira', 'Jira', 'jira-reddit', 'Project Management', 5, '{}'),
  ('hackernews',    'Jira', 'Jira', 'jira-hn', 'Project Management', 3, '{}'),
  ('github',        'Jira', 'Jira', 'jira-github', 'Project Management', 3, '{}'),
  ('youtube',       'Jira', 'Jira', 'Jira', 'Project Management', 3, '{}'),
  ('stackoverflow', 'Jira', 'Jira', 'Jira', 'Project Management', 10, '{}'),
  ('quora',         'Jira', 'Jira', 'Jira', 'Project Management', 3, '{}'),
  ('twitter',       'Jira', 'Jira', 'jira-twitter', 'Project Management', 3, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;

-- RSS
INSERT INTO b2b_scrape_targets (source, vendor_name, product_name, product_slug, product_category, max_pages, metadata)
VALUES
  ('rss', 'Jira', 'Jira',
   'https://news.google.com/rss/search?q=Jira+churn+OR+switching+OR+alternative+OR+migration&hl=en-US&gl=US&ceid=US:en',
   'Project Management', 1, '{}')
ON CONFLICT (source, product_slug) DO NOTHING;
