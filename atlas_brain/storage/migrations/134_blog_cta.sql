ALTER TABLE blog_posts ADD COLUMN IF NOT EXISTS cta jsonb;

COMMENT ON COLUMN blog_posts.cta IS 'Structured CTA: {headline, body, button_text, report_type, vendor_filter, category_filter}';
