-- Blog blueprints: pre-generated structural plans the AI Content Ops blog
-- post generator consumes when producing prose. The package's
-- BlogBlueprintRepository Protocol (extracted_content_pipeline/blog_ports.py)
-- previously had no implementation; this table backs the new
-- PostgresBlogBlueprintRepository so the bundle factory can wire blog_post.
--
-- The payload JSONB carries the rich blueprint dict (sections / charts /
-- tags / data_context / etc.) the LLM prompt consumes. Flattening into
-- typed columns would force a migration on every blueprint-shape tweak;
-- JSONB also lets multiple topic_types coexist without sparse columns.
--
-- consumed_at tracks single-use semantics: once the generator emits a
-- draft from a blueprint, the row is marked consumed and the default
-- read path filters it out. Hosts can clear consumed_at to re-run a
-- blueprint for regeneration / experimentation.

CREATE TABLE IF NOT EXISTS blog_blueprints (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id TEXT NOT NULL DEFAULT '',
    target_mode TEXT NOT NULL,
    topic_type TEXT NOT NULL,
    slug TEXT NOT NULL,
    suggested_title TEXT NOT NULL DEFAULT '',
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    consumed_at TIMESTAMPTZ
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_blog_blueprints_slug
    ON blog_blueprints (account_id, target_mode, slug);

CREATE INDEX IF NOT EXISTS idx_blog_blueprints_unconsumed
    ON blog_blueprints (account_id, target_mode, created_at DESC)
    WHERE consumed_at IS NULL;

CREATE INDEX IF NOT EXISTS idx_blog_blueprints_topic
    ON blog_blueprints (account_id, target_mode, topic_type, created_at DESC);
