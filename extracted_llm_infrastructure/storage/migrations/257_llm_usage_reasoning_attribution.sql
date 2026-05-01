-- Add durable generic-reasoning attribution columns to llm_usage.

ALTER TABLE llm_usage
    ADD COLUMN IF NOT EXISTS source_name TEXT,
    ADD COLUMN IF NOT EXISTS event_type TEXT,
    ADD COLUMN IF NOT EXISTS entity_type TEXT,
    ADD COLUMN IF NOT EXISTS entity_id TEXT;

UPDATE llm_usage
SET source_name = metadata ->> 'source_name'
WHERE source_name IS NULL
  AND metadata ->> 'source_name' IS NOT NULL
  AND (metadata ->> 'source_name') != '';

UPDATE llm_usage
SET source_name = metadata #>> '{business,source_name}'
WHERE source_name IS NULL
  AND metadata #>> '{business,source_name}' IS NOT NULL
  AND (metadata #>> '{business,source_name}') != '';

UPDATE llm_usage
SET event_type = metadata ->> 'event_type'
WHERE event_type IS NULL
  AND metadata ->> 'event_type' IS NOT NULL
  AND (metadata ->> 'event_type') != '';

UPDATE llm_usage
SET event_type = metadata #>> '{business,event_type}'
WHERE event_type IS NULL
  AND metadata #>> '{business,event_type}' IS NOT NULL
  AND (metadata #>> '{business,event_type}') != '';

UPDATE llm_usage
SET entity_type = metadata ->> 'entity_type'
WHERE entity_type IS NULL
  AND metadata ->> 'entity_type' IS NOT NULL
  AND (metadata ->> 'entity_type') != '';

UPDATE llm_usage
SET entity_type = metadata #>> '{business,entity_type}'
WHERE entity_type IS NULL
  AND metadata #>> '{business,entity_type}' IS NOT NULL
  AND (metadata #>> '{business,entity_type}') != '';

UPDATE llm_usage
SET entity_id = metadata ->> 'entity_id'
WHERE entity_id IS NULL
  AND metadata ->> 'entity_id' IS NOT NULL
  AND (metadata ->> 'entity_id') != '';

UPDATE llm_usage
SET entity_id = metadata #>> '{business,entity_id}'
WHERE entity_id IS NULL
  AND metadata #>> '{business,entity_id}' IS NOT NULL
  AND (metadata #>> '{business,entity_id}') != '';

CREATE INDEX IF NOT EXISTS idx_llm_usage_source_created
    ON llm_usage (source_name, created_at DESC)
    WHERE source_name IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_llm_usage_event_created
    ON llm_usage (event_type, created_at DESC)
    WHERE event_type IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_llm_usage_entity_created
    ON llm_usage (entity_type, entity_id, created_at DESC)
    WHERE entity_id IS NOT NULL;
