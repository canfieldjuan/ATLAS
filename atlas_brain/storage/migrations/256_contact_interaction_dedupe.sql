-- Add durable CRM interaction dedupe to prevent repeated reasoning spend

ALTER TABLE contact_interactions
    ADD COLUMN IF NOT EXISTS interaction_dedupe_key TEXT;

WITH prepared AS (
    SELECT
        id,
        md5(
            'daily|' ||
            lower(COALESCE(interaction_type, '')) || '|' ||
            ((occurred_at AT TIME ZONE 'UTC')::date)::text || '|' ||
            lower(COALESCE(intent, '')) || '|' ||
            left(lower(regexp_replace(btrim(COALESCE(summary, '')), '\s+', ' ', 'g')), 2000)
        ) AS dedupe_key,
        row_number() OVER (
            PARTITION BY
                contact_id,
                lower(COALESCE(interaction_type, '')),
                ((occurred_at AT TIME ZONE 'UTC')::date),
                lower(COALESCE(intent, '')),
                left(lower(regexp_replace(btrim(COALESCE(summary, '')), '\s+', ' ', 'g')), 2000)
            ORDER BY created_at DESC NULLS LAST, id DESC
        ) AS rn
    FROM contact_interactions
    WHERE interaction_dedupe_key IS NULL
      AND NULLIF(btrim(COALESCE(summary, '')), '') IS NOT NULL
)
UPDATE contact_interactions AS ci
SET interaction_dedupe_key = prepared.dedupe_key
FROM prepared
WHERE ci.id = prepared.id
  AND prepared.rn = 1;

CREATE UNIQUE INDEX IF NOT EXISTS idx_contact_interactions_dedupe
    ON contact_interactions(contact_id, interaction_type, interaction_dedupe_key)
    WHERE interaction_dedupe_key IS NOT NULL;
