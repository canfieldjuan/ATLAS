-- Deterministic replay key for Content Ops campaign reasoning contexts.
--
-- Migration 277 introduced campaign_reasoning_contexts with insert-only
-- writes. Hosts can rerun the same reasoning-context ETL, so the table needs
-- a stable uniqueness contract for (account_id, target_mode, logical selector
-- set). selector_key is the sorted-selector MD5 used by the extracted
-- Postgres adapter; MD5 is only a compact deterministic key, not a security
-- primitive.

ALTER TABLE campaign_reasoning_contexts
    ADD COLUMN IF NOT EXISTS selector_key TEXT;

UPDATE campaign_reasoning_contexts
SET selector_key = md5(
    array_to_string(
        ARRAY(
            SELECT DISTINCT selector_value
            FROM unnest(selectors) AS selector_item(selector_value)
            ORDER BY selector_value
        ),
        E'\x1f'
    )
)
WHERE selector_key IS NULL OR selector_key = '';

DELETE FROM campaign_reasoning_contexts AS stale
USING campaign_reasoning_contexts AS keep
WHERE stale.id <> keep.id
  AND stale.account_id = keep.account_id
  AND stale.target_mode = keep.target_mode
  AND stale.selector_key = keep.selector_key
  AND (
      keep.updated_at > stale.updated_at
      OR (
          keep.updated_at = stale.updated_at
          AND keep.created_at > stale.created_at
      )
      OR (
          keep.updated_at = stale.updated_at
          AND keep.created_at = stale.created_at
          AND keep.id::text > stale.id::text
      )
  );

ALTER TABLE campaign_reasoning_contexts
    ALTER COLUMN selector_key SET DEFAULT md5('');

ALTER TABLE campaign_reasoning_contexts
    ALTER COLUMN selector_key SET NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS idx_campaign_reasoning_contexts_scope_mode_selector_key
    ON campaign_reasoning_contexts (account_id, target_mode, selector_key);
