-- Normalize entity names in entity_pressure_baselines
-- Strips corporate suffixes (Inc, Corp, Co, Ltd, LLC, etc.) to merge duplicates.
-- Keeps the row with the highest pressure_score for each (normalized_name, entity_type).

-- Step 1: Create a temp table mapping old names to normalized names
CREATE TEMP TABLE _entity_name_map AS
SELECT
    id,
    entity_name,
    entity_type,
    pressure_score,
    last_computed_at,
    -- Strip trailing corporate suffixes (matches _CORP_SUFFIXES in daily_intelligence.py)
    TRIM(REGEXP_REPLACE(
        entity_name,
        E',?\\s+(?:Inc\\.?|Corp\\.?|Corporation|Company|Co\\.?|Ltd\\.?|LLC|PLC|SA|AG|NV|SE|Group|Holdings?)\\.?$',
        '',
        'i'
    )) AS normalized_name
FROM entity_pressure_baselines;

-- Step 2: Delete duplicate rows, keeping the one with highest pressure_score
-- (tie-break by most recent last_computed_at, then by id)
DELETE FROM entity_pressure_baselines
WHERE id IN (
    SELECT id FROM (
        SELECT
            id,
            ROW_NUMBER() OVER (
                PARTITION BY normalized_name, entity_type
                ORDER BY pressure_score DESC, last_computed_at DESC NULLS LAST, id
            ) AS rn
        FROM _entity_name_map
    ) ranked
    WHERE rn > 1
);

-- Step 3: Update surviving rows to use the normalized name
UPDATE entity_pressure_baselines epb
SET entity_name = m.normalized_name
FROM _entity_name_map m
WHERE epb.id = m.id
  AND epb.entity_name <> m.normalized_name;

-- Cleanup
DROP TABLE _entity_name_map;
