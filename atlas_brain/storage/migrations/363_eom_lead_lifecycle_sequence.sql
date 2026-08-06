-- atlas: atomic-bookkeeping
-- Give every future EOM lead lifecycle row a database-owned append sequence.
--
-- Reopen needs the latest lead_lost evidence. Timestamps are not a safe ordering
-- key because PostgreSQL NOW() is the transaction-start time, and application
-- metadata is not safe across rolling deploys because old app versions will not
-- write it. A table default is compatible with both old and new writers after
-- this migration lands: every insert that omits lifecycle_sequence still receives
-- the next database-owned value.

CREATE SEQUENCE IF NOT EXISTS eom_lead_lifecycle_events_sequence_seq;

ALTER TABLE eom_lead_lifecycle_events
    ADD COLUMN IF NOT EXISTS lifecycle_sequence BIGINT;

ALTER TABLE eom_lead_lifecycle_events
    ALTER COLUMN lifecycle_sequence SET DEFAULT nextval('eom_lead_lifecycle_events_sequence_seq'::regclass);

ALTER SEQUENCE eom_lead_lifecycle_events_sequence_seq
    OWNED BY eom_lead_lifecycle_events.lifecycle_sequence;

COMMENT ON COLUMN eom_lead_lifecycle_events.lifecycle_sequence IS
    'Database-owned append ordering for lifecycle rows; compatible with old app writers that omit the column.';

-- Rollback evidence:
--   ALTER SEQUENCE eom_lead_lifecycle_events_sequence_seq OWNED BY NONE;
--   ALTER TABLE eom_lead_lifecycle_events ALTER COLUMN lifecycle_sequence DROP DEFAULT;
--   ALTER TABLE eom_lead_lifecycle_events DROP COLUMN lifecycle_sequence;
--   DROP SEQUENCE IF EXISTS eom_lead_lifecycle_events_sequence_seq;
