-- LLM Gateway batch per-item idempotency: UNIQUE index (PR-D4d).
--
-- Companion to 319 (column adds + CHECK constraint). Split out
-- because ``CREATE INDEX CONCURRENTLY`` can't run inside a
-- transaction block -- atlas's migration runner sends each .sql
-- file as its own statement, so a single-statement file is the
-- guaranteed-safe pattern (mirrors migration 090).
--
-- Why CONCURRENTLY: regular CREATE UNIQUE INDEX takes
-- ACCESS EXCLUSIVE on llm_usage during the build, which blocks
-- live inserts for the duration. atlas's pipeline writes 100k+
-- rows/month into llm_usage; a multi-second lock during a
-- production deploy would stall request handling. CONCURRENTLY
-- builds in the background with only a SHARE UPDATE EXCLUSIVE
-- lock, which is compatible with concurrent INSERTs.
--
-- Why ``custom_id IS NOT NULL`` in the predicate (Copilot on
-- PR-D4d): the partial UNIQUE alone wouldn't dedup rows with
-- NULL custom_id (NULLs are distinct in PG UNIQUE indexes).
-- The CHECK in 319 already forbids that combination, but
-- including the predicate here is defense-in-depth -- if the
-- CHECK is ever disabled or NOT VALID, the index still won't
-- index NULL rows, so the failure mode is "insert succeeds,
-- doesn't dedup" rather than "insert succeeds, falsely passes
-- as unique."
--
-- Per-item idempotency end-to-end:
-- _persist_batch_usage emits INSERT ... ON CONFLICT
-- (account_id, batch_id, custom_id) DO NOTHING. A retry that
-- re-emits already-written rows no-ops at the DB level instead
-- of double-counting against the customer's billing rollup.

CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS uq_llm_usage_batch_item
    ON llm_usage (account_id, batch_id, custom_id)
    WHERE batch_id IS NOT NULL AND custom_id IS NOT NULL;
