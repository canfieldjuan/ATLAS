-- Track which processed emails have been synced to the knowledge graph
ALTER TABLE processed_emails
    ADD COLUMN IF NOT EXISTS graph_synced_at TIMESTAMPTZ;

-- Index for the email graph sync job to find unsynced emails efficiently
CREATE INDEX IF NOT EXISTS idx_processed_emails_graph_sync
    ON processed_emails (graph_synced_at)
    WHERE graph_synced_at IS NULL;
