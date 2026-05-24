-- Ticket FAQ search projection: one compact searchable row per generated FAQ
-- item. The generated Markdown document remains the canonical artifact in
-- ticket_faq_markdown; this table is a read-optimized projection for deflection
-- search routes and concurrent demo traffic.

CREATE TABLE IF NOT EXISTS ticket_faq_search_documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id TEXT NOT NULL DEFAULT '',
    corpus_id TEXT NOT NULL DEFAULT '',
    faq_id UUID NOT NULL REFERENCES ticket_faq_markdown(id) ON DELETE CASCADE,
    target_id TEXT NOT NULL DEFAULT '',
    target_mode TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'draft',
    rank INTEGER NOT NULL DEFAULT 0,
    topic TEXT NOT NULL DEFAULT '',
    question TEXT NOT NULL DEFAULT '',
    answer_summary TEXT NOT NULL DEFAULT '',
    source_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    ticket_count INTEGER NOT NULL DEFAULT 0,
    search_text TEXT NOT NULL DEFAULT '',
    search_vector TSVECTOR GENERATED ALWAYS AS (
        to_tsvector(
            'english',
            question || ' ' || topic || ' ' || answer_summary || ' ' || search_text
        )
    ) STORED,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT chk_ticket_faq_search_account_id CHECK (btrim(account_id) <> ''),
    CONSTRAINT chk_ticket_faq_search_corpus_id CHECK (btrim(corpus_id) <> ''),
    CONSTRAINT chk_ticket_faq_search_status CHECK (btrim(status) <> ''),
    CONSTRAINT chk_ticket_faq_search_rank CHECK (rank > 0),
    CONSTRAINT chk_ticket_faq_search_ticket_count CHECK (ticket_count >= 0),
    UNIQUE (account_id, corpus_id, faq_id, rank)
);

CREATE INDEX IF NOT EXISTS idx_ticket_faq_search_scope
    ON ticket_faq_search_documents (account_id, corpus_id, status, rank);

CREATE INDEX IF NOT EXISTS idx_ticket_faq_search_faq
    ON ticket_faq_search_documents (faq_id);

CREATE INDEX IF NOT EXISTS idx_ticket_faq_search_vector
    ON ticket_faq_search_documents USING GIN (search_vector);
