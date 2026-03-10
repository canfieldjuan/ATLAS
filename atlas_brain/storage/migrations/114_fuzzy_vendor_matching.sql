-- Phase 0/2 Sprint: Fuzzy vendor name matching
--
-- Enables pg_trgm extension for trigram-based similarity search on vendor names.
-- Adds GiST trigram index on canonical_name for fast fuzzy SQL queries.

CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Trigram index for fuzzy vendor search
CREATE INDEX IF NOT EXISTS idx_b2b_vendors_canonical_trgm
    ON b2b_vendors USING GIST (canonical_name gist_trgm_ops);

-- Also index company names for company-level identity resolution
CREATE INDEX IF NOT EXISTS idx_b2b_company_signals_company_trgm
    ON b2b_company_signals USING GIST (company_name gist_trgm_ops);
