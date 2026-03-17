-- Persist normalized reviewer-company names for stable joins and audits.
-- Migration 159
--
-- `reviewer_company` is user-facing/raw source text. We also need a
-- normalized form so joins to prospect/company tables stop drifting on case,
-- legal suffixes, punctuation, and spacing differences.

ALTER TABLE b2b_reviews
    ADD COLUMN IF NOT EXISTS reviewer_company_norm TEXT;

UPDATE b2b_reviews
SET reviewer_company_norm = NULLIF(
        BTRIM(
            regexp_replace(
                regexp_replace(
                    regexp_replace(
                        lower(COALESCE(reviewer_company, '')),
                        '\b(inc|incorporated|llc|ltd|limited|corp|corporation|co|company|plc|gmbh|ag|sa|srl|pty|nv|bv)\b\.?',
                        '',
                        'gi'
                    ),
                    '\s+',
                    ' ',
                    'g'
                ),
                '[,\.\-;:]+$',
                '',
                'g'
            )
        ),
        ''
    )
WHERE reviewer_company IS NOT NULL
  AND BTRIM(reviewer_company) <> ''
  AND (reviewer_company_norm IS NULL OR reviewer_company_norm = '');

CREATE INDEX IF NOT EXISTS idx_b2b_reviews_reviewer_company_norm
    ON b2b_reviews (reviewer_company_norm)
    WHERE reviewer_company_norm IS NOT NULL AND reviewer_company_norm <> '';
