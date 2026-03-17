-- Backfill promoted review columns and add final Reddit admin analytics fields.
-- Migration 157
--
-- Migrations 152-156 introduced first-class columns for several values that
-- previously lived only inside JSON blobs. This migration:
--   1. backfills those columns for existing rows
--   2. adds the remaining Reddit/admin analytics fields still read from
--      raw_metadata (reddit_score, harvested comment thread count,
--      crosspost_subreddits)
--   3. adds the missing turning-point index for B2B sentiment analytics

ALTER TABLE b2b_reviews
    ADD COLUMN IF NOT EXISTS reddit_score INT,
    ADD COLUMN IF NOT EXISTS reddit_comment_thread_count INT,
    ADD COLUMN IF NOT EXISTS reddit_crosspost_subreddits JSONB;

-- b2b_reviews backfill -------------------------------------------------------

UPDATE b2b_reviews
SET source_weight = CASE
        WHEN raw_metadata->>'source_weight' IS NOT NULL
        THEN (raw_metadata->>'source_weight')::real
        ELSE source_weight
    END
WHERE source_weight IS NULL
  AND raw_metadata->>'source_weight' IS NOT NULL;

UPDATE b2b_reviews
SET reddit_subreddit = COALESCE(reddit_subreddit, raw_metadata->>'subreddit'),
    reddit_trending = COALESCE(reddit_trending, raw_metadata->>'trending_score'),
    reddit_flair = COALESCE(reddit_flair, raw_metadata->>'post_flair'),
    reddit_is_edited = COALESCE(reddit_is_edited, (raw_metadata->>'is_edited')::boolean),
    reddit_is_crosspost = COALESCE(reddit_is_crosspost, (raw_metadata->>'is_crosspost')::boolean),
    reddit_num_comments = COALESCE(
        reddit_num_comments,
        CASE
            WHEN (raw_metadata->>'num_comments') ~ '^\-?[0-9]+$'
            THEN (raw_metadata->>'num_comments')::int
            ELSE NULL
        END
    ),
    reddit_score = COALESCE(
        reddit_score,
        CASE
            WHEN (raw_metadata->>'score') ~ '^\-?[0-9]+$'
            THEN (raw_metadata->>'score')::int
            ELSE NULL
        END
    ),
    reddit_comment_thread_count = COALESCE(
        reddit_comment_thread_count,
        COALESCE(jsonb_array_length(raw_metadata->'comment_threads'), 0)
    ),
    reddit_crosspost_subreddits = COALESCE(
        reddit_crosspost_subreddits,
        CASE
            WHEN jsonb_typeof(raw_metadata->'crosspost_subreddits') = 'array'
            THEN raw_metadata->'crosspost_subreddits'
            ELSE NULL
        END
    )
WHERE source = 'reddit';

UPDATE b2b_reviews
SET sentiment_direction = COALESCE(sentiment_direction, enrichment->'sentiment_trajectory'->>'direction'),
    sentiment_tenure = COALESCE(sentiment_tenure, enrichment->'sentiment_trajectory'->>'tenure'),
    sentiment_turning_point = COALESCE(sentiment_turning_point, enrichment->'sentiment_trajectory'->>'turning_point')
WHERE enrichment IS NOT NULL
  AND enrichment != '{}'::jsonb
  AND (
      sentiment_direction IS NULL
      OR sentiment_tenure IS NULL
      OR sentiment_turning_point IS NULL
  );

-- product_reviews backfill ---------------------------------------------------

UPDATE product_reviews
SET would_repurchase = COALESCE(
        would_repurchase,
        CASE
            WHEN deep_extraction->>'would_repurchase' = 'true' THEN true
            WHEN deep_extraction->>'would_repurchase' = 'false' THEN false
            ELSE NULL
        END
    ),
    replacement_behavior = COALESCE(replacement_behavior, deep_extraction->>'replacement_behavior'),
    sentiment_trajectory = COALESCE(sentiment_trajectory, deep_extraction->>'sentiment_trajectory'),
    consequence_severity = COALESCE(consequence_severity, deep_extraction->>'consequence_severity')
WHERE deep_extraction IS NOT NULL
  AND deep_extraction != '{}'::jsonb
  AND (
      would_repurchase IS NULL
      OR replacement_behavior IS NULL
      OR sentiment_trajectory IS NULL
      OR consequence_severity IS NULL
  );

-- indexes -------------------------------------------------------------------

CREATE INDEX IF NOT EXISTS idx_b2b_reviews_sentiment_turning_point
    ON b2b_reviews (sentiment_turning_point)
    WHERE sentiment_turning_point IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_b2b_reviews_reddit_score
    ON b2b_reviews (reddit_score DESC)
    WHERE reddit_score IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_b2b_reviews_reddit_comment_threads
    ON b2b_reviews (reddit_comment_thread_count DESC)
    WHERE reddit_comment_thread_count IS NOT NULL AND reddit_comment_thread_count > 0;

CREATE INDEX IF NOT EXISTS idx_b2b_reviews_reddit_crosspost_subs
    ON b2b_reviews USING GIN (reddit_crosspost_subreddits)
    WHERE reddit_crosspost_subreddits IS NOT NULL;
