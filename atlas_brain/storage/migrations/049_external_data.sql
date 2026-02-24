-- 049: External data tables (news + financial markets)
--
-- data_watchlist: what Atlas monitors (stocks, commodities, news topics)
-- data_dedup: prevent re-emitting same article or price alert
-- market_snapshots: price history for context aggregation

-- Watchlist: what Atlas monitors
CREATE TABLE IF NOT EXISTS data_watchlist (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    category      TEXT NOT NULL,       -- stock | etf | commodity | crypto | forex | news_topic | news_region
    symbol        TEXT,                -- ticker (null for news topics)
    name          TEXT NOT NULL,       -- "Coffee Futures", "Natural disasters"
    keywords      TEXT[],              -- for news: {'coffee','brazil','frost'}
    threshold_pct FLOAT,              -- price move % to trigger (null for news)
    enabled       BOOLEAN NOT NULL DEFAULT true,
    metadata      JSONB NOT NULL DEFAULT '{}',
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_watchlist_enabled ON data_watchlist(category) WHERE enabled = true;

-- Dedup: prevent re-emitting same article or price alert
CREATE TABLE IF NOT EXISTS data_dedup (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source        TEXT NOT NULL,       -- 'news' or 'market'
    dedup_key     TEXT NOT NULL,       -- sha256(url) for news, "SYMBOL:date:direction" for market
    first_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_data_dedup_key ON data_dedup(source, dedup_key);

-- Market snapshots: price history for context aggregation
CREATE TABLE IF NOT EXISTS market_snapshots (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol      TEXT NOT NULL,
    price       NUMERIC(16,6) NOT NULL,
    change_pct  NUMERIC(8,4),
    volume      BIGINT,
    snapshot_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_snapshots_symbol ON market_snapshots(symbol, snapshot_at DESC);
