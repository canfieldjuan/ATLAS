# Reddit Scraper Enhancement: Search Profiles + Comment Harvesting + Insider Signals

## Overview

Upgrade the Reddit parser from a simple vendor-name search to a profile-driven system with comment harvesting and insider signal extraction. Three search profiles (`churn`, `deep`, `insider`) control query templates, comment depth, and subreddit targeting.

## Step 1: Migration `133_reddit_insider_signals.sql`

Add columns to `b2b_reviews` for content typing and comment threading:

```sql
ALTER TABLE b2b_reviews ADD COLUMN IF NOT EXISTS content_type TEXT NOT NULL DEFAULT 'review';
ALTER TABLE b2b_reviews ADD COLUMN IF NOT EXISTS parent_review_id UUID REFERENCES b2b_reviews(id);
ALTER TABLE b2b_reviews ADD COLUMN IF NOT EXISTS thread_id TEXT;
ALTER TABLE b2b_reviews ADD COLUMN IF NOT EXISTS comment_depth SMALLINT NOT NULL DEFAULT 0;

CREATE INDEX IF NOT EXISTS idx_b2b_reviews_thread ON b2b_reviews(thread_id) WHERE thread_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_b2b_reviews_content_type ON b2b_reviews(content_type) WHERE content_type != 'review';
CREATE INDEX IF NOT EXISTS idx_b2b_reviews_parent ON b2b_reviews(parent_review_id) WHERE parent_review_id IS NOT NULL;
```

Add insider aggregate columns to `b2b_churn_signals`:

```sql
ALTER TABLE b2b_churn_signals ADD COLUMN IF NOT EXISTS insider_signal_count INT NOT NULL DEFAULT 0;
ALTER TABLE b2b_churn_signals ADD COLUMN IF NOT EXISTS insider_org_health_summary JSONB NOT NULL DEFAULT '{}';
ALTER TABLE b2b_churn_signals ADD COLUMN IF NOT EXISTS insider_talent_drain_rate NUMERIC(5,4);
ALTER TABLE b2b_churn_signals ADD COLUMN IF NOT EXISTS insider_quotable_evidence JSONB NOT NULL DEFAULT '[]';
```

## Step 2: Reddit Parser — Search Profiles + Comment Harvesting

**File:** `atlas_brain/services/scraping/parsers/reddit.py`

### Search Profiles (read from `target.metadata.get("search_profile", "churn")`)

| Profile | Query Templates | Subreddits | Comments | Use Case |
|---------|----------------|------------|----------|----------|
| `churn` (default) | vendor name + churn qualifiers (current behavior, improved templates) | target.metadata subreddits or defaults | No | Backward compatible |
| `deep` | vendor + churn qualifiers + pain/frustration queries + comparison queries | Expanded: adds vendor-specific subs | Top 10 comments per post (score >= 3) | Full signal extraction |
| `insider` | Employee/org queries ("working at {vendor}", "left {vendor}", "{vendor} culture") | r/cscareerquestions, r/ExperiencedDevs, r/sysadmin, Blind-style subs | Top 15 comments | Insider/talent drain signals |

### Comment Harvesting

For `deep` and `insider` profiles, after collecting posts:
1. For each post with `num_comments >= 3`, fetch `https://oauth.reddit.com/comments/{post_id}?sort=best&limit=N`
2. Parse comments into review dicts with:
   - `content_type`: `"comment"`
   - `parent_review_id`: will be resolved after parent post insert (store post's `source_review_id` in `raw_metadata.parent_source_review_id`)
   - `thread_id`: Reddit post fullname (e.g. `t3_abc123`)
   - `comment_depth`: actual depth from Reddit API
   - `source_review_id`: comment ID (e.g. `t1_xyz789`)
3. Filter: skip comments < 80 chars, skip `[deleted]`/`[removed]`, skip comments with score < 2

### Insider Query Templates

```python
_INSIDER_QUERIES = [
    '"{vendor}" culture toxic',
    '"{vendor}" leaving why',
    '"worked at {vendor}"',
    '"left {vendor}"',
    '"{vendor}" layoffs morale',
    '"{vendor}" product quality declining',
]

_INSIDER_SUBREDDITS = [
    "cscareerquestions", "ExperiencedDevs", "ITCareerQuestions",
    "sysadmin", "devops", "antiwork", "jobs",
]
```

### Content Type Assignment

- Posts from `churn`/`deep` profiles → `content_type = "review"` (backward compat) or `"community_discussion"`
- Posts from `insider` profile → `content_type = "insider_account"`
- Comments from any profile → `content_type = "comment"`

### Version Bump

`version = "reddit:2"` — triggers automatic re-processing of old `reddit:1` reviews.

## Step 3: Update `_INSERT_SQL` in `b2b_scrape_intake.py`

Add new columns to INSERT query: `content_type`, `parent_review_id`, `thread_id`, `comment_depth`.

For comments with `raw_metadata.parent_source_review_id`, resolve to actual UUID after parent insert (lookup by `source_review_id` + `source='reddit'`).

## Step 4: Relevance Scorer — Flip Insider Patterns

**File:** `atlas_brain/services/scraping/relevance.py`

Current noise patterns penalize insider signals:
- `laid off|layoffs|headcount` → -0.10
- `CEO|CFO|CTO|executive|leadership churn` → -0.10
- `revenue|billion|workforce` → -0.10

Fix: Make these penalties conditional on `content_type`. When `content_type == "insider_account"`:
- Flip HR/layoff patterns from -0.10 to +0.10 (talent drain signal)
- Flip executive/leadership patterns from -0.10 to +0.05 (org health signal)
- Add new insider boost patterns: "culture", "morale", "toxic", "micromanagement", "bureaucracy", "dead end"

Implementation: Add `content_type` parameter to `score_relevance()` (default `"review"` for backward compat).

## Step 5: Enrichment Triage Skill Update

**File:** `atlas_brain/skills/digest/b2b_churn_triage.md`

Add insider signal criteria to the YES list:
- Employee describes organizational dysfunction, talent drain, or culture problems
- Post discusses product quality decline from insider perspective
- Content type is `insider_account` (always YES — we specifically searched for this)

## Step 6: Enrichment Extraction Skill Update

**File:** `atlas_brain/skills/digest/b2b_churn_extraction.md`

Add insider-specific fields to the extraction schema (inside the existing JSONB — no schema change):

```json
{
  "content_classification": "insider_account | community_discussion | review",
  "insider_signals": {
    "role_at_company": "string or null",
    "departure_type": "voluntary | involuntary | still_employed | unknown",
    "org_health": {
      "bureaucracy_level": "high | medium | low | unknown",
      "leadership_quality": "poor | mixed | good | unknown",
      "innovation_climate": "stagnant | declining | healthy | unknown",
      "culture_indicators": ["list of keywords"]
    },
    "talent_drain": {
      "departures_mentioned": "bool",
      "layoff_fear": "bool",
      "morale": "high | medium | low | unknown"
    }
  }
}
```

Add instruction: "When content_type is insider_account, prioritize insider_signals extraction. The existing churn_signals and urgency_score still apply — insider accounts revealing product stagnation or talent exodus ARE churn signals for customers."

## Step 7: MCP Server — Add `content_type` Filter to `search_reviews`

**File:** `atlas_brain/mcp/b2b_churn_server.py`

Add optional `content_type` parameter to `search_reviews()`:
```python
content_type: Optional[str] = None,  # 'review', 'insider_account', 'community_discussion', 'comment'
```

Add `thread_id` to the SELECT output so MCP users can pull full threads.

## Step 8: Churn Intelligence Aggregation

**File:** `atlas_brain/autonomous/tasks/b2b_churn_intelligence.py`

In the weekly aggregation query, include insider content:
- Count `insider_signal_count` from `content_type = 'insider_account'`
- Aggregate `insider_org_health_summary` from enrichment JSONB
- Compute `insider_talent_drain_rate` (insider posts mentioning departures / total insider posts)
- Collect `insider_quotable_evidence` (top quotes from insider posts by urgency)
- Upsert these into `b2b_churn_signals`

## File Change Summary

| File | Change |
|------|--------|
| `atlas_brain/storage/migrations/133_reddit_insider_signals.sql` | **NEW** — schema additions |
| `atlas_brain/services/scraping/parsers/reddit.py` | Search profiles, comment harvesting, insider queries, version bump |
| `atlas_brain/autonomous/tasks/b2b_scrape_intake.py` | Update INSERT SQL + parent_review_id resolution |
| `atlas_brain/services/scraping/relevance.py` | Content-type-aware scoring, insider boosts |
| `atlas_brain/skills/digest/b2b_churn_triage.md` | Insider signal criteria |
| `atlas_brain/skills/digest/b2b_churn_extraction.md` | Insider fields in extraction schema |
| `atlas_brain/mcp/b2b_churn_server.py` | `content_type` filter + `thread_id` in search_reviews |
| `atlas_brain/autonomous/tasks/b2b_churn_intelligence.py` | Insider aggregate computation |
