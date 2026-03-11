"""
Pipeline registrations for news/market intelligence and complaint mining.

Importing this module registers both pipelines with the pipeline registry.
New pipelines can be added here without editing __init__.py, scheduler.py,
or runner.py.
"""

from atlas_brain.pipelines import (
    CleanupRule,
    PipelineConfig,
    TaskDef,
    register_pipeline,
)

# ------------------------------------------------------------------
# News / market intelligence pipeline
# ------------------------------------------------------------------

register_pipeline(PipelineConfig(
    name="news",
    enabled_key="external_data.enabled",
    tasks=[
        TaskDef(
            name="news_intake",
            module="news_intake",
            schedule_type="interval",
            interval_seconds=None,
            timeout_seconds=120,
            description="Poll news feeds, match watchlist keywords, store for daily intelligence",
            metadata={"builtin_handler": "news_intake"},
            interval_config_key="external_data.news_interval_seconds",
        ),
        TaskDef(
            name="market_intake",
            module="market_intake",
            schedule_type="interval",
            interval_seconds=None,
            timeout_seconds=60,
            description="Poll market prices for watchlist symbols, store snapshots for daily intelligence",
            metadata={"builtin_handler": "market_intake"},
            interval_config_key="external_data.market_interval_seconds",
        ),
        TaskDef(
            name="article_enrichment",
            module="article_enrichment",
            schedule_type="interval",
            interval_seconds=None,
            timeout_seconds=180,
            description="Fetch article content and classify via SORAM pressure channels",
            metadata={"builtin_handler": "article_enrichment"},
            interval_config_key="external_data.enrichment_interval_seconds",
        ),
        TaskDef(
            name="daily_intelligence",
            module="daily_intelligence",
            schedule_type="cron",
            cron_expression="0 20 * * *",
            timeout_seconds=300,
            description="Daily deep analysis of accumulated market, news, and business data",
            metadata={
                "builtin_handler": "daily_intelligence",
                "notify_priority": "default",
                "notify_tags": "brain,chart_with_upwards_trend",
            },
            cron_config_key="external_data.intelligence_cron",
        ),
    ],
    cleanup_rules=[
        CleanupRule(
            table="data_dedup",
            where_clause="DELETE FROM data_dedup WHERE first_seen_at < CURRENT_TIMESTAMP - make_interval(days => $1)",
            retention_config_key="external_data.retention_days",
            result_key="data_dedup_cleaned",
        ),
        CleanupRule(
            table="market_snapshots",
            where_clause="DELETE FROM market_snapshots WHERE snapshot_at < CURRENT_TIMESTAMP - make_interval(days => $1)",
            retention_config_key="external_data.retention_days",
            result_key="market_snapshots_cleaned",
        ),
        CleanupRule(
            table="news_articles",
            where_clause="DELETE FROM news_articles WHERE created_at < CURRENT_TIMESTAMP - make_interval(days => $1)",
            retention_config_key="external_data.intelligence_news_retention_days",
            result_key="news_articles_cleaned",
        ),
        CleanupRule(
            table="reasoning_journal",
            where_clause="DELETE FROM reasoning_journal WHERE created_at < CURRENT_TIMESTAMP - make_interval(days => $1)",
            retention_config_key="external_data.intelligence_journal_retention_days",
            result_key="reasoning_journal_cleaned",
        ),
        CleanupRule(
            table="entity_pressure_baselines",
            where_clause="DELETE FROM entity_pressure_baselines WHERE last_computed_at < CURRENT_TIMESTAMP - make_interval(days => $1)",
            retention_config_key="external_data.intelligence_journal_retention_days",
            result_key="pressure_baselines_cleaned",
        ),
    ],
))

# ------------------------------------------------------------------
# Complaint mining pipeline
# ------------------------------------------------------------------

register_pipeline(PipelineConfig(
    name="complaint",
    enabled_key="external_data.complaint_mining_enabled",
    tasks=[
        TaskDef(
            name="complaint_enrichment",
            module="complaint_enrichment",
            schedule_type="interval",
            interval_seconds=None,
            timeout_seconds=300,
            description="Classify product reviews via LLM for root cause, severity, and pain score",
            metadata={"builtin_handler": "complaint_enrichment"},
            interval_config_key="external_data.complaint_enrichment_interval_seconds",
        ),
        TaskDef(
            name="deep_enrichment",
            module="deep_enrichment",
            schedule_type="interval",
            interval_seconds=None,
            timeout_seconds=300,
            description="Deep extraction of 32 structured fields per review (product analysis, buyer psychology, extended context)",
            metadata={"builtin_handler": "deep_enrichment"},
            interval_config_key="external_data.deep_enrichment_interval_seconds",
        ),
        TaskDef(
            name="complaint_analysis",
            module="complaint_analysis",
            schedule_type="cron",
            cron_expression="0 21 * * *",
            timeout_seconds=300,
            description="Daily aggregation of product complaint pain points and opportunities",
            metadata={
                "builtin_handler": "complaint_analysis",
                "notify_priority": "default",
                "notify_tags": "brain,shopping_cart",
            },
            cron_config_key="external_data.complaint_analysis_cron",
        ),
        TaskDef(
            name="complaint_content_generation",
            module="complaint_content_generation",
            schedule_type="cron",
            cron_expression="0 22 * * *",
            timeout_seconds=600,
            description="Generate sellable content (forum posts, articles, email copy) from top pain points via Claude",
            metadata={
                "builtin_handler": "complaint_content_generation",
                "notify_priority": "default",
                "notify_tags": "brain,memo",
            },
            cron_config_key="external_data.complaint_content_cron",
        ),
        TaskDef(
            name="competitive_intelligence",
            module="competitive_intelligence",
            schedule_type="cron",
            cron_expression="30 21 * * *",
            timeout_seconds=300,
            description="Cross-brand competitive intelligence from deep-extracted product reviews",
            metadata={
                "builtin_handler": "competitive_intelligence",
                "notify_priority": "default",
                "notify_tags": "brain,bar_chart",
            },
            cron_config_key="external_data.competitive_intelligence_cron",
        ),
        TaskDef(
            name="consumer_analytics_refresh",
            module="consumer_analytics_refresh",
            schedule_type="interval",
            interval_seconds=21600,  # 6 hours
            timeout_seconds=120,
            description="Refresh consumer analytics materialized views (brand, category, ASIN summaries)",
            metadata={"builtin_handler": "consumer_analytics_refresh"},
        ),
        TaskDef(
            name="blog_post_generation",
            module="blog_post_generation",
            schedule_type="cron",
            cron_expression="0 23 * * 0",
            timeout_seconds=600,
            description="Weekly data-backed blog post generation with interactive charts",
            metadata={
                "builtin_handler": "blog_post_generation",
                "notify_priority": "default",
                "notify_tags": "brain,newspaper",
            },
            cron_config_key="external_data.blog_post_cron",
        ),
    ],
    cleanup_rules=[
        CleanupRule(
            table="complaint_reports",
            where_clause="DELETE FROM complaint_reports WHERE created_at < CURRENT_TIMESTAMP - make_interval(days => $1)",
            retention_config_key="external_data.complaint_retention_days",
            result_key="complaint_reports_cleaned",
        ),
        CleanupRule(
            table="product_reviews",
            where_clause="DELETE FROM product_reviews WHERE enrichment_status = 'failed' AND imported_at < CURRENT_TIMESTAMP - make_interval(days => $1)",
            retention_config_key="external_data.complaint_retention_days",
            result_key="failed_reviews_cleaned",
        ),
        CleanupRule(
            table="complaint_content",
            where_clause="DELETE FROM complaint_content WHERE created_at < CURRENT_TIMESTAMP - make_interval(days => $1)",
            retention_config_key="external_data.complaint_retention_days",
            result_key="complaint_content_cleaned",
        ),
    ],
))

# ------------------------------------------------------------------
# B2B churn prediction pipeline
# ------------------------------------------------------------------

register_pipeline(PipelineConfig(
    name="b2b_churn",
    enabled_key="b2b_churn.enabled",
    tasks=[
        TaskDef(
            name="b2b_enrichment",
            module="b2b_enrichment",
            schedule_type="interval",
            interval_seconds=None,
            timeout_seconds=180,
            description="Enrich B2B reviews with churn signals via LLM",
            metadata={"builtin_handler": "b2b_enrichment"},
            interval_config_key="b2b_churn.enrichment_interval_seconds",
        ),
        TaskDef(
            name="b2b_churn_intelligence",
            module="b2b_churn_intelligence",
            schedule_type="cron",
            cron_expression="0 21 * * *",
            timeout_seconds=600,
            description="Daily churn intelligence aggregation and feed generation",
            metadata={
                "builtin_handler": "b2b_churn_intelligence",
                "notify_priority": "high",
                "notify_tags": "brain,chart_with_downwards_trend",
            },
            cron_config_key="b2b_churn.intelligence_cron",
        ),
        TaskDef(
            name="b2b_keyword_signal",
            module="b2b_keyword_signal",
            schedule_type="cron",
            cron_expression="0 6 * * 1",
            timeout_seconds=600,
            description="Weekly Google Trends keyword signal collection",
            metadata={"builtin_handler": "b2b_keyword_signal"},
            cron_config_key="b2b_churn.keyword_signal_cron",
        ),
        TaskDef(
            name="b2b_product_profiles",
            module="b2b_product_profiles",
            schedule_type="cron",
            cron_expression="30 21 * * *",
            timeout_seconds=600,
            description="Generate/refresh product profile knowledge cards from enriched reviews",
            metadata={"builtin_handler": "b2b_product_profiles"},
            cron_config_key="b2b_churn.product_profile_cron",
        ),
    ],
    cleanup_rules=[
        CleanupRule(
            table="b2b_intelligence",
            where_clause="DELETE FROM b2b_intelligence WHERE created_at < CURRENT_TIMESTAMP - make_interval(days => $1)",
            retention_config_key="b2b_churn.intelligence_window_days",
            result_key="b2b_intelligence_cleaned",
        ),
        CleanupRule(
            table="b2b_campaigns",
            where_clause="DELETE FROM b2b_campaigns WHERE status IN ('expired', 'sent') AND created_at < CURRENT_TIMESTAMP - make_interval(days => $1)",
            retention_config_key="b2b_campaign.retention_days",
            result_key="b2b_campaigns_cleaned",
        ),
        CleanupRule(
            table="b2b_keyword_signals",
            where_clause="DELETE FROM b2b_keyword_signals WHERE snapshot_at < CURRENT_TIMESTAMP - make_interval(days => $1)",
            retention_config_key="b2b_churn.keyword_retention_days",
            result_key="b2b_keyword_signals_cleaned",
        ),
    ],
))

# ------------------------------------------------------------------
# B2B review scraping pipeline
# ------------------------------------------------------------------

register_pipeline(PipelineConfig(
    name="b2b_scrape",
    enabled_key="b2b_scrape.enabled",
    tasks=[
        TaskDef(
            name="b2b_scrape_intake",
            module="b2b_scrape_intake",
            schedule_type="interval",
            interval_seconds=None,
            timeout_seconds=1800,
            description="Scrape B2B review sites per configured targets",
            metadata={"builtin_handler": "b2b_scrape_intake"},
            interval_config_key="b2b_scrape.intake_interval_seconds",
        ),
    ],
    cleanup_rules=[
        CleanupRule(
            table="b2b_scrape_log",
            where_clause="DELETE FROM b2b_scrape_log WHERE started_at < CURRENT_TIMESTAMP - make_interval(days => $1)",
            retention_config_key="b2b_scrape.scrape_log_retention_days",
            result_key="b2b_scrape_log_cleaned",
        ),
    ],
))

# ------------------------------------------------------------------
# B2B prospect enrichment pipeline (Apollo.io)
# ------------------------------------------------------------------

register_pipeline(PipelineConfig(
    name="b2b_prospect",
    enabled_key="apollo.enabled",
    tasks=[
        TaskDef(
            name="vendor_target_enrichment",
            module="vendor_target_enrichment",
            schedule_type="cron",
            timeout_seconds=900,
            description="Enrich vendor/challenger targets with Apollo.io contacts",
            metadata={"builtin_handler": "vendor_target_enrichment"},
            cron_config_key="apollo.vendor_enrichment_cron",
        ),
        TaskDef(
            name="prospect_enrichment",
            module="prospect_enrichment",
            schedule_type="cron",
            timeout_seconds=1800,
            description="Enrich companies with Apollo.io prospects (org + people + email)",
            metadata={"builtin_handler": "prospect_enrichment"},
            cron_config_key="apollo.enrichment_cron",
        ),
        TaskDef(
            name="prospect_matching",
            module="prospect_matching",
            schedule_type="interval",
            timeout_seconds=120,
            description="Match enriched prospects to unmatched campaign sequences",
            metadata={"builtin_handler": "prospect_matching"},
            interval_config_key="apollo.matching_interval_seconds",
        ),
    ],
    cleanup_rules=[
        CleanupRule(
            table="prospect_org_cache",
            where_clause="DELETE FROM prospect_org_cache WHERE status = 'not_found' AND enriched_at < CURRENT_TIMESTAMP - make_interval(days => $1)",
            retention_config_key="apollo.org_cache_days",
            result_key="prospect_org_not_found_cleaned",
        ),
        CleanupRule(
            table="prospects",
            where_clause="DELETE FROM prospects WHERE status IN ('bounced', 'suppressed') AND updated_at < CURRENT_TIMESTAMP - make_interval(days => $1)",
            retention_config_key="apollo.org_cache_days",
            result_key="bounced_prospects_cleaned",
        ),
    ],
))
