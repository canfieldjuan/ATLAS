from atlas_brain.services.scraping.parsers.quora import (
    _build_serp_query_suffixes,
    _classify_quora_url,
    _extract_answer_url,
    _extract_question_urls_from_search,
    _is_historical_vendor_question_url,
    _is_scrapable_quora_url,
    _parse_question_page,
    _question_root_url,
    _select_serp_question_urls,
    _serp_discovery_cutoff,
    _should_try_direct_question,
    _should_try_http_fallback,
    _url_matches_vendor,
)
from atlas_brain.services.scraping.parsers import ScrapeTarget


class _Target:
    vendor_name = "Zendesk"
    product_name = "Zendesk"
    product_category = "Helpdesk"


def test_classify_quora_url_filters_junk_pages():
    assert _classify_quora_url("https://www.quora.com/search?q=Zendesk") == "search_page"
    assert _classify_quora_url("https://www.quora.com/profile/Jane-Doe") == "profile_page"
    assert _classify_quora_url("https://www.quora.com/topic/Software-and-Applications") == "topic_page"
    assert _classify_quora_url("https://www.quora.com/unanswered/What-is-the-best-Zendesk-alternative") == "unanswered_page"
    assert _classify_quora_url("https://www.quora.com/qemail/track_click?id=123") == "tracking_page"
    assert _classify_quora_url("https://emailmarketingsspace.quora.com/Sendinblue-vs-GetResponse") == "space_page"
    assert _classify_quora_url("https://www.quora.com/What-is-the-best-alternative-to-Zendesk") == "question_page"
    assert _classify_quora_url("https://www.quora.com/What-is-the-best-alternative-to-Zendesk/answer/Alice") == "answer_page"
    assert _classify_quora_url("https://www.quora.com/What-is-the-best-alternative-to-Zendesk/answers/123") == "answer_page"


def test_is_scrapable_quora_url_only_allows_question_and_answer_pages():
    assert _is_scrapable_quora_url("https://www.quora.com/What-is-the-best-alternative-to-Zendesk")
    assert _is_scrapable_quora_url("https://www.quora.com/What-is-the-best-alternative-to-Zendesk/answer/Alice")
    assert _is_scrapable_quora_url("https://www.quora.com/What-is-the-best-alternative-to-Zendesk/answers/123")
    assert not _is_scrapable_quora_url("https://www.quora.com/search?q=Zendesk")
    assert not _is_scrapable_quora_url("https://www.quora.com/profile/Jane-Doe")
    assert not _is_scrapable_quora_url("https://www.quora.com/topic/Software-and-Applications")
    assert not _is_scrapable_quora_url("https://www.quora.com/unanswered/What-is-the-best-Zendesk-alternative")
    assert not _is_scrapable_quora_url("https://www.quora.com/qemail/track_click?id=123")


def test_extract_answer_url_rejects_profile_and_topic_links():
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(
        """
        <div>
          <a href="/profile/Jane-Doe">Jane</a>
          <a href="/topic/Software">Software</a>
          <a href="/What-is-the-best-alternative-to-Zendesk">Question</a>
        </div>
        """,
        "html.parser",
    )
    assert _extract_answer_url(soup) == "https://www.quora.com/What-is-the-best-alternative-to-Zendesk"


def test_extract_question_urls_from_search_filters_junk_links():
    html = """
    <html><body>
      <a href="/search?q=Zendesk">search</a>
      <a href="/profile/Jane-Doe">profile</a>
      <a href="/topic/Software">topic</a>
      <a href="/What-is-the-best-alternative-to-Zendesk">question</a>
      <a href="/What-is-the-best-alternative-to-Zendesk/answer/Alice">answer</a>
      <a href="/What-is-the-best-alternative-to-Zendesk/answers/123">answer2</a>
      <a href="/qemail/track_click?id=123">tracking</a>
      <a href="https://emailmarketingsspace.quora.com/Sendinblue-vs-GetResponse">space</a>
    </body></html>
    """
    assert _extract_question_urls_from_search(html) == [
        "https://www.quora.com/What-is-the-best-alternative-to-Zendesk",
        "https://www.quora.com/What-is-the-best-alternative-to-Zendesk/answer/Alice",
        "https://www.quora.com/What-is-the-best-alternative-to-Zendesk/answers/123",
    ]


def test_parse_question_page_rejects_non_question_url_even_if_html_is_long():
    html = "<html><body><div class='q-box'><span>" + ("x" * 200) + "</span></div></body></html>"
    reviews = _parse_question_page(
        html,
        _Target(),
        set(),
        page_url="https://www.quora.com/profile/Jane-Doe",
    )
    assert reviews == []


def test_parse_question_page_extracts_answer_timestamp_when_present():
    html = """
    <html><body>
      <h1><span>What is the best alternative to Zendesk?</span></h1>
      <div class="q-box">
        <time datetime="2026-03-20T10:30:00Z">Mar 20, 2026</time>
        <span>{}</span>
      </div>
    </body></html>
    """.format("Useful answer text " * 12)
    reviews = _parse_question_page(
        html,
        _Target(),
        set(),
        page_url="https://www.quora.com/What-is-the-best-alternative-to-Zendesk",
    )
    assert len(reviews) == 1
    assert reviews[0]["reviewed_at"] == "2026-03-20T10:30:00Z"


def test_parse_question_page_prefers_real_answer_cards_and_skips_promoted_and_related_blocks():
    html = """
    <html><body>
      <div class="q-text puppeteer_test_question_title">
        <span>What is a cheaper alternative to Intercom for startups?</span>
      </div>
      <div class="q-box dom_annotate_question_answer_item_0 qu-borderAll">
        <div class="q-box spacing_log_answer_header">
          <span class="q-text qu-dynamicFontSize--small qu-bold">Guijin Ding</span>
        </div>
        <div class="q-box spacing_log_answer_content puppeteer_test_answer_content">
          PPMessage is my open source customer communication project, an Intercom alternative
          you can try. Definitely cheaper for startups that need core messaging features without
          the Intercom price tag.
        </div>
      </div>
      <div class="q-box dom_annotate_ad_promoted_answer">
        <div class="q-box spacing_log_answer_content puppeteer_test_answer_content">
          Sponsored by Liquid Web Managed VPS Hosting Made Easy. Shop Now.
        </div>
      </div>
      <div class="q-box">
        <span>
          What is a cheaper alternative to Intercom for startups? What's the best alternative to
          Intercom? Is there a cheap alternative to Intercom? What are some alternatives to
          Intercom's live chat?
        </span>
      </div>
    </body></html>
    """
    target = ScrapeTarget(
        id="q1",
        source="quora",
        vendor_name="Intercom",
        product_name="Intercom",
        product_slug="intercom",
        product_category="Helpdesk",
        max_pages=5,
        metadata={},
    )
    reviews = _parse_question_page(
        html,
        target,
        set(),
        page_url="https://www.quora.com/What-is-a-cheaper-alternative-to-Intercom-for-startups",
    )
    assert len(reviews) == 1
    assert reviews[0]["summary"] == "What is a cheaper alternative to Intercom for startups?"
    assert "Sponsored by" not in reviews[0]["review_text"]
    assert reviews[0]["reviewer_name"] == "Guijin Ding"
    assert "PPMessage is my open source customer communication project" in reviews[0]["review_text"]


def test_parse_question_page_prefers_question_title_over_author_q_text():
    html = """
    <html><body>
      <div class="q-text puppeteer_test_question_title">
        <span>What are some alternatives to HubSpot?</span>
      </div>
      <div class="q-box dom_annotate_question_answer_item_0 qu-borderAll">
        <div class="q-box spacing_log_answer_header">
          <span class="q-text qu-dynamicFontSize--small qu-bold">Alec Blunden</span>
        </div>
        <div class="q-box spacing_log_answer_content puppeteer_test_answer_content">
          We switched from HubSpot to a lighter CRM because of pricing and complexity. Pipedrive
          and Freshsales were the two main options we evaluated.
        </div>
      </div>
    </body></html>
    """
    target = ScrapeTarget(
        id="q1",
        source="quora",
        vendor_name="HubSpot",
        product_name="HubSpot",
        product_slug="hubspot",
        product_category="CRM",
        max_pages=5,
        metadata={},
    )
    reviews = _parse_question_page(
        html,
        target,
        set(),
        page_url="https://www.quora.com/What-are-some-alternatives-to-HubSpot",
    )
    assert len(reviews) == 1
    assert reviews[0]["summary"] == "What are some alternatives to HubSpot?"


def test_serp_suffixes_add_after_filter_for_incremental_quora_scrapes():
    target = ScrapeTarget(
        id="q1",
        source="quora",
        vendor_name="Zendesk",
        product_name="Zendesk",
        product_slug="zendesk",
        product_category="Helpdesk",
        max_pages=5,
        metadata={},
        date_cutoff="2026-03-01",
    )
    suffixes = _build_serp_query_suffixes(target)
    assert suffixes
    assert all("after:" in suffix for suffix in suffixes)
    assert all("2026-03-01" not in suffix for suffix in suffixes)
    assert _serp_discovery_cutoff(target) is not None


def test_incremental_quora_scrapes_skip_direct_evergreen_question_fallback():
    target = ScrapeTarget(
        id="q1",
        source="quora",
        vendor_name="Zendesk",
        product_name="Zendesk",
        product_slug="zendesk",
        product_category="Helpdesk",
        max_pages=5,
        metadata={},
        date_cutoff="2026-03-01",
    )
    assert not _should_try_direct_question(target)
    assert not _should_try_http_fallback(target)


def test_quora_discovered_urls_must_reference_vendor_or_product():
    target = ScrapeTarget(
        id="q1",
        source="quora",
        vendor_name="Intercom",
        product_name="Intercom",
        product_slug="Intercom",
        product_category="Helpdesk",
        max_pages=5,
        metadata={},
        date_cutoff="2026-03-01",
    )
    assert _url_matches_vendor(
        "https://www.quora.com/What-is-the-best-Intercom-alternative",
        target,
    )
    assert not _url_matches_vendor(
        "https://www.quora.com/What-tools-help-automate-marketing-workflows",
        target,
    )


def test_quora_url_matching_ignores_generic_product_words():
    target = ScrapeTarget(
        id="q1",
        source="quora",
        vendor_name="HubSpot",
        product_name="HubSpot Marketing Hub",
        product_slug="HubSpot Marketing Hub",
        product_category="Marketing Automation",
        max_pages=5,
        metadata={},
        date_cutoff="2026-03-01",
    )
    assert not _url_matches_vendor(
        "https://www.quora.com/What-are-the-benefits-of-learning-SEO-and-social-media-marketing-together",
        target,
    )
    assert not _url_matches_vendor(
        "https://www.quora.com/Can-someone-start-digital-marketing-with-zero-experience-and-no-money",
        target,
    )
    assert _url_matches_vendor(
        "https://www.quora.com/What-are-some-alternatives-to-HubSpot",
        target,
    )
    assert _url_matches_vendor(
        "https://www.quora.com/Can-you-suggest-some-alternatives-to-HubSpots-CRM-and-marketing-tools",
        target,
    )


def test_historical_quora_refresh_requires_vendor_match_and_intent():
    zendesk = ScrapeTarget(
        id="q1",
        source="quora",
        vendor_name="Zendesk",
        product_name="Zendesk",
        product_slug="zendesk",
        product_category="Helpdesk",
        max_pages=5,
        metadata={},
        date_cutoff="2026-03-01",
    )
    assert not _is_historical_vendor_question_url(
        "https://www.quora.com/Whats-your-take-on-Zendesk-not-disclosing-the-terms-of-its-acquisition-of-Forethought",
        "",
        zendesk,
    )
    assert _is_historical_vendor_question_url(
        "https://www.quora.com/How-do-I-migrate-from-zendesk-to-Freshdesk",
        "",
        zendesk,
    )
    assert _is_historical_vendor_question_url(
        "https://www.quora.com/What-are-some-alternatives-to-Zendesk",
        "",
        zendesk,
    )


def test_quora_serp_selection_uses_snippet_relevance_not_just_url_tokens():
    target = ScrapeTarget(
        id="q1",
        source="quora",
        vendor_name="Intercom",
        product_name="Intercom",
        product_slug="intercom",
        product_category="Helpdesk",
        max_pages=5,
        metadata={},
        date_cutoff="2026-03-01",
    )
    selected = _select_serp_question_urls(
        [
            {
                "url": "https://www.quora.com/What-tools-help-scale-marketing-automation",
                "snippet": "ActiveCampaign · Intercom Marketing Automation · Drip.",
            },
            {
                "url": "https://www.quora.com/What-are-some-polite-ways-to-handle-a-seat-switch-request-on-an-airplane-especially-if-you-re-not-feeling-well",
                "snippet": "How can I deal with people asking to switch seats with me on a plane?",
            },
            {
                "url": "https://www.quora.com/What-is-a-cheaper-alternative-to-Intercom-for-startups",
                "snippet": "What is a cheaper alternative to Intercom for startups?",
            },
            {
                "url": "https://www.quora.com/Who-are-the-competitors-to-Intercom-io",
                "snippet": "Who are the competitors to Intercom.io?",
            },
        ],
        target,
    )
    assert "https://www.quora.com/What-tools-help-scale-marketing-automation" not in selected
    assert "https://www.quora.com/What-are-some-polite-ways-to-handle-a-seat-switch-request-on-an-airplane-especially-if-you-re-not-feeling-well" not in selected
    assert "https://www.quora.com/What-is-a-cheaper-alternative-to-Intercom-for-startups" in selected
    assert "https://www.quora.com/Who-are-the-competitors-to-Intercom-io" in selected


def test_incremental_quora_serp_selection_requires_vendor_in_url():
    target = ScrapeTarget(
        id="q1",
        source="quora",
        vendor_name="Intercom",
        product_name="Intercom",
        product_slug="intercom",
        product_category="Helpdesk",
        max_pages=5,
        metadata={},
        date_cutoff="2026-03-01",
    )
    selected = _select_serp_question_urls(
        [
            {
                "url": "https://www.quora.com/Why-are-automation-tools-becoming-SaaS-platforms",
                "snippet": "Typeform ... Intercom.io. Talk to your website's visitors in real-time. Great alternative ...",
            },
            {
                "url": "https://www.quora.com/What-is-a-cheaper-alternative-to-Intercom-for-startups",
                "snippet": "What is a cheaper alternative to Intercom for startups?",
            },
        ],
        target,
    )
    assert selected == [
        "https://www.quora.com/What-is-a-cheaper-alternative-to-Intercom-for-startups",
    ]


def test_incremental_quora_serp_selection_requires_intent_in_url_not_just_snippet():
    target = ScrapeTarget(
        id="q1",
        source="quora",
        vendor_name="Zendesk",
        product_name="Zendesk",
        product_slug="zendesk",
        product_category="Helpdesk",
        max_pages=5,
        metadata={},
        date_cutoff="2026-03-01",
    )
    selected = _select_serp_question_urls(
        [
            {
                "url": "https://www.quora.com/Whats-your-take-on-Zendesk-not-disclosing-the-terms-of-its-acquisition-of-Forethought",
                "snippet": "What are some alternatives to Zendesk and how hard is migration to Freshdesk?",
            },
            {
                "url": "https://www.quora.com/How-do-I-migrate-from-zendesk-to-Freshdesk",
                "snippet": "How do I migrate from Zendesk to Freshdesk?",
            },
        ],
        target,
    )
    assert selected == [
        "https://www.quora.com/How-do-I-migrate-from-zendesk-to-Freshdesk",
    ]


def test_question_root_url_normalizes_answer_permalinks():
    assert _question_root_url(
        "https://www.quora.com/What-is-a-cheaper-alternative-to-Intercom-for-startups/answer/Alice",
    ) == "https://www.quora.com/What-is-a-cheaper-alternative-to-Intercom-for-startups"
    assert _question_root_url(
        "https://www.quora.com/What-is-a-cheaper-alternative-to-Intercom-for-startups/answers/123",
    ) == "https://www.quora.com/What-is-a-cheaper-alternative-to-Intercom-for-startups"
