"""Tests for Trustpilot raw-page audit helpers."""

from atlas_brain.services.scraping.trustpilot_audit import (
    analyze_trustpilot_html_field_hints,
    analyze_trustpilot_jsonld_fields,
    analyze_trustpilot_raw_html,
)


def test_analyze_trustpilot_jsonld_fields_detects_author_employer_metadata():
    html = """
    <html><body>
      <script type="application/ld+json">
      {
        "@context": "https://schema.org",
        "@type": "Organization",
        "review": [{
          "@type": "Review",
          "reviewBody": "Detailed review body here.",
          "author": {
            "@type": "Person",
            "name": "Jane Reviewer",
            "jobTitle": "VP Operations",
            "worksFor": {"@type": "Organization", "name": "Acme Corp"}
          }
        }]
      }
      </script>
    </body></html>
    """

    report = analyze_trustpilot_jsonld_fields(html)

    assert report["review_object_count"] == 1
    assert report["author_object_count"] == 1
    assert report["author_job_title_count"] == 1
    assert report["author_company_count"] == 1
    assert "author.jobTitle" in report["matched_keys"]
    assert "author.worksFor.name" in report["matched_keys"]


def test_analyze_trustpilot_html_field_hints_detects_markup_keywords():
    html = """
    <html><body>
      <article data-service-review-card-paper="true">
        <span class="reviewer-title">Chief Technology Officer</span>
        <span class="reviewer-company">Northwind</span>
        <script>window.__STATE__ = {"jobTitle":"Chief Technology Officer","companyName":"Northwind"}</script>
      </article>
    </body></html>
    """

    report = analyze_trustpilot_html_field_hints(html)

    assert report["review_card_count"] == 1
    assert report["keyword_hits"]["jobTitle"] >= 1
    assert report["keyword_hits"]["companyName"] >= 1
    assert report["selector_hits"]["reviewer_title"] >= 1
    assert report["selector_hits"]["reviewer_company"] >= 1


def test_analyze_trustpilot_raw_html_combines_jsonld_and_markup_signals():
    html = """
    <html><body>
      <script type="application/ld+json">
      {"@type":"Review","reviewBody":"Body","author":{"name":"Sam","jobTitle":"Director","worksFor":{"name":"Acme"}}}
      </script>
      <article data-service-review-card-paper="true">
        <div class="reviewer-company">Acme</div>
      </article>
    </body></html>
    """

    report = analyze_trustpilot_raw_html(html)

    assert report["employer_fields_present"] is True
    assert report["title_fields_present"] is True


def test_analyze_trustpilot_raw_html_ignores_page_level_company_details():
    html = """
    <html><body>
      <div class="styles_companyDetails__30xkf">Company details for Atlassian</div>
      <article data-service-review-card-paper="true">
        <div class="styles_consumerDetails__POC79">
          <span class="styles_consumerName__xKr9c">Ivan Burmistrov</span>
          <div class="styles_consumerExtraDetails__NY6RP">Feb 19, 2026</div>
        </div>
      </article>
    </body></html>
    """

    report = analyze_trustpilot_raw_html(html)

    assert report["markup"]["selector_hits"]["reviewer_company"] == 0
    assert report["markup"]["selector_hits"]["reviewer_title"] == 0
    assert report["employer_fields_present"] is False
    assert report["title_fields_present"] is False
