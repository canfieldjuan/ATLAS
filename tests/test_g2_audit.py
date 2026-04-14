"""Tests for G2 raw-page audit helpers."""

from atlas_brain.services.scraping.g2_audit import (
    analyze_g2_jsonld_fields,
    analyze_g2_raw_html,
    analyze_g2_review_cards,
)


def test_analyze_g2_jsonld_fields_detects_author_identity_metadata():
    html = """
    <html><body>
      <script type="application/ld+json">
      {
        "@context": "https://schema.org",
        "@type": "Product",
        "review": [{
          "@type": "Review",
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

    report = analyze_g2_jsonld_fields(html)

    assert report["review_object_count"] == 1
    assert report["author_job_title_count"] == 1
    assert report["author_company_count"] == 1


def test_analyze_g2_review_cards_detects_tailwind_identity_fields():
    html = """
    <html><body>
      <article class="review-card">
        <div>
          <div>
            <div class="elv-font-bold">Verified User</div>
          </div>
          <div>
            <div class="elv-text-subtle">Engineering Manager</div>
            <div class="elv-text-subtle">Acme Corp</div>
            <div class="elv-text-subtle">Computer Software</div>
            <div class="elv-text-subtle">Mid-Market (51-1000 emp.)</div>
          </div>
        </div>
      </article>
    </body></html>
    """

    report = analyze_g2_review_cards(html)

    assert report["review_card_count"] >= 1
    assert report["reviewer_title_count"] == 1
    assert report["reviewer_company_count"] == 1
    assert report["reviewer_industry_count"] == 1
    assert report["company_size_count"] == 1
    assert report["samples"][0]["reviewer_company"] == "Acme Corp"


def test_analyze_g2_raw_html_combines_jsonld_and_review_cards():
    html = """
    <html><body>
      <script type="application/ld+json">
      {"@type":"Review","author":{"name":"Sam","jobTitle":"Director","worksFor":{"name":"Acme"}}}
      </script>
      <article class="review-card">
        <div class="reviewer-title">Director</div>
        <div class="reviewer-company">Acme</div>
      </article>
    </body></html>
    """

    report = analyze_g2_raw_html(html)

    assert report["employer_fields_present"] is True
    assert report["title_fields_present"] is True
