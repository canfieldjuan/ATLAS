"""Tests for Gartner raw-page audit helpers."""

from atlas_brain.services.scraping.gartner_audit import (
    analyze_gartner_jsonld_fields,
    analyze_gartner_next_data_fields,
    analyze_gartner_raw_html,
)


def test_analyze_gartner_jsonld_fields_detects_author_identity_metadata():
    html = """
    <html><body>
      <script type="application/ld+json">
      {
        "@context": "https://schema.org",
        "@type": "SoftwareApplication",
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

    report = analyze_gartner_jsonld_fields(html)

    assert report["review_object_count"] == 1
    assert report["author_job_title_count"] == 1
    assert report["author_company_count"] == 1


def test_analyze_gartner_next_data_fields_detects_reviewer_identity_fields():
    html = """
    <html><body>
      <script id="__NEXT_DATA__" type="application/json">
      {
        "props": {
          "pageProps": {
            "serverSideXHRData": {
              "user-reviews-by-market-vendor-product": {
                "userReviews": [
                  {
                    "reviewId": 123,
                    "jobTitle": "Director",
                    "companyName": "Acme Corp",
                    "companySize": "201-500",
                    "industryName": "Software"
                  }
                ]
              },
              "vendor-snippets": {}
            }
          }
        }
      }
      </script>
    </body></html>
    """

    report = analyze_gartner_next_data_fields(html)

    assert report["review_object_count"] == 1
    assert report["reviewer_title_count"] == 1
    assert report["reviewer_company_count"] == 1
    assert report["company_size_count"] == 1
    assert report["reviewer_industry_count"] == 1


def test_analyze_gartner_raw_html_combines_jsonld_and_next_data():
    html = """
    <html><body>
      <script type="application/ld+json">
      {"@type":"SoftwareApplication","review":[{"@type":"Review","author":{"jobTitle":"Director","worksFor":{"name":"Acme"}}}]}
      </script>
      <script id="__NEXT_DATA__" type="application/json">
      {"props":{"pageProps":{"serverSideXHRData":{"user-reviews-by-market-vendor-product":{"userReviews":[{"reviewId":1,"jobTitle":"Director","companyName":"Acme","companySize":"201-500","industryName":"Software"}]},"vendor-snippets":{}}}}}
      </script>
    </body></html>
    """

    report = analyze_gartner_raw_html(html)

    assert report["employer_fields_present"] is True
    assert report["title_fields_present"] is True
    assert report["industry_fields_present"] is True
    assert report["company_size_fields_present"] is True
