"""Tests for JSON-LD HTML supplementation on structured review parsers."""

import sys
from unittest.mock import MagicMock

for _mod in (
    "torch", "torchaudio", "transformers", "accelerate", "bitsandbytes",
    "PIL", "PIL.Image", "numpy", "cv2", "sounddevice", "soundfile",
    "nemo.collections", "nemo.collections.asr",
    "nemo.collections.asr.models",
    "asyncpg",
    "playwright", "playwright.async_api",
    "playwright_stealth",
    "curl_cffi", "curl_cffi.requests",
):
    sys.modules.setdefault(_mod, MagicMock())


def test_capterra_supplement_fills_firmographics_from_html():
    from atlas_brain.services.scraping.parsers.capterra import _supplement_pros_cons_from_html

    reviews = [{
        "source_review_id": "https://www.capterra.com/reviews/12345",
        "summary": None,
        "review_text": "This product is great for our operations team and easy to adopt.",
        "pros": None,
        "cons": None,
        "reviewer_title": None,
        "reviewer_company": None,
        "company_size_raw": None,
        "reviewer_industry": None,
    }]
    html = """
    <html><body>
      <div id="review-12345" class="review-card">
        <h3 class="review-title">Great for lean ops</h3>
        <div class="review-text">This product is great for our operations team and easy to adopt.</div>
        <div class="pros">Fast onboarding</div>
        <div class="cons">Limited dashboards</div>
        <div class="title">Operations Manager</div>
        <div class="company">Acme Corp</div>
        <div class="employees">11-50 employees</div>
        <div class="industry">Computer Software</div>
      </div>
    </body></html>
    """

    _supplement_pros_cons_from_html(html, reviews)

    review = reviews[0]
    assert review["pros"] == "Fast onboarding"
    assert review["cons"] == "Limited dashboards"
    assert review["reviewer_title"] == "Operations Manager"
    assert review["reviewer_company"] == "Acme Corp"
    assert review["company_size_raw"] == "11-50 employees"
    assert review["reviewer_industry"] == "Computer Software"


def test_capterra_supplement_fills_live_dom_metadata():
    from atlas_brain.services.scraping.parsers.capterra import _supplement_pros_cons_from_html

    reviews = [{
        "source_review_id": None,
        "summary": None,
        "review_text": "It is user-friendly, visually organized, and easy to adopt without a steep learning curve.",
        "pros": None,
        "cons": None,
        "reviewer_title": None,
        "reviewer_company": None,
        "company_size_raw": None,
        "reviewer_industry": None,
    }]
    html = """
    <html><body>
      <div class="flex scroll-mt-[200px] flex-col gap-y-6">
        <div class="capterra-live-card">
          <div class="flex w-full flex-row flex-wrap justify-between gap-x-6 gap-y-2 lg:justify-start">
            <div><img data-testid="reviewer-profile-pic" alt="Namratha C. avatar" src="/avatar.png" /></div>
            <div class="typo-10 text-neutral-90 w-full lg:w-fit">
              <span class="typo-20 text-neutral-99 font-semibold">Namratha C.</span><br/>
              Product owner<br/>
              Insurance, 1,001 - 5,000 employees<br/>
              Used the software for: 1-2 years
            </div>
          </div>
          <span>5.0</span><span>Overall Rating</span>
          <p>It is user-friendly, visually organized, and easy to adopt without a steep learning curve.</p>
          <p>Fast onboarding and clear workflow visibility.</p>
          <p>Reporting is limited for complex projects.</p>
        </div>
      </div>
    </body></html>
    """

    _supplement_pros_cons_from_html(html, reviews)

    review = reviews[0]
    assert review["pros"] == "Fast onboarding and clear workflow visibility."
    assert review["cons"] == "Reporting is limited for complex projects."
    assert review["reviewer_title"] == "Product owner"
    assert review["reviewer_company"] is None
    assert review["company_size_raw"] == "1,001 - 5,000 employees"
    assert review["reviewer_industry"] == "Insurance"


def test_capterra_parse_html_handles_live_dom_shape():
    from atlas_brain.services.scraping.parsers import ScrapeTarget
    from atlas_brain.services.scraping.parsers.capterra import _parse_html

    html = """
    <html><body>
      <div class="flex scroll-mt-[200px] flex-col gap-y-6">
        <div class="capterra-live-card">
          <div class="flex w-full flex-row flex-wrap justify-between gap-x-6 gap-y-2 lg:justify-start">
            <div><img data-testid="reviewer-profile-pic" alt="Namratha C. avatar" src="/avatar.png" /></div>
            <div class="typo-10 text-neutral-90 w-full lg:w-fit">
              <span class="typo-20 text-neutral-99 font-semibold">Namratha C.</span><br/>
              Product owner<br/>
              Computer Software<br/>
              Used the software for: 1-2 years
            </div>
          </div>
          <span>5.0</span><span>Overall Rating</span>
          <p>It is user-friendly, visually organized, and easy to adopt without a steep learning curve.</p>
          <p>Fast onboarding and clear workflow visibility.</p>
          <p>Reporting is limited for complex projects.</p>
        </div>
      </div>
    </body></html>
    """
    target = ScrapeTarget(
        id="debug",
        source="capterra",
        vendor_name="Trello",
        product_name="Trello",
        product_slug="211559/Trello",
        product_category="project-management",
        max_pages=1,
        metadata={},
    )

    reviews = _parse_html(html, target, set())

    assert len(reviews) == 1
    review = reviews[0]
    assert review["rating"] == 5.0
    assert review["review_text"] == "It is user-friendly, visually organized, and easy to adopt without a steep learning curve."
    assert review["pros"] == "Fast onboarding and clear workflow visibility."
    assert review["cons"] == "Reporting is limited for complex projects."
    assert review["reviewer_title"] == "Product owner"
    assert review["reviewer_industry"] == "Computer Software"


def test_software_advice_supplement_matches_by_review_text_fingerprint():
    from atlas_brain.services.scraping.parsers.software_advice import _supplement_reviews_from_html

    reviews = [{
        "source_review_id": "5c49b947bff85fec",
        "summary": None,
        "review_text": "Our overall experience with Trello was excellent. It is easy to use and keeps our team aligned every day.",
        "pros": None,
        "cons": None,
        "reviewer_title": None,
        "reviewer_company": None,
        "company_size_raw": None,
        "reviewer_industry": None,
    }]
    html = """
    <html><body>
      <div data-review-id="review-card-1" class="review-card">
        <div class="review-title">Excellent for daily planning</div>
        <div class="review-text">Our overall experience with Trello was excellent. It is easy to use and keeps our team aligned every day.</div>
        <div class="pros">Very intuitive</div>
        <div class="cons">Reporting is basic</div>
        <div class="role">Marketing Manager</div>
        <div class="company">Northwind</div>
        <div class="employees">51-200 employees</div>
        <div class="sector">Marketing and Advertising</div>
      </div>
    </body></html>
    """

    _supplement_reviews_from_html(html, reviews)

    review = reviews[0]
    assert review["pros"] == "Very intuitive"
    assert review["cons"] == "Reporting is basic"
    assert review["reviewer_title"] == "Marketing Manager"
    assert review["reviewer_company"] == "Northwind"
    assert review["company_size_raw"] == "51-200 employees"
    assert review["reviewer_industry"] == "Marketing and Advertising"


def test_software_advice_supplement_reads_react_payload():
    from atlas_brain.services.scraping.parsers.software_advice import _supplement_reviews_from_html

    reviews = [{
        "source_review_id": None,
        "summary": None,
        "review_text": "Good for leave tracking and onboarding",
        "pros": None,
        "cons": None,
        "reviewer_name": "Tomas",
        "reviewer_title": None,
        "reviewer_company": None,
        "company_size_raw": None,
        "reviewer_industry": None,
        "reviewed_at": "2023-11-09T10:06:05Z",
    }]
    html = """
    <html><body>
      <script>self.__next_f.push([1,"5b:{\\\"companySize\\\":\\\"C\\\",\\\"firstName\\\":\\\"Tomas\\\",\\\"industry\\\":\\\"Computer Software\\\",\\\"isValidated\\\":true,\\\"lastName\\\":\\\"Ruiz\\\",\\\"profilePictureUrl\\\":null}\\n5a:{\\\"consText\\\":\\\"Needs better sync\\\",\\\"generalComments\\\":\\\"Good for leave tracking and onboarding\\\",\\\"id\\\":\\\"Capterra___6017101\\\",\\\"overallRating\\\":4,\\\"prosText\\\":\\\"Easy to use\\\",\\\"reviewer\\\":\\\"$5b\\\",\\\"reviewerAnonymityOn\\\":false,\\\"title\\\":\\\"Good to track annual leaves\\\",\\\"writtenOn\\\":\\\"2023-11-09T10:06:05Z\\\"}"])</script>
    </body></html>
    """

    _supplement_reviews_from_html(html, reviews)

    review = reviews[0]
    assert review["pros"] == "Easy to use"
    assert review["cons"] == "Needs better sync"
    assert review["company_size_raw"] == "11-50"
    assert review["reviewer_industry"] == "Computer Software"


def test_software_advice_supplement_uses_payload_even_with_stale_review_container():
    from atlas_brain.services.scraping.parsers.software_advice import _supplement_reviews_from_html

    reviews = [{
        "source_review_id": None,
        "summary": None,
        "review_text": "Good for leave tracking and onboarding",
        "pros": None,
        "cons": None,
        "reviewer_name": "Tomas",
        "reviewer_title": None,
        "reviewer_company": None,
        "company_size_raw": None,
        "reviewer_industry": None,
        "reviewed_at": "2023-11-09T10:06:05Z",
    }]
    html = """
    <html><body>
      <div itemtype="Review">aggregate container only</div>
      <script>self.__next_f.push([1,"5b:{\\\"companySize\\\":\\\"C\\\",\\\"firstName\\\":\\\"Tomas\\\",\\\"industry\\\":\\\"Computer Software\\\"}\\n5a:{\\\"generalComments\\\":\\\"Good for leave tracking and onboarding\\\",\\\"id\\\":\\\"Capterra___6017101\\\",\\\"overallRating\\\":4,\\\"prosText\\\":\\\"Easy to use\\\",\\\"consText\\\":\\\"Needs better sync\\\",\\\"reviewer\\\":\\\"$5b\\\",\\\"writtenOn\\\":\\\"2023-11-09T10:06:05Z\\\"}"])</script>
    </body></html>
    """

    _supplement_reviews_from_html(html, reviews)

    review = reviews[0]
    assert review["company_size_raw"] == "11-50"
    assert review["reviewer_industry"] == "Computer Software"


def test_software_advice_parse_html_uses_react_payload_when_cards_missing():
    from atlas_brain.services.scraping.parsers import ScrapeTarget
    from atlas_brain.services.scraping.parsers.software_advice import _parse_html

    html = """
    <html><body>
      <script>self.__next_f.push([1,"5b:{\\\"companySize\\\":\\\"D\\\",\\\"firstName\\\":\\\"Holly\\\",\\\"industry\\\":\\\"Hospital &amp; Health Care\\\",\\\"isValidated\\\":true,\\\"lastName\\\":\\\"Dohner\\\",\\\"profilePictureUrl\\\":null}\\n5a:{\\\"consText\\\":\\\"Needs better integrations\\\",\\\"generalComments\\\":\\\"Very positive overall experience\\\",\\\"id\\\":\\\"Capterra___777\\\",\\\"overallRating\\\":5,\\\"prosText\\\":\\\"Simple to use\\\",\\\"reviewer\\\":\\\"$5b\\\",\\\"reviewerAnonymityOn\\\":false,\\\"title\\\":\\\"Easy to use\\\",\\\"writtenOn\\\":\\\"2026-01-19T11:57:29Z\\\"}"])</script>
    </body></html>
    """
    target = ScrapeTarget(
        id="debug",
        source="software_advice",
        vendor_name="BambooHR",
        product_name="BambooHR",
        product_slug="hr/bamboohr-profile",
        product_category="hr",
        max_pages=1,
        metadata={},
    )

    reviews = _parse_html(html, target, set())

    assert len(reviews) == 1
    review = reviews[0]
    assert review["rating"] == 5.0
    assert review["review_text"] == "Very positive overall experience"
    assert review["pros"] == "Simple to use"
    assert review["cons"] == "Needs better integrations"
    assert review["company_size_raw"] == "51-200"
    assert review["reviewer_industry"] == "Hospital & Health Care"


def test_software_advice_parse_html_uses_payload_when_stale_review_container_exists():
    from atlas_brain.services.scraping.parsers import ScrapeTarget
    from atlas_brain.services.scraping.parsers.software_advice import _parse_html

    html = """
    <html><body>
      <div itemtype="Review">aggregate container only</div>
      <script>self.__next_f.push([1,"5b:{\\\"companySize\\\":\\\"D\\\",\\\"firstName\\\":\\\"Holly\\\",\\\"industry\\\":\\\"Hospital &amp; Health Care\\\"}\\n5a:{\\\"generalComments\\\":\\\"Very positive overall experience\\\",\\\"id\\\":\\\"Capterra___777\\\",\\\"overallRating\\\":5,\\\"prosText\\\":\\\"Simple to use\\\",\\\"consText\\\":\\\"Needs better integrations\\\",\\\"reviewer\\\":\\\"$5b\\\",\\\"title\\\":\\\"Easy to use\\\",\\\"writtenOn\\\":\\\"2026-01-19T11:57:29Z\\\"}"])</script>
    </body></html>
    """
    target = ScrapeTarget(
        id="debug",
        source="software_advice",
        vendor_name="BambooHR",
        product_name="BambooHR",
        product_slug="hr/bamboohr-profile",
        product_category="hr",
        max_pages=1,
        metadata={},
    )

    reviews = _parse_html(html, target, set())

    assert len(reviews) == 1
    review = reviews[0]
    assert review["review_text"] == "Very positive overall experience"
    assert review["company_size_raw"] == "51-200"
    assert review["reviewer_industry"] == "Hospital & Health Care"
