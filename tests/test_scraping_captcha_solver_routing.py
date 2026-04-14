"""Tests for CAPTCHA solver routing by domain."""

from atlas_brain.services.scraping import captcha


def test_required_2captcha_domains_include_capterra():
    assert "capterra.com" in captcha._required_2captcha_domains()
    assert "gartner.com" in captcha._required_2captcha_domains()


def test_domain_in_set_matches_subdomains():
    assert captcha._domain_in_set("www.capterra.com", {"capterra.com"}) is True
    assert captcha._domain_in_set("api.gartner.com", {"gartner.com"}) is True
    assert captcha._domain_in_set("capterra.com", {"capterra.com"}) is True
    assert captcha._domain_in_set("example.com", {"capterra.com"}) is False


def test_detect_captcha_marks_cloudflare_hard_block_as_non_solvable():
    html = """
    <html><head><title>Attention Required! | Cloudflare</title></head>
    <body>
      <div id="cf-error-details">
        <h1>Sorry, you have been blocked</h1>
        <h2>You are unable to access capterra.com</h2>
        <div class="challenge-platform"></div>
      </div>
    </body></html>
    """

    detected = captcha.detect_captcha(html, 403)

    assert detected == captcha.CaptchaType.CLOUDFLARE_BLOCK
