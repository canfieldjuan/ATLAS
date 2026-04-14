"""Tests for CAPTCHA proxy selection."""

from atlas_brain.config import settings
from atlas_brain.services.scraping import captcha


def test_select_capsolver_compatible_proxy_prefers_non_brightdata_residential(monkeypatch):
    monkeypatch.setattr(
        settings.b2b_scrape,
        "captcha_proxy_url",
        "http://user:pass@brd.superproxy.io:33335",
        raising=False,
    )
    monkeypatch.setattr(
        settings.b2b_scrape,
        "proxy_residential_urls",
        ",".join(
            [
                "http://user:pass@brd.superproxy.io:33335",
                "http://user:pass@na.proxy.2captcha.com:2334",
            ]
        ),
        raising=False,
    )

    selected = captcha._select_capsolver_compatible_proxy_url()

    assert selected == "http://user:pass@na.proxy.2captcha.com:2334"


def test_select_capsolver_compatible_proxy_uses_configured_when_already_compatible(monkeypatch):
    monkeypatch.setattr(
        settings.b2b_scrape,
        "captcha_proxy_url",
        "http://user:pass@na.proxy.2captcha.com:2334",
        raising=False,
    )
    monkeypatch.setattr(
        settings.b2b_scrape,
        "proxy_residential_urls",
        "http://user:pass@na.proxy.2captcha.com:2334",
        raising=False,
    )

    selected = captcha._select_capsolver_compatible_proxy_url()

    assert selected == "http://user:pass@na.proxy.2captcha.com:2334"
