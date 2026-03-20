"""Tests for CAPTCHA solver task construction."""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

for _mod in ("asyncpg",):
    sys.modules.setdefault(_mod, MagicMock())


class _FakeResponse:
    def __init__(self, payload):
        self.status_code = 200
        self._payload = payload

    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self, post_mock):
        self.post = post_mock

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None


@pytest.mark.asyncio
async def test_2captcha_cloudflare_uses_proxyless_task_without_proxy():
    from atlas_brain.services.scraping.captcha import CaptchaSolver, CaptchaType

    post_mock = AsyncMock(return_value=_FakeResponse({"errorId": 0, "taskId": 123}))
    fake_client = _FakeClient(post_mock)
    solver = CaptchaSolver(provider="2captcha", api_key="test-key")

    with patch("httpx.AsyncClient", return_value=fake_client), \
         patch.object(solver, "_poll_2captcha", new=AsyncMock(return_value={"cf_clearance": "ok"})):
        await solver._solve_2captcha(
            CaptchaType.CLOUDFLARE,
            "https://www.getapp.com/software/project-management-software/a/acme/reviews/",
            '<html><div data-sitekey="site-key"></div></html>',
            user_agent="Mozilla/5.0",
            proxy_url=None,
        )

    task = post_mock.await_args.kwargs["json"]["task"]
    assert task["type"] == "TurnstileTaskProxyless"
    assert task["websiteKey"] == "site-key"
