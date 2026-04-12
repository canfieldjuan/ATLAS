from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from atlas_brain.api import blog_public as mod


@pytest.mark.asyncio
async def test_list_published_posts_normalizes_blank_topic_and_query_defaults(monkeypatch):
    class Pool:
        async def fetch(self, query, *args):
            assert "topic_type = $1" not in query
            assert args == (50, 0)
            return []

        async def fetchval(self, query, *args):
            assert args == ()
            return 0

    monkeypatch.setattr(mod, "get_db_pool", lambda: Pool())

    result = await mod.list_published_posts(topic_type="   ")

    assert result == {"posts": [], "total": 0}


@pytest.mark.asyncio
async def test_list_published_posts_trims_active_topic_type(monkeypatch):
    class Pool:
        async def fetch(self, query, *args):
            assert "topic_type = $1" in query
            assert args == ("competitive_set", 25, 5)
            return []

        async def fetchval(self, query, *args):
            assert args == ("competitive_set",)
            return 0

    monkeypatch.setattr(mod, "get_db_pool", lambda: Pool())

    result = await mod.list_published_posts(topic_type="  competitive_set  ", limit=25, offset=5)

    assert result == {"posts": [], "total": 0}


@pytest.mark.asyncio
async def test_get_published_post_rejects_blank_slug_before_db_touch(monkeypatch):
    def _fail_pool():
        raise AssertionError("db touched")

    monkeypatch.setattr(mod, "get_db_pool", _fail_pool)

    with pytest.raises(mod.HTTPException) as exc:
        await mod.get_published_post("   ")

    assert exc.value.status_code == 422
    assert exc.value.detail == "slug is required"


@pytest.mark.asyncio
async def test_get_published_post_trims_slug_before_lookup(monkeypatch):
    pool = SimpleNamespace(fetchrow=AsyncMock(return_value=None))
    monkeypatch.setattr(mod, "get_db_pool", lambda: pool)

    result = await mod.get_published_post("  launch-post  ")

    assert pool.fetchrow.await_args.args[1] == "launch-post"
    assert result == {"post": None}
