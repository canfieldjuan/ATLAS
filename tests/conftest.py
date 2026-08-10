"""
Pytest fixtures for Atlas Brain end-to-end testing.

Provides fixtures for:
- Database initialization and cleanup
- Orchestrator with mocked services
- Session and conversation management
"""

import asyncio
import os
from pathlib import Path
from uuid import UUID, uuid4

import pytest
import pytest_asyncio

# Set test environment before importing atlas modules
os.environ.setdefault("ATLAS_DB_ENABLED", "true")
os.environ.setdefault("ATLAS_DB_HOST", "localhost")
os.environ.setdefault("ATLAS_DB_PORT", "5433")
os.environ.setdefault("ATLAS_DB_DATABASE", "atlas")
os.environ.setdefault("ATLAS_DB_USER", "atlas")
os.environ.setdefault("ATLAS_DB_PASSWORD", "atlas_dev_password")

# Tests must never inherit the developer's git configuration. A global
# core.hooksPath made the pre-push hook fire inside the throwaway repos tests
# build, and that hook re-ran this suite, which pushed again -- unbounded
# recursion that exhausted system memory. Assigned, not setdefault: hermetic
# git is a guarantee, not a default a stale environment can override.
os.environ["GIT_CONFIG_GLOBAL"] = os.devnull
os.environ["GIT_CONFIG_SYSTEM"] = os.devnull
# Closed source for env-delivered Git config controls. Exact names are the
# complete non-numbered set this slice handles; prefixes cover Git's numbered
# key/value pairs. Other GIT_CONFIG_* names stay unchanged unless a future proof
# shows Git treats them as config injection.
_GIT_CONFIG_INJECTION_ENV_NAMES = ("GIT_CONFIG_COUNT", "GIT_CONFIG_PARAMETERS")
_GIT_CONFIG_INJECTION_ENV_PREFIXES = ("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_")
for name in list(os.environ):
    if (
        name in _GIT_CONFIG_INJECTION_ENV_NAMES
        or name.startswith(_GIT_CONFIG_INJECTION_ENV_PREFIXES)
    ):
        os.environ.pop(name, None)

# GIT_TEMPLATE_DIR is not config, but it can install a developer hook into every
# fixture repo created by plain git init. Clear it with the inherited config
# controls so throwaway repos cannot copy hook state before their first push.
os.environ.pop("GIT_TEMPLATE_DIR", None)

# Keep the recursion guard scripts/local_pr_review.sh exports into the unit
# gate. A test that pushes into a repo carrying a managed pre-push hook would
# otherwise re-enter review -> unit gate -> pytest and exhaust the machine, so
# the guard is preserved by default rather than dropped suite-wide. The handful
# of tests that assert the hook *runs* clear it per module (see
# tests/test_install_local_pr_hook.py and tests/test_push_pr_wrapper.py), which
# keeps every other test protected.
os.environ.setdefault("ATLAS_SKIP_LOCAL_PR_REVIEW", "1")

# Neutralizing global config also drops user.name/user.email. Most git fixtures
# set an identity locally, but not all of them do, so supply one via env -- it
# applies with no config file at all. setdefault so a fixture can still override.
os.environ.setdefault("GIT_AUTHOR_NAME", "Atlas Test")
os.environ.setdefault("GIT_AUTHOR_EMAIL", "test@example.invalid")
os.environ.setdefault("GIT_COMMITTER_NAME", "Atlas Test")
os.environ.setdefault("GIT_COMMITTER_EMAIL", "test@example.invalid")

# The unit backstop installs asyncpg, so load the real driver before test module
# collection. This prevents legacy import-time sys.modules.setdefault("asyncpg",
# MagicMock()) helpers from poisoning later DB-fixture tests.
try:
    import asyncpg  # noqa: F401
    import asyncpg.exceptions  # noqa: F401
except ModuleNotFoundError:
    pass


_SELF_POOL_LIVE_FILES = {
    "test_b2b_challenger_claims_api_live.py",
    "test_b2b_vendor_claims_api_live.py",
    "test_evidence_claim_audit_live.py",
    "test_evidence_claim_builder_live.py",
    "test_evidence_claim_repository_live.py",
    "test_live_autonomous.py",
    "test_reasoning_live.py",
    "test_vendor_dashboard_claims_live.py",
}

_INTEGRATION_FIXTURE_NAMES = {"db_pool", "live_pool"}


def _markexpr_excludes_integration(config) -> bool:
    markexpr = getattr(config.option, "markexpr", "") or ""
    return "not integration" in " ".join(markexpr.lower().split())


def pytest_ignore_collect(collection_path, config):
    if _markexpr_excludes_integration(config):
        if Path(str(collection_path)).name in _SELF_POOL_LIVE_FILES:
            return True
    return None


def pytest_collection_modifyitems(session, config, items):
    integration = pytest.mark.integration
    for item in items:
        if (
            _INTEGRATION_FIXTURE_NAMES.intersection(item.fixturenames)
            or Path(str(item.fspath)).name in _SELF_POOL_LIVE_FILES
        ):
            item.add_marker(integration)


@pytest.fixture(autouse=True)
def _disable_leads_ntfy_topic(monkeypatch):
    """Repo-wide safety: keep the new-lead push OFF in EVERY test, regardless of
    the checkout's .env, so no route-level test in any module can publish fake
    lead PII to the live ntfy topic (the public intake route wires the production
    notifier). Tests that exercise the transport re-enable it explicitly, and
    their own monkeypatch — applied after this autouse — wins.
    (Codex #2332 R2/R12: the per-file disable fixture did not cover route tests
    such as test_eom_lead_pipeline_integration.py / test_eom_sent_email_tenant_scope.py.)
    """
    try:
        from atlas_brain.config import settings
    except Exception:
        return
    alerts = getattr(settings, "alerts", None)
    if alerts is not None and hasattr(alerts, "leads_ntfy_topic"):
        monkeypatch.setattr(alerts, "leads_ntfy_topic", "", raising=False)


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="function")
async def db_pool():
    """
    Initialize database pool for testing.

    Yields the pool and cleans up after tests.
    """
    from atlas_brain.storage.database import get_db_pool, DatabasePool

    # Reset global pool to ensure fresh state
    import atlas_brain.storage.database as db_module
    db_module._db_pool = None

    pool = get_db_pool()
    await pool.initialize()

    yield pool

    await pool.close()
    db_module._db_pool = None


@pytest_asyncio.fixture
async def test_session(db_pool) -> UUID:
    """
    Create a test session in the database.

    Creates a new session for testing and cleans up after.
    """
    from atlas_brain.storage.repositories.session import get_session_repo
    import atlas_brain.storage.repositories.session as session_module

    # Reset global repo
    session_module._session_repo = None

    repo = get_session_repo()
    session = await repo.create_session(
        user_id=None,  # Anonymous session for tests
        terminal_id="test-terminal",
    )

    yield session.id

    # Cleanup: close and delete session
    try:
        await repo.close_session(session.id)
    except Exception:
        pass


async def create_test_user(db_pool, name: str = "Test User") -> UUID:
    """Helper to create a test user in the database."""
    user_id = uuid4()
    await db_pool.execute(
        """
        INSERT INTO users (id, name, created_at)
        VALUES ($1, $2, NOW())
        ON CONFLICT (id) DO NOTHING
        """,
        user_id,
        name,
    )
    return user_id


@pytest_asyncio.fixture
async def test_user_session(db_pool) -> tuple[UUID, UUID]:
    """
    Create a test session with a user ID.

    Returns (session_id, user_id) tuple.
    """
    from atlas_brain.storage.repositories.session import get_session_repo
    import atlas_brain.storage.repositories.session as session_module

    session_module._session_repo = None
    repo = get_session_repo()

    # Create user first (required for foreign key)
    user_id = await create_test_user(db_pool, "Test User")

    session = await repo.create_session(
        user_id=user_id,
        terminal_id="test-terminal",
    )

    yield session.id, user_id

    # Cleanup
    try:
        await repo.close_session(session.id)
        await db_pool.execute("DELETE FROM users WHERE id = $1", user_id)
    except Exception:
        pass


@pytest_asyncio.fixture
async def conversation_repo(db_pool):
    """Get conversation repository with fresh state."""
    from atlas_brain.storage.repositories.conversation import get_conversation_repo
    import atlas_brain.storage.repositories.conversation as conv_module

    conv_module._conversation_repo = None
    return get_conversation_repo()


@pytest_asyncio.fixture
async def session_repo(db_pool):
    """Get session repository with fresh state."""
    from atlas_brain.storage.repositories.session import get_session_repo
    import atlas_brain.storage.repositories.session as session_module

    session_module._session_repo = None
    return get_session_repo()
