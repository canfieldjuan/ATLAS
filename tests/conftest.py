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

# The unit backstop installs asyncpg, so load the real driver before test module
# collection. This prevents legacy import-time sys.modules.setdefault("asyncpg",
# MagicMock()) helpers from poisoning later DB-fixture tests.
try:
    import asyncpg  # noqa: F401
    import asyncpg.exceptions  # noqa: F401
except ModuleNotFoundError:
    pass


_INTEGRATION_TEST_FILES = {
    "test_b2b_challenger_claims_api_live.py",
    "test_b2b_enrichment_batch_integration.py",
    "test_b2b_enrichment_repair_batch_integration.py",
    "test_b2b_vendor_claims_api_live.py",
    "test_campaign_recipient_dedup.py",
    "test_comparisons_live.py",
    "test_evidence_claim_audit_live.py",
    "test_evidence_claim_builder_live.py",
    "test_evidence_claim_repository_live.py",
    "test_evidence_gate.py",
    "test_live_autonomous.py",
    "test_reasoning_live.py",
    "test_vendor_dashboard_claims_live.py",
}


def pytest_collection_modifyitems(session, config, items):
    integration = pytest.mark.integration
    for item in items:
        if Path(str(item.fspath)).name in _INTEGRATION_TEST_FILES:
            item.add_marker(integration)


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
