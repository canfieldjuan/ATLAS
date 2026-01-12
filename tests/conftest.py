"""
Pytest fixtures for Atlas Brain end-to-end testing.

Provides fixtures for:
- Database initialization and cleanup
- Orchestrator with mocked services
- Session and conversation management
"""

import asyncio
import os
from datetime import date
from typing import AsyncGenerator, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
import pytest_asyncio

# Set test environment before importing atlas modules
os.environ.setdefault("ATLAS_DB_ENABLED", "true")
os.environ.setdefault("ATLAS_DB_HOST", "localhost")
os.environ.setdefault("ATLAS_DB_PORT", "5432")
os.environ.setdefault("ATLAS_DB_DATABASE", "atlas")
os.environ.setdefault("ATLAS_DB_USER", "atlas")
os.environ.setdefault("ATLAS_DB_PASSWORD", "atlas_dev_password")


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


class MockIntentParser:
    """Mock intent parser that returns predictable results."""

    def __init__(self, return_intent: bool = False):
        self.return_intent = return_intent
        self.parse_calls = []

    async def parse(self, text: str):
        self.parse_calls.append(text)

        if not self.return_intent:
            return None

        # Check for device control keywords
        text_lower = text.lower()
        if any(word in text_lower for word in ["turn on", "turn off", "switch"]):
            from atlas_brain.capabilities.actions import Intent
            return Intent(
                action="turn_on" if "turn on" in text_lower else "turn_off",
                target_type="light",
                target_name="living room",
                confidence=0.95,
            )
        return None


class MockActionDispatcher:
    """Mock action dispatcher for testing."""

    def __init__(self, success: bool = True):
        self.success = success
        self.dispatch_calls = []

    async def dispatch_intent(self, intent):
        self.dispatch_calls.append(intent)

        from atlas_brain.capabilities.protocols import ActionResult
        return ActionResult(
            success=self.success,
            message="Done" if self.success else "Failed",
        )


class MockLLM:
    """Mock LLM service for testing."""

    def __init__(self, response: str = "This is a test response."):
        self.response = response
        self.chat_calls = []

    def chat(self, messages, max_tokens=100, temperature=0.7):
        self.chat_calls.append((messages, max_tokens, temperature))
        return {"response": self.response}


@pytest.fixture
def mock_intent_parser():
    """Create a mock intent parser."""
    return MockIntentParser(return_intent=True)


@pytest.fixture
def mock_action_dispatcher():
    """Create a mock action dispatcher."""
    return MockActionDispatcher(success=True)


@pytest.fixture
def mock_llm():
    """Create a mock LLM."""
    return MockLLM()


@pytest_asyncio.fixture
async def test_orchestrator(
    db_pool,
    test_session,
    mock_intent_parser,
    mock_action_dispatcher,
    mock_llm,
):
    """
    Create an orchestrator with mocked services for testing.

    Includes mocked:
    - Intent parser
    - Action dispatcher
    - LLM
    - TTS (disabled)
    """
    from atlas_brain.orchestration.orchestrator import Orchestrator, OrchestratorConfig

    config = OrchestratorConfig(
        wake_word_enabled=False,
        require_wake_word=False,
        auto_execute=True,
    )

    orchestrator = Orchestrator(
        config=config,
        session_id=str(test_session),
    )

    # Inject mocks
    orchestrator._intent_parser = mock_intent_parser
    orchestrator._action_dispatcher = mock_action_dispatcher
    orchestrator._llm = mock_llm
    orchestrator._tts = None  # Disable TTS

    yield orchestrator


@pytest_asyncio.fixture
async def cleanup_test_data(db_pool, test_session):
    """Clean up test data after tests."""
    yield

    # Delete conversation turns for test session
    try:
        await db_pool.execute(
            "DELETE FROM conversation_turns WHERE session_id = $1",
            test_session,
        )
    except Exception:
        pass
