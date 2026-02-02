"""
Phase 6: Comprehensive Email Workflow Testing & Validation.

Tests all email workflow functionality:
- Intent classification (all patterns)
- Draft preview flow
- Real email sending (estimate, proposal, generic)
- Email history query
- Follow-up reminder creation
- Context extraction (auto-fill)
- Edge cases and error handling
"""

import asyncio
import os
import sys

sys.path.insert(0, "/home/juan-canfield/Desktop/Atlas-LangGraph-Agents-ToolUse")

# Load environment variables from Atlas project .env file
from dotenv import load_dotenv
load_dotenv("/home/juan-canfield/Desktop/Atlas/.env")


# Database initialization flag
_db_initialized = False


async def initialize_database():
    """Initialize database pool for real tests."""
    global _db_initialized
    if _db_initialized:
        return True

    try:
        from atlas_brain.storage.database import get_db_pool
        pool = get_db_pool()
        await pool.initialize()
        _db_initialized = True
        print("  [DB] Database pool initialized successfully")
        return True
    except Exception as e:
        print(f"  [DB] Failed to initialize database: {e}")
        return False


async def shutdown_database():
    """Shutdown database pool."""
    global _db_initialized
    if _db_initialized:
        try:
            from atlas_brain.storage.database import get_db_pool
            pool = get_db_pool()
            await pool.close()
            _db_initialized = False
        except Exception:
            pass


class TestResults:
    """Track test results."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.tests = []

    def add(self, name: str, passed: bool, message: str = ""):
        self.tests.append({"name": name, "passed": passed, "message": message})
        if passed:
            self.passed += 1
        else:
            self.failed += 1
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if message and not passed:
            print(f"         {message}")

    def skip(self, name: str, reason: str):
        self.tests.append({"name": name, "passed": None, "message": reason})
        self.skipped += 1
        print(f"  [SKIP] {name} - {reason}")

    def summary(self):
        print(f"\n  Results: {self.passed} passed, {self.failed} failed, {self.skipped} skipped")
        return self.failed == 0


async def test_intent_classification():
    """Test 1: Intent Classification - All patterns."""
    print("\n" + "=" * 70)
    print("TEST 1: Intent Classification")
    print("=" * 70)

    os.environ["USE_REAL_TOOLS"] = "false"
    from atlas_brain.agents.graphs.email import run_email_workflow

    results = TestResults()

    test_cases = [
        # send_email patterns
        ("send email to test@example.com", "send_email"),
        ("email test@example.com", "send_email"),
        ("send an email", "send_email"),
        ("compose email to john@test.com", "send_email"),
        ("mail this to recipient@domain.com", "send_email"),

        # send_estimate patterns
        ("send estimate to client", "send_estimate"),
        ("send the estimate email", "send_estimate"),
        ("email the estimate", "send_estimate"),
        ("send cleaning estimate", "send_estimate"),
        ("estimate confirmation email", "send_estimate"),

        # send_proposal patterns
        ("send proposal to business", "send_proposal"),
        ("email the proposal", "send_proposal"),
        ("send cleaning proposal", "send_proposal"),
        ("proposal email to company", "send_proposal"),

        # query_history patterns
        ("what emails did I send", "query_history"),
        ("show email history", "query_history"),
        ("list sent emails", "query_history"),
        ("emails sent today", "query_history"),
        ("check my email history", "query_history"),
    ]

    for text, expected_intent in test_cases:
        result = await run_email_workflow(text)
        actual_intent = result.get("intent")
        passed = actual_intent == expected_intent
        results.add(
            f"'{text[:40]}...' -> {expected_intent}",
            passed,
            f"Got: {actual_intent}" if not passed else ""
        )

    return results.summary()


async def test_draft_preview_flow():
    """Test 2: Draft Preview Flow."""
    print("\n" + "=" * 70)
    print("TEST 2: Draft Preview Flow")
    print("=" * 70)

    os.environ["USE_REAL_TOOLS"] = "false"
    from atlas_brain.agents.graphs.email import run_email_workflow, send_email_confirmed

    results = TestResults()

    # Test estimate draft
    result = await run_email_workflow(
        "send estimate",
        to_address="test@example.com",
        client_name="Test Client",
        client_type="residential",
        address="123 Test St",
        service_date="February 1, 2026",
        service_time="9:00 AM",
        price="150.00",
    )

    results.add(
        "Estimate generates draft",
        result.get("awaiting_confirmation") == True,
        f"awaiting_confirmation={result.get('awaiting_confirmation')}"
    )
    results.add(
        "Draft has subject",
        bool(result.get("draft_subject")),
        f"draft_subject={result.get('draft_subject')}"
    )
    results.add(
        "Draft has body",
        bool(result.get("draft_body")),
        "Missing draft_body"
    )
    results.add(
        "Draft has recipient",
        result.get("draft_to") == "test@example.com",
        f"draft_to={result.get('draft_to')}"
    )
    results.add(
        "Response contains DRAFT PREVIEW",
        "DRAFT" in result.get("response", ""),
        "Response missing DRAFT marker"
    )

    # Test proposal draft
    result = await run_email_workflow(
        "send proposal",
        to_address="business@example.com",
        client_name="Acme Corp",
        client_type="business",
        contact_name="John Smith",
        address="456 Business Ave",
        areas_to_clean="Offices",
        cleaning_description="Full cleaning",
        price="500.00",
    )

    results.add(
        "Proposal generates draft",
        result.get("awaiting_confirmation") == True,
        f"awaiting_confirmation={result.get('awaiting_confirmation')}"
    )
    results.add(
        "Proposal draft has template",
        result.get("draft_template") == "proposal",
        f"draft_template={result.get('draft_template')}"
    )

    # Test confirm flow (mock)
    if result.get("awaiting_confirmation"):
        send_result = await send_email_confirmed(result)
        results.add(
            "Confirmed draft sends email",
            send_result.get("email_sent") == True,
            f"email_sent={send_result.get('email_sent')}"
        )
        results.add(
            "Send has message ID",
            bool(send_result.get("resend_message_id")),
            "Missing message ID"
        )

    return results.summary()


async def test_generic_email():
    """Test 3: Generic Email Sending."""
    print("\n" + "=" * 70)
    print("TEST 3: Generic Email Sending")
    print("=" * 70)

    os.environ["USE_REAL_TOOLS"] = "false"
    from atlas_brain.agents.graphs.email import run_email_workflow, send_email_confirmed

    results = TestResults()

    # Test with all required fields
    result = await run_email_workflow(
        "send email",
        to_address="test@example.com",
        subject="Test Subject",
        body="This is a test email body.",
    )

    results.add(
        "Generic email generates draft",
        result.get("awaiting_confirmation") == True,
        f"awaiting_confirmation={result.get('awaiting_confirmation')}"
    )
    results.add(
        "Draft template is generic",
        result.get("draft_template") == "generic",
        f"draft_template={result.get('draft_template')}"
    )

    # Test missing required fields
    result = await run_email_workflow(
        "send email",
        to_address="test@example.com",
        # Missing subject and body
    )

    results.add(
        "Missing fields triggers clarification",
        result.get("needs_clarification") == True,
        f"needs_clarification={result.get('needs_clarification')}"
    )
    results.add(
        "Clarification mentions missing fields",
        "subject" in result.get("clarification_prompt", "").lower() or "body" in result.get("clarification_prompt", "").lower(),
        f"clarification_prompt={result.get('clarification_prompt')}"
    )

    return results.summary()


async def test_email_history():
    """Test 4: Email History Query."""
    print("\n" + "=" * 70)
    print("TEST 4: Email History Query")
    print("=" * 70)

    os.environ["USE_REAL_TOOLS"] = "false"
    from atlas_brain.agents.graphs.email import run_email_workflow

    results = TestResults()

    # Test today query
    result = await run_email_workflow("what emails did I send today")

    results.add(
        "History query detected",
        result.get("intent") == "query_history",
        f"intent={result.get('intent')}"
    )
    results.add(
        "History queried flag set",
        result.get("history_queried") == True,
        f"history_queried={result.get('history_queried')}"
    )
    results.add(
        "Response has email count",
        "email" in result.get("response", "").lower(),
        "Response missing email info"
    )

    # Test week query
    result = await run_email_workflow("show emails from this week")

    results.add(
        "Week query works",
        result.get("history_queried") == True,
        f"history_queried={result.get('history_queried')}"
    )

    return results.summary()


async def test_follow_up_reminders():
    """Test 5: Follow-up Reminder Creation."""
    print("\n" + "=" * 70)
    print("TEST 5: Follow-up Reminder Creation")
    print("=" * 70)

    os.environ["USE_REAL_TOOLS"] = "false"
    from atlas_brain.agents.graphs.email import run_email_workflow, send_email_confirmed

    results = TestResults()

    # Test estimate with explicit follow-up
    result = await run_email_workflow(
        "send estimate",
        to_address="test@example.com",
        client_name="Follow-up Client",
        client_type="residential",
        address="123 Test St",
        service_date="February 1, 2026",
        service_time="9:00 AM",
        price="150.00",
        create_follow_up=True,
        follow_up_days=3,
    )

    results.add(
        "Estimate with follow-up generates draft",
        result.get("awaiting_confirmation") == True,
        f"awaiting_confirmation={result.get('awaiting_confirmation')}"
    )

    if result.get("awaiting_confirmation"):
        send_result = await send_email_confirmed(result)
        results.add(
            "Follow-up reminder created",
            send_result.get("follow_up_created") == True,
            f"follow_up_created={send_result.get('follow_up_created')}"
        )
        results.add(
            "Follow-up has reminder ID",
            bool(send_result.get("follow_up_reminder_id")),
            "Missing follow_up_reminder_id"
        )

    # Test proposal (auto follow-up)
    result = await run_email_workflow(
        "send proposal",
        to_address="business@example.com",
        client_name="Auto Follow-up Corp",
        client_type="business",
        contact_name="Jane Doe",
        address="789 Corp Blvd",
        areas_to_clean="All areas",
        cleaning_description="Complete cleaning",
        price="750.00",
    )

    if result.get("awaiting_confirmation"):
        send_result = await send_email_confirmed(result)
        results.add(
            "Proposal auto-creates follow-up",
            send_result.get("follow_up_created") == True,
            f"follow_up_created={send_result.get('follow_up_created')}"
        )

    return results.summary()


async def test_context_extraction():
    """Test 6: Context Extraction (Auto-fill)."""
    print("\n" + "=" * 70)
    print("TEST 6: Context Extraction (Auto-fill from Bookings)")
    print("=" * 70)

    os.environ["USE_REAL_TOOLS"] = "false"
    from atlas_brain.agents.graphs.email import run_email_workflow

    results = TestResults()

    # Test with "Test Client" which triggers mock context
    result = await run_email_workflow(
        "send estimate",
        client_name="Test Client",  # "test" triggers mock context
        service_date="February 15, 2026",
        service_time="3:00 PM",
        price="225.00",
        # Not providing: to_address, address, client_type
    )

    results.add(
        "Context extracted flag set",
        result.get("context_extracted") == True,
        f"context_extracted={result.get('context_extracted')}"
    )
    results.add(
        "Context source is booking",
        result.get("context_source") == "booking",
        f"context_source={result.get('context_source')}"
    )
    results.add(
        "Address auto-filled",
        result.get("address") == "123 Mock Street",
        f"address={result.get('address')}"
    )
    results.add(
        "Email auto-filled",
        result.get("draft_to") == "test@example.com" or result.get("to_address") == "test@example.com",
        f"draft_to={result.get('draft_to')}, to_address={result.get('to_address')}"
    )
    results.add(
        "Client type auto-filled",
        result.get("client_type") == "residential",
        f"client_type={result.get('client_type')}"
    )
    results.add(
        "Response shows auto-fill notice",
        "[Auto-filled" in result.get("response", ""),
        "Response missing auto-fill notice"
    )

    # Test without context (should not extract)
    result = await run_email_workflow(
        "send estimate",
        client_name="Unknown Person",  # Does not trigger mock
        to_address="manual@example.com",
        address="Manual Address",
        client_type="business",
        service_date="February 20, 2026",
        service_time="10:00 AM",
        price="300.00",
    )

    results.add(
        "No context for unknown client",
        result.get("context_extracted") != True,
        f"context_extracted={result.get('context_extracted')}"
    )

    return results.summary()


async def test_real_email_estimate():
    """Test 7: Real Email - Estimate."""
    print("\n" + "=" * 70)
    print("TEST 7: Real Email - Estimate (canfieldjuan24@gmail.com)")
    print("=" * 70)

    os.environ["USE_REAL_TOOLS"] = "true"

    # Re-import to pick up new env var
    import importlib
    import atlas_brain.agents.graphs.email as email_module
    importlib.reload(email_module)
    from atlas_brain.agents.graphs.email import run_email_workflow, send_email_confirmed

    results = TestResults()

    # Generate draft
    result = await run_email_workflow(
        "send estimate",
        to_address="canfieldjuan24@gmail.com",
        client_name="Phase 6 Test - Estimate",
        client_type="residential",
        address="123 Phase 6 Test Street",
        service_date="February 5, 2026",
        service_time="10:00 AM",
        price="199.00",
    )

    results.add(
        "Draft generated",
        result.get("awaiting_confirmation") == True,
        f"awaiting_confirmation={result.get('awaiting_confirmation')}"
    )

    if result.get("awaiting_confirmation"):
        # Send it
        send_result = await send_email_confirmed(result)

        results.add(
            "Email sent successfully",
            send_result.get("email_sent") == True,
            f"error={send_result.get('error')}"
        )
        results.add(
            "Has Resend message ID",
            bool(send_result.get("resend_message_id")) and "mock" not in send_result.get("resend_message_id", ""),
            f"resend_message_id={send_result.get('resend_message_id')}"
        )
        results.add(
            "Template is residential",
            send_result.get("template_used") == "residential",
            f"template_used={send_result.get('template_used')}"
        )

        if send_result.get("email_sent"):
            print(f"\n  >>> REAL EMAIL SENT: Message ID = {send_result.get('resend_message_id')}")

    return results.summary()


async def test_real_email_proposal():
    """Test 8: Real Email - Proposal."""
    print("\n" + "=" * 70)
    print("TEST 8: Real Email - Proposal (canfieldjuan24@gmail.com)")
    print("=" * 70)

    os.environ["USE_REAL_TOOLS"] = "true"

    # Check if database was initialized (done in main before real tests)
    db_available = _db_initialized

    import importlib
    import atlas_brain.agents.graphs.email as email_module
    importlib.reload(email_module)
    from atlas_brain.agents.graphs.email import run_email_workflow, send_email_confirmed

    results = TestResults()

    # Generate draft
    result = await run_email_workflow(
        "send proposal",
        to_address="canfieldjuan24@gmail.com",
        client_name="Phase 6 Test Corp",
        client_type="business",
        contact_name="Test Contact",
        contact_phone="555-TEST",
        address="456 Phase 6 Business Blvd",
        areas_to_clean="Main Office, Conference Rooms, Restrooms",
        cleaning_description="Daily cleaning: vacuuming, dusting, trash removal, restroom sanitation",
        price="450.00",
        frequency="Weekly",
    )

    results.add(
        "Proposal draft generated",
        result.get("awaiting_confirmation") == True,
        f"awaiting_confirmation={result.get('awaiting_confirmation')}"
    )

    if result.get("awaiting_confirmation"):
        send_result = await send_email_confirmed(result)

        results.add(
            "Proposal sent successfully",
            send_result.get("email_sent") == True,
            f"error={send_result.get('error')}"
        )
        results.add(
            "Has Resend message ID",
            bool(send_result.get("resend_message_id")) and "mock" not in send_result.get("resend_message_id", ""),
            f"resend_message_id={send_result.get('resend_message_id')}"
        )
        results.add(
            "Template is business",
            send_result.get("template_used") == "business",
            f"template_used={send_result.get('template_used')}"
        )
        # Follow-up requires database connection
        if send_result.get("follow_up_created"):
            results.add(
                "Follow-up auto-created",
                True,
                ""
            )
        elif db_available:
            # DB was available but follow-up still failed - this is a real failure
            results.add(
                "Follow-up auto-created",
                False,
                f"DB available but follow_up_created={send_result.get('follow_up_created')}"
            )
        else:
            results.skip(
                "Follow-up auto-created",
                "Database not available"
            )

        if send_result.get("email_sent"):
            print(f"\n  >>> REAL EMAIL SENT: Message ID = {send_result.get('resend_message_id')}")

    return results.summary()


async def test_real_email_generic():
    """Test 9: Real Email - Generic."""
    print("\n" + "=" * 70)
    print("TEST 9: Real Email - Generic (canfieldjuan24@gmail.com)")
    print("=" * 70)

    os.environ["USE_REAL_TOOLS"] = "true"

    import importlib
    import atlas_brain.agents.graphs.email as email_module
    importlib.reload(email_module)
    from atlas_brain.agents.graphs.email import run_email_workflow, send_email_confirmed

    results = TestResults()

    # Generate draft
    result = await run_email_workflow(
        "send email",
        to_address="canfieldjuan24@gmail.com",
        subject="Phase 6 Test - Generic Email",
        body="""Hello,

This is a test email from the Atlas Email Workflow Phase 6 validation.

Testing:
- Generic email sending
- Draft preview flow
- Real Resend API integration

Best regards,
Atlas Email Workflow Test Suite
""",
    )

    results.add(
        "Generic draft generated",
        result.get("awaiting_confirmation") == True,
        f"awaiting_confirmation={result.get('awaiting_confirmation')}"
    )

    if result.get("awaiting_confirmation"):
        send_result = await send_email_confirmed(result)

        results.add(
            "Generic email sent",
            send_result.get("email_sent") == True,
            f"error={send_result.get('error')}"
        )
        results.add(
            "Has Resend message ID",
            bool(send_result.get("resend_message_id")) and "mock" not in send_result.get("resend_message_id", ""),
            f"resend_message_id={send_result.get('resend_message_id')}"
        )
        results.add(
            "Template is generic",
            send_result.get("template_used") == "generic",
            f"template_used={send_result.get('template_used')}"
        )

        if send_result.get("email_sent"):
            print(f"\n  >>> REAL EMAIL SENT: Message ID = {send_result.get('resend_message_id')}")

    return results.summary()


async def main():
    """Run all Phase 6 tests."""
    print("=" * 70)
    print("PHASE 6: EMAIL WORKFLOW TESTING & VALIDATION")
    print("=" * 70)

    all_passed = True

    # Mock tests (no real emails sent)
    print("\n>>> MOCK TESTS (no real emails)")
    all_passed &= await test_intent_classification()
    all_passed &= await test_draft_preview_flow()
    all_passed &= await test_generic_email()
    all_passed &= await test_email_history()
    all_passed &= await test_follow_up_reminders()
    all_passed &= await test_context_extraction()

    # Real email tests (with delays to avoid rate limiting)
    print("\n>>> REAL EMAIL TESTS (sending to canfieldjuan24@gmail.com)")

    # Initialize database for email history and follow-up reminders
    db_ok = await initialize_database()
    if not db_ok:
        print("  [WARN] Database not available - email history won't be saved")

    all_passed &= await test_real_email_estimate()
    print("  (waiting 2s to avoid rate limit...)")
    await asyncio.sleep(2)
    all_passed &= await test_real_email_proposal()
    print("  (waiting 2s to avoid rate limit...)")
    await asyncio.sleep(2)
    all_passed &= await test_real_email_generic()

    print("\n" + "=" * 70)
    if all_passed:
        print("PHASE 6 VALIDATION: ALL TESTS PASSED")
    else:
        print("PHASE 6 VALIDATION: SOME TESTS FAILED")
    print("=" * 70)

    print("\n>>> Check canfieldjuan24@gmail.com for 3 test emails:")
    print("    1. Estimate email (residential)")
    print("    2. Proposal email (business)")
    print("    3. Generic email")

    # Cleanup
    await shutdown_database()

    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
