"""Test email workflow with mock and real tools."""

import asyncio
import os
import sys

sys.path.insert(0, "/home/juan-canfield/Desktop/Atlas-LangGraph-Agents-ToolUse")

# Load environment variables from Atlas project .env file
from dotenv import load_dotenv
load_dotenv("/home/juan-canfield/Desktop/Atlas/.env")


async def test_mock():
    """Test with mock tools."""
    os.environ["USE_REAL_TOOLS"] = "false"

    from atlas_brain.agents.graphs.email import run_email_workflow, send_email_confirmed

    print("=" * 70)
    print("EMAIL WORKFLOW TEST (Mock Tools)")
    print("=" * 70)

    # Test 1: Intent classification
    test_cases = [
        ("send email to test@example.com", "send_email"),
        ("email test@example.com", "send_email"),
        ("send estimate to client", "send_estimate"),
        ("send the estimate email", "send_estimate"),
        ("send proposal to business", "send_proposal"),
        ("email the proposal", "send_proposal"),
        ("what emails did I send", "query_history"),
    ]

    passed = 0
    failed = 0

    for text, expected_intent in test_cases:
        result = await run_email_workflow(text)
        actual_intent = result.get("intent")
        status = "PASS" if actual_intent == expected_intent else "FAIL"

        if status == "PASS":
            passed += 1
        else:
            failed += 1

        print(f"\n[{status}] '{text}'")
        print(f"       Expected: {expected_intent}")
        print(f"       Got:      {actual_intent}")

    print(f"\n\nIntent Classification: {passed}/{len(test_cases)} passed")
    print("-" * 50)

    # Test 2: Estimate draft generation
    print("\n\nTest: Estimate Draft Generation")
    print("-" * 50)

    result = await run_email_workflow(
        "send estimate",
        to_address="canfieldjuan24@gmail.com",
        client_name="Test Client",
        client_type="residential",
        address="123 Main St",
        service_date="February 1, 2026",
        service_time="9:00 AM",
        price="150.00",
    )

    print(f"Intent: {result.get('intent')}")
    print(f"Awaiting Confirmation: {result.get('awaiting_confirmation')}")
    print(f"Draft To: {result.get('draft_to')}")
    print(f"Draft Subject: {result.get('draft_subject')}")
    print(f"\nResponse:\n{result.get('response')}")

    # Test 3: Proposal draft generation
    print("\n\nTest: Proposal Draft Generation")
    print("-" * 50)

    result = await run_email_workflow(
        "send proposal",
        to_address="canfieldjuan24@gmail.com",
        client_name="Acme Corp",
        client_type="business",
        contact_name="John Smith",
        contact_phone="555-1234",
        address="456 Business Ave",
        areas_to_clean="Offices, Bathrooms, Break Room",
        cleaning_description="Dust surfaces, vacuum floors, empty trash, sanitize bathrooms",
        price="250.00",
        frequency="Weekly",
    )

    print(f"Intent: {result.get('intent')}")
    print(f"Awaiting Confirmation: {result.get('awaiting_confirmation')}")
    print(f"Draft To: {result.get('draft_to')}")
    print(f"Draft Subject: {result.get('draft_subject')}")
    print(f"\nResponse:\n{result.get('response')}")

    # Test 4: Query email history
    print("\n\nTest: Query Email History")
    print("-" * 50)

    result = await run_email_workflow("what emails did I send today")

    print(f"Intent: {result.get('intent')}")
    print(f"History Queried: {result.get('history_queried')}")
    print(f"Email Count: {result.get('history_count')}")
    print(f"\nResponse:\n{result.get('response')}")

    # Test 5: Follow-up reminder with estimate
    print("\n\nTest: Estimate with Follow-up Reminder")
    print("-" * 50)

    result = await run_email_workflow(
        "send estimate",
        to_address="canfieldjuan24@gmail.com",
        client_name="Follow-up Test Client",
        client_type="residential",
        address="789 Test Lane",
        service_date="February 10, 2026",
        service_time="2:00 PM",
        price="200.00",
        create_follow_up=True,
        follow_up_days=3,
    )

    print(f"Intent: {result.get('intent')}")
    print(f"Awaiting Confirmation: {result.get('awaiting_confirmation')}")
    print(f"Follow-up will be created: {result.get('create_follow_up', 'N/A')}")

    # Simulate confirming the send
    if result.get("awaiting_confirmation"):
        print("\nSimulating send confirmation...")
        send_result = await send_email_confirmed(result)
        print(f"Email Sent: {send_result.get('email_sent')}")
        print(f"Follow-up Created: {send_result.get('follow_up_created')}")
        print(f"Follow-up Reminder ID: {send_result.get('follow_up_reminder_id')}")
        print(f"\nResponse:\n{send_result.get('response')}")

    # Test 6: Context extraction (auto-fill from booking)
    print("\n\nTest: Context Extraction (Auto-fill)")
    print("-" * 50)

    # Test with a name that triggers mock context lookup
    result = await run_email_workflow(
        "send estimate",
        client_name="Test Client",  # "test" triggers mock context
        service_date="February 15, 2026",
        service_time="3:00 PM",
        price="225.00",
    )

    print(f"Intent: {result.get('intent')}")
    print(f"Context Extracted: {result.get('context_extracted')}")
    print(f"Context Source: {result.get('context_source')}")
    print(f"Auto-filled Address: {result.get('address')}")
    print(f"Auto-filled Email: {result.get('to_address') or result.get('draft_to')}")
    print(f"Auto-filled Client Type: {result.get('client_type')}")
    print(f"\nResponse:\n{result.get('response')}")

    print("\n" + "=" * 70)
    print("MOCK TESTS COMPLETE")
    print("=" * 70)


async def test_real_email():
    """Test with real email sending."""
    os.environ["USE_REAL_TOOLS"] = "true"

    from atlas_brain.agents.graphs.email import run_email_workflow, send_email_confirmed

    print("\n" + "=" * 70)
    print("EMAIL WORKFLOW TEST (REAL EMAIL)")
    print("=" * 70)

    # Test: Send real estimate email
    print("\nTest: Send Real Estimate Email to canfieldjuan24@gmail.com")
    print("-" * 50)

    # First generate draft
    result = await run_email_workflow(
        "send estimate",
        to_address="canfieldjuan24@gmail.com",
        client_name="Atlas Test User",
        client_type="residential",
        address="123 Test Street, Effingham IL",
        service_date="February 5, 2026",
        service_time="10:00 AM",
        price="175.00",
    )

    print(f"Draft generated:")
    print(f"  To: {result.get('draft_to')}")
    print(f"  Subject: {result.get('draft_subject')}")
    print(f"  Template: {result.get('draft_template')}")

    if result.get("awaiting_confirmation"):
        print("\nSending confirmed email...")

        # Now send it
        send_result = await send_email_confirmed(result)

        print(f"\nSend Result:")
        print(f"  Sent: {send_result.get('email_sent')}")
        print(f"  Message ID: {send_result.get('resend_message_id')}")
        print(f"  Template: {send_result.get('template_used')}")
        print(f"  Error: {send_result.get('error')}")
        print(f"\nResponse: {send_result.get('response')}")

    print("\n" + "=" * 70)
    print("CHECK YOUR EMAIL: canfieldjuan24@gmail.com")
    print("=" * 70)


async def main():
    # First run mock tests
    await test_mock()

    # Ask before sending real email
    print("\n\nReady to send REAL email to canfieldjuan24@gmail.com")
    print("Running real email test...")

    await test_real_email()


if __name__ == "__main__":
    asyncio.run(main())
