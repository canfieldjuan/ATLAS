"""Test reminder workflow with proper ntfy setup."""

import asyncio
import os
import sys

os.environ["USE_REAL_TOOLS"] = "true"
sys.path.insert(0, "/home/juan-canfield/Desktop/Atlas-LangGraph-Agents-ToolUse")

from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path("/home/juan-canfield/Desktop/Atlas/.env"), override=True)


async def setup_ntfy_alerts():
    """Register NtfyDelivery with the alert manager (like main.py does)."""
    from atlas_brain.config import settings
    from atlas_brain.alerts import get_alert_manager, NtfyDelivery, setup_default_callbacks

    if not settings.alerts.enabled:
        print("  [WARN] Alerts disabled in config")
        return False

    alert_manager = get_alert_manager()
    setup_default_callbacks(alert_manager)

    if settings.alerts.ntfy_enabled:
        ntfy_delivery = NtfyDelivery(
            base_url=settings.alerts.ntfy_url,
            topic=settings.alerts.ntfy_topic,
        )
        alert_manager.register_callback(ntfy_delivery.deliver)
        print(f"  [OK] ntfy registered: {settings.alerts.ntfy_url}/{settings.alerts.ntfy_topic}")
        return True
    else:
        print("  [WARN] ntfy_enabled=False in config")
        return False


async def main():
    from atlas_brain.storage.database import init_database, close_database
    await init_database()

    # Setup ntfy BEFORE initializing reminder service
    print("Setting up ntfy alerts...")
    ntfy_ok = await setup_ntfy_alerts()
    if not ntfy_ok:
        print("ntfy not configured properly, exiting")
        return

    from atlas_brain.services.reminders import initialize_reminder_service, shutdown_reminder_service
    await initialize_reminder_service()

    from atlas_brain.agents.graphs.reminder import run_reminder_workflow

    print("\n" + "=" * 70)
    print("REMINDER TEST WITH NTFY PROPERLY CONFIGURED")
    print("=" * 70)

    # Test 1: Immediate reminder
    print("\n[1/2] Sending reminder NOW...")
    result = await run_reminder_workflow("remind me now to test ntfy delivery")
    print(f"  Response: {result.get('response')}")
    print(f"  Created: {result.get('reminder_created')}")
    print("  >> Check ntfy NOW - notification #1 should appear")

    # Wait for delivery
    await asyncio.sleep(5)

    # Test 2: Second reminder after short gap
    print("\n[2/2] Sending second reminder...")
    result = await run_reminder_workflow("send me a reminder")
    print(f"  Response: {result.get('response')}")
    print(f"  Created: {result.get('reminder_created')}")
    print("  >> Check ntfy NOW - notification #2 should appear")

    await asyncio.sleep(5)

    print("\n" + "=" * 70)
    print("TEST COMPLETE - Check ntfy for 2 notifications")
    print("=" * 70)

    await shutdown_reminder_service()
    await close_database()


if __name__ == "__main__":
    asyncio.run(main())
