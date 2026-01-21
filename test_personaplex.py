#!/usr/bin/env python3
"""
Test PersonaPlex integration with Atlas.

This script tests the connection and conversation flow without needing
an actual phone call. It connects to PersonaPlex with the business context
and lets you type messages to simulate a customer.
"""

import asyncio
import os
import sys

# Configure for local server without SSL
os.environ["ATLAS_PERSONAPLEX_USE_SSL"] = "false"
os.environ["ATLAS_PERSONAPLEX_HOST"] = "localhost"
os.environ["ATLAS_PERSONAPLEX_PORT"] = "8998"

# Clear cached config
from atlas_brain.services.personaplex.config import get_personaplex_config
get_personaplex_config.cache_clear()

from atlas_brain.comms.config import EFFINGHAM_MAIDS_CONTEXT
from atlas_brain.comms.personaplex_processor import (
    PersonaPlexProcessor,
    PersonaPlexCallState,
)


async def main():
    print("=" * 60)
    print("PersonaPlex Integration Test")
    print("=" * 60)
    print()
    print("Business: Effingham Office Maids")
    print("This simulates the receptionist handling a booking call.")
    print()

    # Create call state
    call_state = PersonaPlexCallState(
        call_sid="test-call-001",
        from_number="+15551234567",
        to_number="+16183683696",
        context_id="effingham_maids",
    )

    def on_audio(audio_b64: str):
        print(f"[Audio response: {len(audio_b64)} bytes]")

    # Create processor with business context
    processor = PersonaPlexProcessor(
        call_state=call_state,
        business_context=EFFINGHAM_MAIDS_CONTEXT,
        on_audio_ready=on_audio,
    )

    print("Connecting to PersonaPlex server (takes ~80s)...")
    connected = await processor.connect()

    if not connected:
        print("ERROR: Failed to connect to PersonaPlex")
        print("Make sure the server is running:")
        print("  cd /home/juan-canfield/Desktop/live-translator/personaplex")
        print("  python run_server.py --host 0.0.0.0 --port 8998")
        sys.exit(1)

    print("Connected!")
    print()
    print("PersonaPlex is now listening. Speak into your microphone")
    print("(if using web UI) or the model will speak first.")
    print()
    print("Press Ctrl+C to exit")
    print("-" * 60)

    try:
        # Keep running and print any extracted info
        while True:
            await asyncio.sleep(5)
            ctx = processor._tool_bridge.context
            if ctx.customer_name or ctx.customer_phone or ctx.preferred_date:
                print()
                print("Extracted booking info:")
                print(f"  Name: {ctx.customer_name}")
                print(f"  Phone: {ctx.customer_phone}")
                print(f"  Date: {ctx.preferred_date}")
                print(f"  Time: {ctx.preferred_time}")
                print()
    except KeyboardInterrupt:
        print()
        print("Disconnecting...")

    await processor.disconnect()
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
