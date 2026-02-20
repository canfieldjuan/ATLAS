#!/usr/bin/env python3
"""
Voice enrollment script.

Records microphone samples and enrolls via the Atlas speaker API.
Usage: python scripts/enroll_voice.py --name "Juan"
"""

import argparse
import base64
import json
import subprocess
import sys
import tempfile
import time
import urllib.request

API_BASE = "http://127.0.0.1:8000/api/v1/speaker"
SAMPLE_RATE = 16000
RECORD_SECONDS = 5
PROMPTS = [
    "Say something natural, like describing your day.",
    "Read aloud: 'Hey Atlas, what is the weather today?'",
    "Read aloud: 'Turn on the kitchen lights and set brightness to 80 percent.'",
    "Read aloud: 'Remind me to check my email tomorrow morning.'",
    "Read aloud: 'What time is my next appointment?'",
]


def api_post(endpoint, data):
    """POST JSON to Atlas API and return response dict."""
    url = f"{API_BASE}/{endpoint}"
    payload = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def api_get(endpoint):
    """GET from Atlas API and return response dict."""
    url = f"{API_BASE}/{endpoint}"
    with urllib.request.urlopen(url, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def record_sample(index, total):
    """Record a PCM sample from the microphone using arecord."""
    prompt = PROMPTS[index % len(PROMPTS)]
    print(f"\n--- Sample {index + 1}/{total} ---")
    print(f"  {prompt}")
    print(f"  Recording {RECORD_SECONDS}s in 3... ", end="", flush=True)
    time.sleep(1)
    print("2... ", end="", flush=True)
    time.sleep(1)
    print("1... ", end="", flush=True)
    time.sleep(1)
    print("GO!")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    cmd = [
        "arecord",
        "-D", "plughw:4,0",
        "-f", "S16_LE",
        "-r", str(SAMPLE_RATE),
        "-c", "1",
        "-d", str(RECORD_SECONDS),
        "-q",
        tmp_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: arecord failed: {result.stderr}")
        return None

    # Read raw PCM (skip 44-byte WAV header)
    with open(tmp_path, "rb") as f:
        wav_data = f.read()

    pcm_data = wav_data[44:]
    print(f"  Recorded {len(pcm_data)} bytes of PCM audio.")
    return pcm_data


def main():
    parser = argparse.ArgumentParser(description="Enroll voice with Atlas")
    parser.add_argument("--name", required=True, help="Your name (e.g., 'Juan')")
    parser.add_argument("--samples", type=int, default=3, help="Number of samples")
    args = parser.parse_args()

    # 1. Check service status
    print("Checking speaker ID service...")
    status = api_get("status")
    if not status.get("enabled"):
        print("ERROR: Speaker ID is disabled. Set ATLAS_SPEAKER_ID_ENABLED=true")
        sys.exit(1)
    print(f"  Service enabled, threshold={status['confidence_threshold']}")

    # 2. Start enrollment
    print(f"\nStarting enrollment for '{args.name}'...")
    start_resp = api_post("enroll/start", {"user_name": args.name})
    session_id = start_resp["session_id"]
    samples_needed = start_resp["samples_needed"]
    total_samples = max(args.samples, samples_needed)
    print(f"  Session: {session_id}")
    print(f"  User ID: {start_resp['user_id']}")
    print(f"  Samples needed: {samples_needed}")

    # 3. Record and submit samples
    for i in range(total_samples):
        pcm_data = record_sample(i, total_samples)
        if pcm_data is None:
            print("Skipping failed sample, trying again...")
            pcm_data = record_sample(i, total_samples)
            if pcm_data is None:
                print("ERROR: Could not record sample. Check microphone.")
                sys.exit(1)

        audio_b64 = base64.b64encode(pcm_data).decode("ascii")
        print("  Submitting to API...", end=" ", flush=True)
        sample_resp = api_post("enroll/sample", {
            "session_id": session_id,
            "audio_base64": audio_b64,
            "sample_rate": SAMPLE_RATE,
        })

        if sample_resp.get("error"):
            print(f"ERROR: {sample_resp['error']}")
            sys.exit(1)

        collected = sample_resp["samples_collected"]
        needed = sample_resp["samples_needed"]
        ready = sample_resp["is_ready"]
        print(f"OK ({collected}/{needed})")

        if ready and i < total_samples - 1:
            print("  (Enough samples collected, continuing for better quality)")

    # 4. Complete enrollment
    print("\nCompleting enrollment...")
    complete_resp = api_post("enroll/complete", {"session_id": session_id})

    if complete_resp.get("success"):
        print(f"\n  Enrollment successful!")
        print(f"  User: {complete_resp['user_name']}")
        print(f"  User ID: {complete_resp['user_id']}")
        print(f"  Samples used: {complete_resp['samples_used']}")
        print("\nAtlas will now recognize your voice and populate speaker_uuid")
        print("on future conversation turns. preference_learning will start")
        print("working within 7 days.")
    else:
        print(f"\n  Enrollment FAILED: {complete_resp.get('error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
