import json
import os
import time
from typing import Any

from kafka import KafkaConsumer

KAFKA_BROKER = os.getenv("ATLAS_VIDEO_KAFKA_BROKER", "localhost:9092")
KAFKA_TOPIC = os.getenv("ATLAS_VIDEO_KAFKA_TOPIC", "drone_video_stream")
PROCESSOR_MODE = os.getenv("ATLAS_VIDEO_PROCESSOR_MODE", "demo").strip().lower()


def process_frame_demo(frame_data: dict[str, Any]) -> dict[str, Any]:
    """Demo frame processing path used for local pipeline smoke tests."""
    drone_id = frame_data.get("drone_id", "unknown")
    print(f"  [Video Processor]: Demo processing frame from drone {drone_id}...")
    time.sleep(0.05)
    print(f"  [Video Processor]: Demo processing complete for drone {drone_id}.")
    return {
        "mode": "demo",
        "drone_id": drone_id,
        "timestamp": frame_data.get("timestamp"),
        "status": "processed",
    }


def process_video_stream() -> None:
    """Consume video stream data from Kafka using the demo processing path."""
    if PROCESSOR_MODE != "demo":
        raise RuntimeError(
            "video_stream_processor.py only supports ATLAS_VIDEO_PROCESSOR_MODE=demo. "
            "Production decode/detection must be implemented in a dedicated runtime path."
        )

    consumer = None
    max_retries = 5
    for i in range(max_retries):
        try:
            print(
                f"Video Processor: connecting to Kafka {KAFKA_BROKER} "
                f"(attempt {i + 1}/{max_retries})..."
            )
            consumer = KafkaConsumer(
                KAFKA_TOPIC,
                bootstrap_servers=[KAFKA_BROKER],
                auto_offset_reset="earliest",
                enable_auto_commit=True,
                group_id="video-processor-group",
                value_deserializer=lambda x: json.loads(x.decode("utf-8")),
                api_version=(0, 10, 1),
            )
            print(
                f"Video Processor: connected to {KAFKA_BROKER}. "
                f"Listening on topic {KAFKA_TOPIC} in demo mode."
            )
            break
        except Exception as e:
            print(f"Video Processor: Could not connect to Kafka: {e}")
            time.sleep(2 ** i)

    if not consumer:
        print(f"Video Processor: Failed to connect to Kafka after {max_retries} attempts. Exiting.")
        return

    try:
        for message in consumer:
            frame_data = message.value
            print(
                "Video Processor: received frame "
                f"drone={frame_data.get('drone_id')} "
                f"timestamp={frame_data.get('timestamp')}"
            )

            processing_result = process_frame_demo(frame_data)
            print(f"Video Processor: Demo result: {processing_result}")

    except KeyboardInterrupt:
        print("Video Processor: Shutting down.")
    except Exception as e:
        print(f"Video Processor: An error occurred: {e}")
    finally:
        if consumer:
            consumer.close()


if __name__ == "__main__":
    process_video_stream()
