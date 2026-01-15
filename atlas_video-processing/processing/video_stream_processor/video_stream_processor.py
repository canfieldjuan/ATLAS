from kafka import KafkaConsumer
import json
import time
import cv2 # Import OpenCV

KAFKA_BROKER = 'localhost:9092' # This will be the service name in docker-compose
KAFKA_TOPIC = 'drone_video_stream'

def process_frame_with_opencv(frame_data):
    """
    Placeholder function to simulate OpenCV processing on a video frame.
    In a real scenario, this would involve decoding an actual image/frame,
    applying computer vision algorithms (e.g., object detection), etc.
    """
    print(f"  [OpenCV Processor]: Simulating OpenCV processing for frame from Drone {frame_data['drone_id']}...")
    # Simulate some CPU-bound work
    time.sleep(0.05) 
    # In a real application, you would decode the frame data here, e.g.:
    # np_arr = np.frombuffer(base64.b64decode(frame_data['image_data']), np.uint8)
    # img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # ... then apply detection models
    print(f"  [OpenCV Processor]: Finished processing for frame from Drone {frame_data['drone_id']}.")
    return "Processed_Metadata_Example" # Return some simulated result

def process_video_stream():
    """
    Consumes video stream data from a Kafka topic and simulates processing.
    """
    consumer = None
    max_retries = 5
    for i in range(max_retries):
        try:
            print(f"Video Processor: Attempting to connect to Kafka broker at {KAFKA_BROKER} (attempt {i+1}/{max_retries})...")
            consumer = KafkaConsumer(
                KAFKA_TOPIC,
                bootstrap_servers=[KAFKA_BROKER],
                auto_offset_reset='earliest', # Start reading at the earliest message
                enable_auto_commit=True,
                group_id='video-processor-group', # Consumer group ID
                value_deserializer=lambda x: json.loads(x.decode('utf-8')),
                api_version=(0, 10, 1) # Specify API version for compatibility
            )
            print(f"Video Processor: Connected to Kafka broker at {KAFKA_BROKER}. Listening to topic {KAFKA_TOPIC}...")
            break
        except Exception as e:
            print(f"Video Processor: Could not connect to Kafka: {e}")
            time.sleep(2 ** i) # Exponential backoff
    
    if not consumer:
        print(f"Video Processor: Failed to connect to Kafka after {max_retries} attempts. Exiting.")
        return

    try:
        for message in consumer:
            frame_data = message.value
            print(f"Video Processor: Received frame from Drone {frame_data['drone_id']} at {frame_data['timestamp']} - simulated data: {frame_data['simulated_data']}")
            
            # Call the placeholder OpenCV processing function
            processing_result = process_frame_with_opencv(frame_data)
            
            print(f"Video Processor: Processed frame from Drone {frame_data['drone_id']}. Result: {processing_result}")

    except KeyboardInterrupt:
        print("Video Processor: Shutting down.")
    except Exception as e:
        print(f"Video Processor: An error occurred: {e}")
    finally:
        if consumer:
            consumer.close()

if __name__ == "__main__":
    process_video_stream()
