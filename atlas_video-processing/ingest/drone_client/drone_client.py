from kafka import KafkaProducer
import time
import random
import json

KAFKA_BROKER = 'localhost:9092' # This will be the service name in docker-compose
KAFKA_TOPIC = 'drone_video_stream'

def simulate_video_stream(drone_id):
    """
    Simulates sending video data from a drone to a Kafka topic.
    In a real scenario, this would capture frames and send them.
    """
    producer = None
    max_retries = 5
    for i in range(max_retries):
        try:
            print(f"Drone {drone_id}: Attempting to connect to Kafka broker at {KAFKA_BROKER} (attempt {i+1}/{max_retries})...")
            producer = KafkaProducer(bootstrap_servers=[KAFKA_BROKER],
                                     value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                                     api_version=(0, 10, 1)) # Specify API version for compatibility
            print(f"Drone {drone_id}: Connected to Kafka broker at {KAFKA_BROKER}.")
            break
        except Exception as e:
            print(f"Drone {drone_id}: Could not connect to Kafka: {e}")
            time.sleep(2 ** i) # Exponential backoff
    
    if not producer:
        print(f"Drone {drone_id}: Failed to connect to Kafka after {max_retries} attempts. Exiting.")
        return

    print(f"Drone {drone_id}: Starting video stream simulation to topic {KAFKA_TOPIC}...")
    try:
        while True:
            # Simulate capturing a video frame
            frame_data = {
                "drone_id": drone_id,
                "timestamp": time.time(),
                "simulated_data": random.randint(0, 100),
                "message": f"Frame from Drone {drone_id}"
            }
            
            print(f"Drone {drone_id}: Sending frame - {frame_data['message']} with data {frame_data['simulated_data']}")
            
            # Send data to Kafka
            producer.send(KAFKA_TOPIC, value=frame_data)
            producer.flush() # Ensure message is sent

            time.sleep(random.uniform(0.5, 2.0)) # Simulate varying frame rates
    except KeyboardInterrupt:
        print(f"Drone {drone_id}: Shutting down.")
    except Exception as e:
        print(f"Drone {drone_id}: An error occurred: {e}")
    finally:
        if producer:
            producer.close()

if __name__ == "__main__":
    drone_id = "Alpha-001" 
    simulate_video_stream(drone_id)