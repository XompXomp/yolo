import requests
import os
import sys
import time

def main():
    url = "http://localhost:9002/detect"
    image_path = "test_image.jpg"
    fps = 25
    frame_duration = 1.0 / fps

    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found.")
        sys.exit(1)

    # Read image into memory once
    with open(image_path, "rb") as f:
        image_data = f.read()

    print(f"Starting loop: Sending {image_path} to {url} at {fps} FPS...")
    print("Press Ctrl+C to stop.")

    frame_count = 0
    start_time = time.time()

    try:
        while True:
            batch_start = time.time()
            
            files = {"file": ("test_image.jpg", image_data, "image/jpeg")}
            try:
                response = requests.post(url, files=files, timeout=1.0)
                
                if response.status_code == 200:
                    data = response.json()
                    detections = data.get("detections", [])
                    bboxes = [det["bbox"] for det in detections]
                    print(f"frame {frame_count}: {len(detections)} detections, bounding box {bboxes}")
                else:
                    print(f"\nError: Status {response.status_code}")
            
            except requests.exceptions.RequestException as e:
                print(f"\nConnection error: {e}")
                time.sleep(1) # Wait a bit before retrying

            frame_count += 1
            
            # Maintain FPS
            elapsed = time.time() - batch_start
            sleep_time = max(0, frame_duration - elapsed)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        total_time = time.time() - start_time
        actual_fps = frame_count / total_time if total_time > 0 else 0
        print(f"\nStopped. Sent {frame_count} frames in {total_time:.2f}s (Avg FPS: {actual_fps:.2f})")

if __name__ == "__main__":
    main()
