import requests
import os
import sys
import time
import cv2
import numpy as np

def draw_detections(image, detections):
    for det in detections:
        bbox = det["bbox"]  # [x, y, w, h] center-based
        class_name = det["class"]
        confidence = det["confidence"]

        # Convert xywh center to xyxy (top-left, bottom-right)
        x_center, y_center, w, h = bbox
        x1 = int(x_center - w/2)
        y1 = int(y_center - h/2)
        x2 = int(x1 + w)
        y2 = int(y1 + h)

        # Colors and labels
        color = (0, 255, 0) # Green
        label = f"{class_name} {confidence:.2f}"
        
        # Draw box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Draw label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1 - th - 5), (x1 + tw, y1), color, -1)
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

def main():
    url = "http://localhost:9000/detect?model=models/yolov8n.pt"
    image_path = "test_image.jpg"
    fps = 25
    frame_duration = 1.0 / fps

    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found.")
        sys.exit(1)

    # Read image for sending
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    # Create base image for display
    nparr = np.frombuffer(image_bytes, np.uint8)
    base_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if base_image is None:
        print("Error: Could not decode image for display.")
        sys.exit(1)

    window_name = "YOLO Visual Verification"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    print(f"Starting loop: Sending {image_path} to {url} at {fps} FPS...")
    print("Visual window opened. Press 'q' or Ctrl+C to stop.")

    frame_count = 0
    start_time = time.time()

    try:
        while True:
            batch_start = time.time()
            
            # Reset display frame for each iteration
            display_frame = base_image.copy()
            
            files = {"file": ("test_image.jpg", image_bytes, "image/jpeg")}
            try:
                response = requests.post(url, files=files, timeout=2.0)
                
                if response.status_code == 200:
                    data = response.json()
                    detections = data.get("detections", [])
                    
                    # Draw boxes on the current copy
                    draw_detections(display_frame, detections)
                    
                    # Update window
                    cv2.imshow(window_name, display_frame)
                    
                    # Small wait to allow UI to render (handles 'q' to quit)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                        
                    print(f"frame {frame_count:04d}: {len(detections)} detections")
                else:
                    print(f"\nError: Status {response.status_code} - {response.text}")
            
            except requests.exceptions.RequestException as e:
                print(f"\nConnection error: {e}")
                time.sleep(1)

            frame_count += 1
            
            # Maintain FPS
            elapsed = time.time() - batch_start
            sleep_time = max(0.001, frame_duration - elapsed)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        pass
    finally:
        total_time = time.time() - start_time
        actual_fps = frame_count / total_time if total_time > 0 else 0
        print(f"\nStopped. Sent {frame_count} frames in {total_time:.2f}s (Avg FPS: {actual_fps:.2f})")
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
