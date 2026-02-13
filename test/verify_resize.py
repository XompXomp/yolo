import subprocess
import time
import requests
import os
import sys
import signal

def run_test():
    # Configuration
    port = 9005
    width = 320
    height = 320
    model_path = "../yolo/yolov8n.pt"  # Assuming model exists here or in models/
    script_path = "../yolo/yolo.py"
    image_path = "test_image.jpg"

    # Start yolo.py instance
    print(f"Starting yolo.py on port {port} with resize {width}x{height}...")
    process = subprocess.Popen(
        [sys.executable, script_path, "--port", str(port), "--width", str(width), "--height", str(height)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    try:
        # Wait for server to start
        time.sleep(5) 

        # Prepare request
        url = f"http://localhost:{port}/detect"
        if not os.path.exists(image_path):
            print(f"Error: {image_path} not found.")
            return

        print(f"Sending request to {url}...")
        with open(image_path, "rb") as f:
            files = {"file": f}
            response = requests.post(url, files=files)

        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print("Response JSON:")
            print(data)
            
            # Basic validation
            if "detections" in data:
                print("Success: Detections received.")
                # We can't easily verify the exact coordinate mapping without visual inspection or known ground truth,
                # but getting detections means the resize and inference pipeline worked without crashing.
            else:
                print("Failure: No detections key in response.")
        else:
            print("Failure: API returned non-200 status.")
            print(response.text)

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        # Cleanup
        print("Stopping yolo.py...")
        process.terminate()
        process.wait()

if __name__ == "__main__":
    run_test()
