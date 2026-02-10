import sys
import json
import time

import requests


def send_image_request(api_base_url: str, image_path: str, model: str = "yolov8n.pt"):
    """
    Sends an image file to the YOLO instance using the OpenAI-style
    /v1/vision/detections:file endpoint and prints the response.
    """
    url = f"{api_base_url}/v1/vision/detections:file"
    files = {
        "file": open(image_path, "rb"),
    }
    data = {
        "model": model,
        "conf": 0.5,
        "iou": 0.45,
    }

    print(f"--- Sending image '{image_path}' to {url} ---")
    try:
        resp = requests.post(url, data=data, files=files, timeout=10)
    except Exception as e:
        print(f"Request error: {e}")
        return

    print(f"Status: {resp.status_code}")
    try:
        payload = resp.json()
        print(json.dumps(payload, indent=2))
    except Exception:
        print("Non-JSON response:")
        print(resp.text)


if __name__ == "__main__":
    # Usage: python test_client2.py <IP_ADDRESS> <PORT> [IMAGE_PATH]
    # Example: python test_client2.py 127.0.0.1 9000 test_video.jpg
    if len(sys.argv) < 3:
        print("Usage: python test_client2.py <IP_ADDRESS> <PORT> [IMAGE_PATH]")
        print("Example: python test_client2.py 127.0.0.1 9000 test_video.jpg")
        sys.exit(1)

    ip = sys.argv[1]
    port = sys.argv[2]
    image_path = sys.argv[3] if len(sys.argv) > 3 else "test_video.jpg"

    base_url = f"http://{ip}:{port}"
    send_image_request(base_url, image_path)

