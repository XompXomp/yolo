import requests
import os
import sys

def main():
    url = "http://localhost:9002/detect"
    image_path = "test_image.jpg"

    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found.")
        sys.exit(1)

    print(f"Sending {image_path} to {url}...")

    try:
        with open(image_path, "rb") as f:
            files = {"file": f}
            response = requests.post(url, files=files)

        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            try:
                data = response.json()
                print("Response JSON:")
                print(data)
                
                if "detections" in data:
                    print("\nSuccess: Detections received.")
                    for det in data["detections"]:
                        print(f" - {det['class']} ({det['confidence']}): {det['bbox']}")
                else:
                     print("\nWarning: 'detections' key not found in response.")

            except ValueError:
                print("Error: Could not parse JSON response.")
                print(response.text)
        else:
            print(f"Error: Server returned status code {response.status_code}")
            print(response.text)

    except requests.exceptions.RequestException as e:
        print(f"Error connecting to server: {e}")

if __name__ == "__main__":
    main()
