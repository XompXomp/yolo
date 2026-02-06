import sys
import json
import asyncio
import websockets
import requests
import time

def start_stream(api_base_url, stream_url):
    """
    Tells the YOLO instance to start processing the RTSP stream.
    """
    endpoint = f"{api_base_url}/stream/start"
    payload = {
        "stream_url": stream_url,
        "conf": 0.25,
        "iou": 0.45
    }
    
    print(f"--- Sending Start Request to {endpoint} ---")
    try:
        response = requests.post(endpoint, json=payload, timeout=5)
        if response.status_code == 200:
            print(f"Success: {response.json()}")
            return True
        else:
            print(f"Failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"Error connecting to API: {e}")
        return False

async def listen_for_detections(ws_url):
    """
    Connects to the WebSocket and prints detection data in real-time.
    """
    print(f"--- Connecting to WebSocket: {ws_url} ---")
    try:
        async with websockets.connect(ws_url) as websocket:
            print("Connected! Listening for detection frames...")
            while True:
                message = await websocket.recv()
                data = json.loads(message)
                
                frame_id = data.get("frame_id")
                detections = data.get("detections", [])
                
                if detections:
                    print(f"\n[Frame {frame_id}] Time: {time.strftime('%H:%M:%S')}")
                    for d in detections:
                        cls = d.get('class')
                        conf = d.get('confidence')
                        bbox = d.get('bbox')
                        print(f"  > {cls.upper()}: {conf*100:.1f}% at {bbox}")
                else:
                    # Print a dot just to show we are receiving heartbeat frames
                    print(".", end="", flush=True)
                    
    except KeyboardInterrupt:
        print("\nDisconnected by user.")
    except Exception as e:
        print(f"\nWebSocket error: {e}")

def stop_stream(api_base_url):
    """
    Tells the YOLO instance to stop processing.
    """
    print(f"\n--- Sending Stop Request to {api_base_url}/stream/stop ---")
    try:
        requests.post(f"{api_base_url}/stream/stop")
    except:
        pass

if __name__ == "__main__":
    # Usage: python test_client.py 192.168.1.XX 9002 rtsp://your-camera-url
    if len(sys.argv) < 3:
        print("Usage: python test_client.py <IP_ADDRESS> <PORT> [RTSP_URL]")
        print("Example: python test_client.py 127.0.0.1 9002 rtsp://localhost:8554/test")
        sys.exit(1)

    ip = sys.argv[1]
    port = sys.argv[2]
    # Use a placeholder if no RTSP URL is provided
    rtsp_url = sys.argv[3] if len(sys.argv) > 3 else "rtsp://127.0.0.1:8554/live"

    base_url = f"http://{ip}:{port}"
    ws_url = f"ws://{ip}:{port}/detections"

    # 1. Start the stream
    if start_stream(base_url, rtsp_url):
        # 2. Listen to detections via WebSocket
        try:
            asyncio.run(listen_for_detections(ws_url))
        except KeyboardInterrupt:
            pass
        finally:
            # 3. Cleanup on exit
            stop_stream(base_url)