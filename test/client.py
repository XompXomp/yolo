import asyncio
import httpx
import time
import os
import sys
#This is the test client which spins up 10 workers and keeps sending requests to verify FPS
# Configuration
URL = "http://localhost:9000/detect?model=models/yolov8n.pt"
IMAGE_PATH = "test_image.jpg"
CONCURRENCY = 10  # Number of parallel requests to maintain

async def send_request(client, image_data, frame_id):
    """Sends a single inference request and returns the latency."""
    files = {"file": ("test_image.jpg", image_data, "image/jpeg")}
    start_time = time.time()
    try:
        response = await client.post(URL, files=files, timeout=5.0)
        latency = time.time() - start_time
        if response.status_code == 200:
            return True, latency
        else:
            print(f"Frame {frame_id} failed with status {response.status_code}")
            return False, latency
    except Exception as e:
        print(f"Frame {frame_id} request error: {e}")
        return False, time.time() - start_time

async def worker(client, image_data, stats):
    """Worker that keeps sending requests as fast as possible."""
    while stats['running']:
        frame_id = stats['sent']
        stats['sent'] += 1
        success, latency = await send_request(client, image_data, frame_id)
        if success:
            stats['success'] += 1
            stats['latencies'].append(latency)

async def monitor(stats):
    """Prints FPS and latency metrics every second."""
    print(f"\n{'Time':<10} | {'Total Frames':<15} | {'Current FPS':<12} | {'Avg Latency':<12}")
    print("-" * 60)
    
    start_time = time.time()
    last_count = 0
    last_time = start_time
    
    while stats['running']:
        await asyncio.sleep(1.0)
        now = time.time()
        current_count = stats['success']
        
        # Calculate interval metrics
        interval_frames = current_count - last_count
        interval_time = now - last_time
        fps = interval_frames / interval_time if interval_time > 0 else 0
        
        # Calculate average latency
        avg_latency = sum(stats['latencies']) / len(stats['latencies']) if stats['latencies'] else 0
        stats['latencies'] = [] # Clear for next interval
        
        elapsed = now - start_time
        print(f"{elapsed:>8.1f}s | {current_count:>15} | {fps:>12.2f} | {avg_latency*1000:>10.2f}ms")
        
        last_count = current_count
        last_time = now

async def main():
    global IMAGE_PATH
    if not os.path.exists(IMAGE_PATH):
        # Attempt to find it in the same directory as the script if relative fails
        base_dir = os.path.dirname(os.path.abspath(__file__))
        potential_path = os.path.join(base_dir, "test_image.jpg")
        if os.path.exists(potential_path):
            IMAGE_PATH = potential_path
        else:
            print(f"Error: {IMAGE_PATH} not found. Please ensure test_image.jpg exists.")
            sys.exit(1)

    with open(IMAGE_PATH, "rb") as f:
        image_data = f.read()

    print(f"Targeting Overseer at: {URL}")
    print(f"Parallel Workers: {CONCURRENCY}")
    print("Press Ctrl+C to stop.\n")

    stats = {
        'sent': 0,
        'success': 0,
        'latencies': [],
        'running': True
    }

    async with httpx.AsyncClient() as client:
        # Start the monitor
        monitor_task = asyncio.create_task(monitor(stats))
        
        # Start multiple workers to maintain concurrency
        workers = [asyncio.create_task(worker(client, image_data, stats)) for _ in range(CONCURRENCY)]
        
        try:
            # Keep running until interrupted
            await asyncio.gather(*workers)
        except KeyboardInterrupt:
            pass
        finally:
            stats['running'] = False
            monitor_task.cancel()
            total_time = time.time() - stats.get('start_time', time.time()) # Not quite right but placeholder
            print(f"\nFinal Stats: Processed {stats['success']} frames successfully.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped.")