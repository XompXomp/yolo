# YOLO Instance Manager - Project Overview

This codebase is a macOS-optimized system for managing multiple parallel YOLOv8 inference instances. It consists of an Electron-based manager, a FastAPI-powered YOLO backend, and a stress-testing client.

## Components

### 1. Electron Manager (`main.js`, `renderer.js`, `index.html`)
The Electron app serves as the control plane for the entire system.
- **Instance Management**: Spawns and kills `yolo.py` processes as subprocesses.
- **Overseer (Load Balancer)**: Runs a built-in Express proxy on port `9000`. It receives all incoming inference requests and distributes them using a round-robin strategy across all active instances running the requested model.
- **UI**: Provides a dashboard to monitor logs, ports, and status for each running YOLO instance.

### 2. YOLO FastAPI Backend (`yolo/yolo.py`)
Each worker instance runs this script. It is a lightweight REST API for real-time object detection.
- **API**: Exposes a `POST /detect` endpoint that accepts images via `multipart/form-data`.
- **Inference**: Uses the `ultralytics` YOLOv8 library.
- **Resize Logic**: Can be configured to resize incoming images to a target dimension (e.g., 320x320) for consistent inference performance, while automatically scaling the resulting bounding boxes back to the original image dimensions.
- **Device Support**: Hardware accelerated on Mac using Apple Silicon (MPS).

### 3. Test Client (`test/test_client2.py`)
A utility for validating and stress-testing the load balancer.
- **Continuous Logic**: Sends a local image (e.g., `test_image.jpg`) to the Overseer (`localhost:9000`) in a continuous loop.
- **FPS Control**: Configured to run at a target frame rate (default 25 FPS).
- **Detailed Logging**: Prints frame-by-frame detection counts and bounding box coordinates for manual verification of the end-to-end pipeline.

---

## Getting Started
1. **Dependencies**: Run `npm install` for Electron and `pip install -r yolo/requirements.txt` for the Python backend.
2. **Launch**: Use `npm start` to open the manager.
3. **Inference**: Once instances are started in the UI, point your client at `http://localhost:9000/detect?model=models/yolov8n.pt`.
