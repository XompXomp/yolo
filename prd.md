Product Requirements Document (PRD)
Project: YOLO Instance Manager (macOS)
1. Overview

The YOLO Instance Manager is a macOS desktop application that allows users to start, manage, and stop multiple independent YOLOv8 inference instances. Each instance runs as a separate Python process and exposes a network-accessible API that external applications on the same network can use to perform real-time object detection on video streams.

The application is intended as an internal developer tool for technically competent users and will run exclusively on company-managed macOS machines with Apple Silicon.

The desktop application will be built using Electron.js and will act purely as a process and lifecycle manager. All machine learning inference is handled by Python-based backend services using Ultralytics YOLOv8 with MPS acceleration.

2. Goals

Allow users to start multiple YOLOv8 inference instances from a macOS UI

Each instance:

Runs independently

Uses a specified pretrained YOLOv8 model

Processes exactly one video stream

Exposes a network API for external apps

Enable real-time, streaming detection results

Keep architecture simple, debuggable, and extensible

3. Non-Goals (Explicit)

No model training

No cloud deployment

No authentication or user management

No UI for drawing bounding boxes or visualizing video

No multi-stream handling inside a single YOLO instance

No Windows or Linux support

4. Target Users

Software engineers

ML engineers

Internal R&D developers

Users are assumed to be comfortable with:

APIs

RTSP streams

JSON

Local network services

5. High-Level Architecture
Components

Electron macOS App

UI for managing YOLO instances

Spawns and terminates Python processes

Tracks instance state (running, stopped, error)

Assigns instance names (yolo1, yolo2, etc.)

Assigns network ports

YOLO Instance Backend (Python)

One process per instance

Uses Ultralytics YOLOv8

Runs inference loop on a single RTSP stream

Exposes REST + WebSocket API

Uses Apple Silicon MPS if available

External Client Applications

Run on other machines on the same network

Provide RTSP streams

Consume detection results via WebSocket

6. Instance Model
Definition

A YOLO Instance is defined as:

One Python process

One YOLOv8 pretrained model

One video stream

One REST + WebSocket API

One unique port

Naming

Instances are named sequentially: yolo1, yolo2, yolo3, etc.

Names are user-visible identifiers

Names do not imply model uniqueness (multiple instances may use the same model)

7. Lifecycle Management
Creation

User clicks “Add YOLO” in the Electron UI

Electron:

Assigns next available name (yoloN)

Selects an available port

Spawns a Python process with configuration arguments

Running

Python backend starts HTTP server

Model is loaded into memory

Instance waits for stream start command

Stopping

User clicks “Stop” or “Remove”

Electron sends termination signal

Python process shuts down gracefully

Failure Handling

Electron detects process exit

Instance marked as errored

Logs exposed in UI (basic)

8. Backend Technology Stack

Python 3.10+

Ultralytics YOLOv8

FastAPI

Uvicorn

OpenCV

PyTorch with MPS backend

Inference must:

Prefer mps device

Fall back to CPU if MPS is unavailable

9. Video Input
Supported Input (v1)

RTSP streams only

Each YOLO instance:

Accepts exactly one RTSP stream

Pulls frames internally using OpenCV

Processes frames in a continuous loop

Frame rate control may be implemented via:

Frame skipping

Fixed inference interval

10. API Design

Each YOLO instance exposes an API on its assigned port.

REST API
GET /health

Returns instance health status.

Response:

{
  "status": "running",
  "model": "yolov8n.pt",
  "device": "mps"
}

GET /model

Returns model metadata.

{
  "model_name": "yolov8n.pt",
  "classes": ["person", "car", "..."]
}

POST /stream/start

Starts inference on a given RTSP stream.

Request:

{
  "stream_url": "rtsp://...",
  "conf": 0.5,
  "iou": 0.45
}


Response:

{
  "status": "started"
}

POST /stream/stop

Stops the active stream.

Response:

{
  "status": "stopped"
}

WebSocket API
WS /detections

Streams detection results continuously.

Message format:

{
  "frame_id": 1023,
  "timestamp": 1739023132,
  "detections": [
    {
      "class": "person",
      "confidence": 0.87,
      "bbox": [x, y, width, height]
    }
  ]
}


Notes:

Messages are emitted per processed frame or every N frames

Bounding boxes are in pixel coordinates relative to the input frame

11. Electron Application Responsibilities

Electron must not:

Perform inference

Load ML models

Process video

Electron must:

Provide UI to add/remove instances

Spawn Python processes with correct arguments

Track instance ports and names

Display instance status

Handle logs at a basic level

12. Configuration & Launch Parameters

Each YOLO instance process is launched with arguments such as:

Model path

Port

Device preference

Instance name

Example:

python yolo_server.py \
  --model yolov8n.pt \
  --port 9001 \
  --name yolo1 \
  --device mps

13. Networking Assumptions

All machines are on the same LAN

No NAT traversal

No TLS (HTTP + WS only)

IP-based access is allowed

14. Performance Considerations

Each instance consumes GPU (MPS) resources independently

Multiple instances may contend for GPU

No global scheduler in v1

Users are responsible for not oversubscribing hardware

15. Risks & Tradeoffs

Multiple MPS-backed models may degrade performance

RTSP reliability depends on network quality

Python process crashes must be handled gracefully

No authentication means network trust is assumed

16. Future Extensions (Out of Scope for v1)

Multi-stream per instance

Model hot-swapping

UI-based model upload

Metrics (FPS, latency)

Authentication

Containerization

Windows/Linux support





