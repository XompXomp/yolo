import argparse
import asyncio
import json
import logging
import threading
import time
from typing import Optional

import cv2
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from ultralytics import YOLO

# Configuration and Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("YOLOInstance")

app = FastAPI()

# Global state
class InstanceState:
    def __init__(self):
        self.model_path: str = ""
        self.device: str = "cpu"
        self.name: str = ""
        self.port: int = 9000
        self.model: Optional[YOLO] = None
        self.stream_url: Optional[str] = None
        self.is_running: bool = False
        self.conf_threshold: float = 0.5
        self.iou_threshold: float = 0.45
        self.active_connections: list[WebSocket] = []
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None

state = InstanceState()

# Pydantic models for API
class StreamStartRequest(BaseModel):
    stream_url: str
    conf: float = 0.5
    iou: float = 0.45

class HealthResponse(BaseModel):
    status: str
    model: str
    device: str
    stream_active: bool

class ModelMetadata(BaseModel):
    model_name: str
    classes: list[str]

# --- websocket management ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                # Handle stale connections if necessary
                pass

manager = ConnectionManager()

# --- Inference Engine ---
def run_inference_loop():
    logger.info(f"Starting inference loop for {state.stream_url}")
    
    cap = cv2.VideoCapture(state.stream_url)
    if not cap.isOpened():
        logger.error(f"Failed to open video source: {state.stream_url}")
        state.is_running = False
        return

    is_file = not (state.stream_url.startswith("rtsp") or state.stream_url.startswith("http"))
    fps = cap.get(cv2.CAP_PROP_FPS) if is_file else 0
    frame_delay = 1.0 / fps if fps > 0 else 0

    frame_id = 0
    try:
        while not state.stop_event.is_set():
            start_time = time.time()
            success, frame = cap.read()
            if not success:
                if is_file:
                    logger.info("End of video file reached.")
                    break
                logger.warning("Failed to read frame, retrying...")
                time.sleep(1)
                continue

            frame_id += 1
            
            # Perform inference
            results = state.model(frame, conf=state.conf_threshold, iou=state.iou_threshold, verbose=False)
            
            # Prepare detection data
            detections = []
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Get box coordinates (x, y, w, h)
                    b = box.xywh[0].tolist() 
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = state.model.names[cls]
                    
                    detections.append({
                        "class": class_name,
                        "confidence": round(conf, 3),
                        "bbox": [round(x, 2) for x in b]
                    })

            # Broadcast via WebSocket
            payload = {
                "frame_id": frame_id,
                "timestamp": int(time.time()),
                "detections": detections
            }
            asyncio.run_coroutine_threadsafe(manager.broadcast(payload), app.loop)

            # Pacing for files
            if is_file:
                elapsed = time.time() - start_time
                wait = frame_delay - elapsed
                if wait > 0:
                    time.sleep(wait)

    except Exception as e:
        logger.error(f"Inference loop error: {e}")
    finally:
        cap.release()
        state.is_running = False
        logger.info("Inference loop stopped")

# --- API Endpoints ---

@app.on_event("startup")
async def startup_event():
    # Save the running loop for thread-safe websocket broadcasting
    app.loop = asyncio.get_event_loop()

@app.get("/health", response_model=HealthResponse)
async def get_health():
    return {
        "status": "running",
        "model": state.model_path,
        "device": state.device,
        "stream_active": state.is_running
    }

@app.get("/model", response_model=ModelMetadata)
async def get_model():
    return {
        "model_name": state.model_path,
        "classes": list(state.model.names.values())
    }

@app.post("/stream/start")
async def start_stream(req: StreamStartRequest):
    if state.is_running:
        return {"status": "error", "message": "Stream already running"}
    
    state.stream_url = req.stream_url
    state.conf_threshold = req.conf
    state.iou_threshold = req.iou
    state.is_running = True
    state.stop_event.clear()
    
    state.thread = threading.Thread(target=run_inference_loop, daemon=True)
    state.thread.start()
    
    return {"status": "started"}

@app.post("/stream/stop")
async def stop_stream():
    if not state.is_running:
        return {"status": "error", "message": "No stream running"}
    
    state.stop_event.set()
    if state.thread:
        state.thread.join(timeout=2)
    
    state.is_running = False
    return {"status": "stopped"}

@app.websocket("/detections")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # We just need to keep the connection open. 
            # Detections are pushed from the inference loop.
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

# --- CLI Entry Point ---

def main():
    parser = argparse.ArgumentParser(description="YOLO Instance Backend")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to YOLO model")
    parser.add_argument("--port", type=int, default=9000, help="Port to run the API on")
    parser.add_argument("--name", type=str, default="yolo1", help="Instance name")
    parser.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else "cpu", help="Device (mps, cpu, cuda)")
    
    args = parser.parse_args()
    
    state.model_path = args.model
    state.port = args.port
    state.name = args.name
    state.device = args.device
    
    logger.info(f"Initializing instance {state.name} on port {state.port} using {state.device} with model {state.model_path}")
    
    try:
        state.model = YOLO(args.model)
        state.model.to(args.device)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")

if __name__ == "__main__":
    main()
