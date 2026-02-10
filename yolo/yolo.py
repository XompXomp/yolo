import argparse
import asyncio
import base64
import json
import logging
import threading
import time
import uuid
from typing import Optional

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import (
    Depends,
    FastAPI,
    File,
    Header,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
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


# --- OpenAI-style vision detection schemas & helpers ---

class DetectionRequest(BaseModel):
    """
    JSON request body for /v1/vision/detections
    Accepts a base64-encoded image string plus optional thresholds.
    """

    model: str
    image: str  # base64 string, optionally with data:image/... prefix
    conf: float = 0.5
    iou: float = 0.45


class Detection(BaseModel):
    object: str = "detection"
    index: int
    cls: str
    confidence: float
    bbox: list[float]  # [x_center, y_center, width, height] in pixels
    bbox_format: str = "xywh"


class DetectionResponse(BaseModel):
    id: str
    object: str = "image.detections"
    created: int
    model: str
    data: list[Detection]


API_KEY: Optional[str] = None  # optionally set from env/config for real deployments


def _error(detail_message: str, error_type: str, status_code: int) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={
            "error": {
                "message": detail_message,
                "type": error_type,
            }
        },
    )


async def verify_auth(authorization: Optional[str] = Header(None)):
    """
    OpenAI-style Bearer auth.
    If API_KEY is None, auth is effectively disabled (useful for local testing).
    """
    if API_KEY is None:
        return

    if not authorization or not authorization.startswith("Bearer "):
        raise _error("Missing or invalid Authorization header", "invalid_auth", 401)

    token = authorization.removeprefix("Bearer ").strip()
    if token != API_KEY:
        raise _error("Invalid API key", "invalid_api_key", 401)


def decode_base64_image(data: str):
    """
    Decode a base64-encoded image (optionally with a data: URI prefix) into a BGR numpy array.
    """
    if "," in data and data.strip().startswith("data:"):
        data = data.split(",", 1)[1]

    try:
        img_bytes = base64.b64decode(data)
    except Exception:
        raise _error("Invalid base64 image data", "invalid_request_error", 400)

    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise _error("Could not decode image", "invalid_request_error", 400)

    return img


def run_image_inference(model_name: str, image_bgr, conf: float, iou: float) -> list[Detection]:
    """
    Run YOLO on a single image and return structured detections.
    Currently uses the single loaded instance in `state.model` and
    ignores `model_name` selection beyond logging.
    """
    if state.model is None:
        raise _error("Model not loaded", "server_error", 500)

    if model_name != state.model_path:
        logger.warning(
            f"Requested model '{model_name}' does not match loaded model '{state.model_path}'. "
            "Using loaded model."
        )

    try:
        results = state.model(image_bgr, conf=conf, iou=iou, verbose=False)
    except Exception as e:
        logger.error(f"Model inference error: {e}")
        raise _error("Model inference failed", "server_error", 500)

    detections: list[Detection] = []
    idx = 0

    for r in results:
        boxes = r.boxes
        for box in boxes:
            b_xywh = box.xywh[0].tolist()
            conf_val = float(box.conf[0])
            cls_idx = int(box.cls[0])
            class_name = state.model.names[cls_idx]

            detections.append(
                Detection(
                    index=idx,
                    cls=class_name,
                    confidence=conf_val,
                    bbox=[float(x) for x in b_xywh],
                )
            )
            idx += 1

    return detections

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


# --- OpenAI-style image detection endpoints ---

@app.post("/v1/vision/detections", response_model=DetectionResponse)
async def detect_from_base64(
    body: DetectionRequest,
    auth=Depends(verify_auth),
):
    """
    Accept a base64-encoded image and return detections in an OpenAI-style envelope.
    """
    image_bgr = decode_base64_image(body.image)
    detections = run_image_inference(body.model, image_bgr, body.conf, body.iou)

    return DetectionResponse(
        id=f"detect-{uuid.uuid4().hex[:8]}",
        created=int(time.time()),
        model=state.model_path or body.model,
        data=detections,
    )


@app.post("/v1/vision/detections:file", response_model=DetectionResponse)
async def detect_from_file(
    model: str,
    conf: float = 0.5,
    iou: float = 0.45,
    file: UploadFile = File(...),
    auth=Depends(verify_auth),
):
    """
    Accept an uploaded image file and return detections in an OpenAI-style envelope.
    """
    content = await file.read()
    nparr = np.frombuffer(content, np.uint8)
    image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise _error("Could not decode uploaded image", "invalid_request_error", 400)

    detections = run_image_inference(model, image_bgr, conf, iou)

    return DetectionResponse(
        id=f"detect-{uuid.uuid4().hex[:8]}",
        created=int(time.time()),
        model=state.model_path or model,
        data=detections,
    )

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
