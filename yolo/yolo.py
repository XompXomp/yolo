import argparse
import logging
from typing import Optional

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, UploadFile, File
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

state = InstanceState()

# Pydantic models for API
class HealthResponse(BaseModel):
    status: str
    model: str
    device: str

class ModelMetadata(BaseModel):
    model_name: str
    classes: list[str]



# --- API Endpoints ---

@app.get("/health", response_model=HealthResponse)
async def get_health():
    return {
        "status": "running",
        "model": state.model_path,
        "device": state.device,
    }

@app.get("/model", response_model=ModelMetadata)
async def get_model():
    model = state.model
    if model is None:
         return {
            "model_name": state.model_path,
            "classes": []
        }
    return {
        "model_name": state.model_path,
        "classes": list(model.names.values())
    }

@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    conf: float = 0.5,
    iou: float = 0.45,
):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
             return {"error": "Could not decode image"}

        model = state.model
        if model is None:
            return {"error": "Model not initialized"}

        # Perform inference
        results = model(frame, conf=conf, iou=iou, verbose=False)
        
        # Prepare detection data
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get box coordinates (x, y, w, h)
                b = box.xywh[0].tolist() 
                confidence = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = model.names[cls]
                
                detections.append({
                    "class": class_name,
                    "confidence": round(confidence, 3),
                    "bbox": [round(x, 2) for x in b]
                })

        return {
            "detections": detections
        }

    except Exception as e:
        logger.error(f"Inference error: {e}")
        return {"error": str(e)}

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
