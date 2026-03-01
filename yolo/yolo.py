import argparse
import logging
from typing import Optional

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, UploadFile, File, Query
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
        self.width: Optional[int] = None
        self.height: Optional[int] = None
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
    classes: Optional[list[str]] = Query(None)
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

        # Resize logic
        original_height, original_width = frame.shape[:2]
        target_width = state.width
        target_height = state.height
        
        inference_frame = frame
        scale_x = 1.0
        scale_y = 1.0

        if target_width and target_height and (original_width != target_width or original_height != target_height):
            logger.info(f"Resizing image from {original_width}x{original_height} to {target_width}x{target_height}")
            inference_frame = cv2.resize(frame, (target_width, target_height))
            scale_x = original_width / target_width
            scale_y = original_height / target_height

        # Perform inference
        results = model(inference_frame, conf=conf, iou=iou, verbose=False)
        
        # Prepare detection data
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                confidence = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = model.names[cls]
                
                # Apply optional class filter
                if classes and class_name not in classes:
                    continue

                # Get box coordinates (x, y, w, h)
                b = box.xywh[0].tolist()
                
                # Scale back if resized
                if scale_x != 1.0 or scale_y != 1.0:
                    b[0] *= scale_x # x
                    b[1] *= scale_y # y
                    b[2] *= scale_x # w
                    b[3] *= scale_y # h
                
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
    parser.add_argument("--width", type=int, default=None, help="Resize width")
    parser.add_argument("--height", type=int, default=None, help="Resize height")
    
    args = parser.parse_args()
    
    state.model_path = args.model
    state.port = args.port
    state.name = args.name
    state.device = args.device
    state.width = args.width
    state.height = args.height
    
    logger.info(f"Initializing instance {state.name} on port {state.port} using {state.device} with model {state.model_path}")
    if state.width and state.height:
        logger.info(f"Target resize dimensions: {state.width}x{state.height}")
    else:
        logger.info("No target resize dimensions specified (using original size)")
    
    try:
        state.model = YOLO(args.model)
        state.model.to(args.device)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")

if __name__ == "__main__":
    main()
