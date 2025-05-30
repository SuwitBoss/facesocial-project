"""
MediaPipe Face Detection Service - Simple Working Version
Fast face detection using MediaPipe Solutions API with GPU support
"""

import os
import warnings
import time
from typing import List, Dict, Any

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ['GLOG_minloglevel'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import mediapipe as mp
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
import base64
import io
from PIL import Image
from pydantic import BaseModel

class MediaPipeFaceDetector:
    """Simple MediaPipe Face Detector using Solutions API"""
    
    def __init__(self, min_detection_confidence: float = 0.5):
        self.min_detection_confidence = min_detection_confidence
        
        # Initialize MediaPipe
        self.mp_face_detection = mp.solutions.face_detection
        self.detector = self.mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=min_detection_confidence
        )
        
        logger.info(f"MediaPipe detector initialized with confidence: {min_detection_confidence}")
    
    def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces in image"""
        try:
            start_time = time.time()
            
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process image
            results = self.detector.process(rgb_image)
            
            faces = []
            if results.detections:
                h, w, _ = image.shape
                
                for i, detection in enumerate(results.detections):
                    bbox = detection.location_data.relative_bounding_box
                    
                    # Convert to pixel coordinates
                    x = int(bbox.xmin * w)
                    y = int(bbox.ymin * h)
                    width = int(bbox.width * w)
                    height = int(bbox.height * h)
                    
                    confidence = detection.score[0] if detection.score else 0.0
                    
                    faces.append({
                        "bbox": [x, y, width, height],
                        "confidence": float(confidence),
                        "id": i
                    })
            
            processing_time = time.time() - start_time
            logger.info(f"Detected {len(faces)} faces in {processing_time:.3f}s")
            
            return faces
            
        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            return []

class DetectionRequest(BaseModel):
    image: str  # base64 encoded image

# Initialize detector
detector = MediaPipeFaceDetector()

# Create FastAPI app
app = FastAPI(
    title="MediaPipe Face Detection Service",
    description="Face detection using MediaPipe",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "MediaPipe Face Detection Service",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "mediapipe-face-detection",
        "version": "1.0.0"
    }

@app.post("/detect")
async def detect_faces(file: UploadFile = File(...)):
    """Detect faces in uploaded image file"""
    try:
        # Read image data
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Detect faces
        faces = detector.detect_faces(cv_image)
        
        return {
            "success": True,
            "faces": faces,
            "count": len(faces)
        }
        
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect_base64")
async def detect_faces_base64(request: DetectionRequest):
    """Detect faces in base64 encoded image"""
    try:
        # Decode image
        image_data = base64.b64decode(request.image)
        image = Image.open(io.BytesIO(image_data))
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Detect faces
        faces = detector.detect_faces(cv_image)
        
        return {
            "success": True,
            "faces": faces,
            "count": len(faces)
        }
        
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    logger.info("Starting MediaPipe Face Detection Service...")
    uvicorn.run(app, host="0.0.0.0", port=5000)
