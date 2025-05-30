'''
MediaPipe Face Detection Service - Enhanced Response Format
Fast face detection using MediaPipe Solutions API with standardized response format
'''

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
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
import base64
import io
import json
from PIL import Image
from pydantic import BaseModel

class MediaPipeFaceDetector:
    """Enhanced MediaPipe Face Detector with standardized response"""
    
    def __init__(self, min_detection_confidence: float = 0.5):
        self.min_detection_confidence = min_detection_confidence
        
        # Initialize MediaPipe
        self.mp_face_detection = mp.solutions.face_detection
        self.detector = self.mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=min_detection_confidence
        )
        
        logger.info(f"MediaPipe detector initialized with confidence: {min_detection_confidence}")
    
    def detect_faces(self, image: np.ndarray) -> Dict[str, Any]:
        """Detect faces in image with enhanced response format"""
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
                    
                    # Enhanced face object with standardized format
                    face_obj = {
                        "face_id": f"mp_face_{i}",
                        "face_index": i,
                        "bbox": [x, y, width, height],  # For legacy compatibility
                        "bounding_box": {
                            "x": float(x),
                            "y": float(y),
                            "width": float(width),
                            "height": float(height),
                            "confidence": float(confidence)
                        },
                        "confidence": float(confidence),
                        "detection_method": "mediapipe",
                        "landmarks": None,  # MediaPipe basic doesn't provide landmarks
                        "quality_score": self._estimate_quality(image[y:y+height, x:x+width])
                    }
                    
                    faces.append(face_obj)
            
            processing_time = time.time() - start_time
            logger.info(f"Detected {len(faces)} faces in {processing_time:.3f}s")
            
            return {
                "faces": faces,
                "total_faces": len(faces),
                "processing_time": processing_time,
                "image_info": {
                    "width": image.shape[1],
                    "height": image.shape[0],
                    "channels": image.shape[2] if len(image.shape) > 2 else 1
                },
                "model_info": {
                    "detector": "mediapipe",
                    "version": "0.10.7",
                    "confidence_threshold": self.min_detection_confidence
                }
            }
            
        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            return {
                "faces": [],
                "total_faces": 0,
                "processing_time": 0.0,
                "error": str(e)
            }
    
    def _estimate_quality(self, face_crop: np.ndarray) -> float:
        """Simple quality estimation for face crop"""
        try:
            if face_crop.size == 0:
                return 0.0
            
            # Simple quality metrics
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            
            # Sharpness (Laplacian variance)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness = min(laplacian_var / 500.0, 1.0)
            
            # Brightness
            brightness = np.mean(gray) / 255.0
            brightness_score = 1.0 - abs(brightness - 0.5) * 2
            
            # Size adequacy
            area = face_crop.shape[0] * face_crop.shape[1]
            size_score = min(area / (100 * 100), 1.0)  # Optimal at 100x100 or larger
            
            # Combined score
            quality = (sharpness * 0.5 + brightness_score * 0.3 + size_score * 0.2)
            return float(min(max(quality, 0.0), 1.0))
            
        except Exception:
            return 0.5

class DetectionRequest(BaseModel):
    image: str  # base64 encoded image

# Initialize detector
detector = MediaPipeFaceDetector()

# Create FastAPI app
app = FastAPI(
    title="MediaPipe Face Detection Service",
    description="Face detection using MediaPipe with enhanced response format",
    version="1.1.0"
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
        "version": "1.1.0",
        "status": "running",
        "capabilities": {
            "face_detection": True,
            "landmarks": False,
            "quality_assessment": True,
            "batch_processing": True
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "mediapipe-face-detection",
        "version": "1.1.0",
        "model_loaded": True,
        "detector_confidence": detector.min_detection_confidence
    }

@app.post("/detect")
async def detect_faces(file: UploadFile = File(...)):
    """Detect faces in uploaded image file with enhanced response"""
    try:
        # Read image data
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Detect faces
        result = detector.detect_faces(cv_image)
        
        return {
            "success": True,
            **result
        }
        
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect_base64")
async def detect_faces_base64(request: DetectionRequest):
    """Detect faces in base64 encoded image with enhanced response"""
    try:
        # Decode image
        image_data = base64.b64decode(request.image)
        image = Image.open(io.BytesIO(image_data))
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Detect faces
        result = detector.detect_faces(cv_image)
        
        return {
            "success": True,
            **result
        }
        
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect/batch")
async def detect_faces_batch(files: List[UploadFile] = File(...)):
    """Batch detect faces in multiple images"""
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 images per batch")
    
    results = []
    total_faces = 0
    start_time = time.time()
    
    for i, file in enumerate(files):
        try:
            if not file.content_type.startswith('image/'):
                continue
                
            image_data = await file.read()
            image = Image.open(io.BytesIO(image_data))
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            result = detector.detect_faces(cv_image)
            result["image_id"] = i
            result["filename"] = file.filename
            
            results.append(result)
            total_faces += result["total_faces"]
            
        except Exception as e:
            logger.error(f"Error processing {file.filename}: {e}")
            results.append({
                "image_id": i,
                "filename": file.filename,
                "faces": [],
                "total_faces": 0,
                "error": str(e)
            })
    
    batch_time = time.time() - start_time
    
    return {
        "success": True,
        "batch_results": results,
        "summary": {
            "total_images": len(files),
            "processed_images": len(results),
            "total_faces_detected": total_faces,
            "batch_processing_time": batch_time
        }
    }

@app.get("/config")
async def get_config():
    """Get detector configuration"""
    return {
        "min_detection_confidence": detector.min_detection_confidence,
        "model_selection": 0,
        "capabilities": {
            "max_faces": "unlimited",
            "min_face_size": "20x20",
            "supported_formats": ["jpg", "jpeg", "png", "bmp"]
        }
    }

@app.post("/config")
async def update_config(
    min_detection_confidence: float = Form(default=0.5)
):
    """Update detector configuration"""
    try:
        if not 0.0 <= min_detection_confidence <= 1.0:
            raise HTTPException(status_code=400, detail="Confidence must be between 0.0 and 1.0")
        
        # Reinitialize detector with new confidence
        global detector
        detector = MediaPipeFaceDetector(min_detection_confidence)
        
        return {
            "success": True,
            "message": "Configuration updated successfully",
            "new_config": {
                "min_detection_confidence": min_detection_confidence
            }
        }
        
    except Exception as e:
        logger.error(f"Config update failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    logger.info("Starting MediaPipe Face Detection Service...")
    uvicorn.run(app, host="0.0.0.0", port=5000)
