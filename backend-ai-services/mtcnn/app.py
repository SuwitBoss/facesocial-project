'''
MTCNN Face Detection Service - Enhanced Response Format
FastAPI microservice for face detection using MTCNN model with landmarks and standardized response
'''

import os
import cv2
import numpy as np
import torch
import warnings
from facenet_pytorch import MTCNN
from fastapi import FastAPI, File, UploadFile, HTTPException, status, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Tuple
import logging
import psutil
import GPUtil
import time
from contextlib import asynccontextmanager
import asyncio
from PIL import Image
import io
import base64
import json

# Suppress PyTorch warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.load.*weights_only.*")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GPUMemoryManager:
    """Enhanced GPU Memory Manager for MTCNN"""
    
    def __init__(self, memory_limit_mb: int = 1000):
        self.memory_limit_mb = memory_limit_mb
        self.device = None
        self.mtcnn = None
        
    def setup_gpu_device(self):
        """Setup GPU device with memory management"""
        try:
            if torch.cuda.is_available():
                self.device = torch.device('cuda:0')
                
                if hasattr(torch.cuda, 'set_memory_fraction'):
                    memory_fraction = self.memory_limit_mb / 6144
                    torch.cuda.set_memory_fraction(memory_fraction, device=0)
                
                torch.cuda.empty_cache()
                logger.info(f"GPU device setup successful: {self.device}")
            else:
                self.device = torch.device('cpu')
                logger.warning("CUDA not available, using CPU")
                
            return self.device
            
        except Exception as e:
            logger.error(f"Failed to setup GPU device: {e}")
            self.device = torch.device('cpu')
            return self.device
    
    def setup_mtcnn(self):
        """Setup enhanced MTCNN model"""
        try:
            self.mtcnn = MTCNN(
                image_size=160,
                margin=0,
                min_face_size=20,
                thresholds=[0.6, 0.7, 0.7],
                factor=0.709,
                post_process=True,
                device=self.device,
                keep_all=True,
                selection_method='probability'
            )
            
            logger.info("Enhanced MTCNN model loaded successfully")
            return self.mtcnn
            
        except Exception as e:
            logger.error(f"Failed to setup MTCNN: {e}")
            raise

class EnhancedMTCNNDetector:
    """Enhanced MTCNN detector with quality assessment and standardized response"""
    
    def __init__(self, mtcnn_model):
        self.mtcnn = mtcnn_model
        
    def detect_faces(self, image: np.ndarray) -> Dict[str, Any]:
        """Enhanced face detection with landmarks and quality assessment"""
        try:
            start_time = time.time()
            
            # Convert numpy array to PIL Image
            if isinstance(image, np.ndarray):
                # Convert BGR to RGB
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_image)
            else:
                pil_image = image
                rgb_image = np.array(pil_image)
            
            original_width, original_height = pil_image.size
            
            # Run MTCNN detection
            with torch.no_grad():
                boxes, probs, landmarks = self.mtcnn.detect(pil_image, landmarks=True)
            
            faces = []
            
            if boxes is not None and len(boxes) > 0:
                for i, (box, prob, landmark) in enumerate(zip(boxes, probs, landmarks)):
                    if prob > 0.5:
                        x1, y1, x2, y2 = box
                        width = x2 - x1
                        height = y2 - y1
                        
                        # Extract face crop for quality assessment
                        face_crop = self._extract_face_crop(rgb_image, [x1, y1, x2, y2])
                        
                        # Process landmarks
                        landmarks_data = self._process_landmarks(landmark) if landmark is not None else None
                        
                        # Create enhanced face object
                        face_obj = {
                            "face_id": f"mtcnn_face_{i}",
                            "face_index": i,
                            "x": float(x1),  # Legacy format
                            "y": float(y1),
                            "width": float(width),
                            "height": float(height),
                            "confidence": float(prob),
                            "bounding_box": {  # Standardized format
                                "x": float(x1),
                                "y": float(y1),
                                "width": float(width),
                                "height": float(height),
                                "confidence": float(prob)
                            },
                            "landmarks": landmarks_data,
                            "detection_method": "mtcnn",
                            "quality_score": self._assess_face_quality(face_crop),
                            "area": float(width * height)
                        }
                        
                        faces.append(face_obj)
            
            processing_time = time.time() - start_time
            
            return {
                "faces": faces,
                "total_faces": len(faces),
                "processing_time": processing_time,
                "image_info": {
                    "width": original_width,
                    "height": original_height,
                    "channels": 3
                },
                "model_info": {
                    "detector": "mtcnn",
                    "version": "facenet_pytorch_2.5.3",
                    "min_face_size": 20,
                    "thresholds": [0.6, 0.7, 0.7],
                    "landmarks": True
                }
            }
            
        except Exception as e:
            logger.error(f"MTCNN detection failed: {e}")
            return {
                "faces": [],
                "total_faces": 0,
                "processing_time": 0.0,
                "error": str(e)
            }
    
    def _extract_face_crop(self, image: np.ndarray, box: List[float]) -> np.ndarray:
        """Extract face crop for quality assessment"""
        try:
            x1, y1, x2, y2 = [int(coord) for coord in box]
            
            # Ensure coordinates are within bounds
            height, width = image.shape[:2]
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(x1 + 1, min(x2, width))
            y2 = max(y1 + 1, min(y2, height))
            
            return image[y1:y2, x1:x2]
        except Exception:
            return np.zeros((50, 50, 3), dtype=np.uint8)
    
    def _process_landmarks(self, landmarks: torch.Tensor) -> Dict[str, Any]:
        """Process MTCNN landmarks into standardized format"""
        try:
            if landmarks is None or landmarks.numel() == 0:
                return None
            
            if isinstance(landmarks, torch.Tensor):
                landmarks_np = landmarks.cpu().numpy()
            else:
                landmarks_np = landmarks
            
            if landmarks_np.size == 0:
                return None
            
            # MTCNN provides 5 landmarks: left_eye, right_eye, nose, left_mouth, right_mouth
            landmark_names = ['left_eye', 'right_eye', 'nose', 'left_mouth', 'right_mouth']
            
            landmarks_list = []
            for i, name in enumerate(landmark_names):
                if i < len(landmarks_np):
                    landmarks_list.append({
                        "name": name,
                        "x": float(landmarks_np[i, 0]),
                        "y": float(landmarks_np[i, 1]),
                        "confidence": 1.0
                    })
            
            return {
                "type": "5_point",
                "points": landmarks_list,
                "confidence": 0.9
            }
            
        except Exception as e:
            logger.error(f"Error processing landmarks: {e}")
            return None
    
    def _assess_face_quality(self, face_crop: np.ndarray) -> float:
        """Assess face quality with multiple metrics"""
        try:
            if face_crop.size == 0:
                return 0.0
            
            gray = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY)
            
            # Sharpness (Laplacian variance)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness = min(laplacian_var / 500.0, 1.0)
            
            # Contrast
            contrast = gray.std() / 255.0
            contrast_score = min(contrast / 0.5, 1.0)
            
            # Brightness
            brightness = np.mean(gray) / 255.0
            brightness_score = 1.0 - abs(brightness - 0.5) * 2
            
            # Size adequacy
            area = face_crop.shape[0] * face_crop.shape[1]
            size_score = min(area / (80 * 80), 1.0)
            
            # Face symmetry (simple check)
            height, width = gray.shape
            if width > 20:
                left_half = gray[:, :width//2]
                right_half = gray[:, width//2:]
                right_flipped = cv2.flip(right_half, 1)
                
                if left_half.shape == right_flipped.shape:
                    symmetry = cv2.matchTemplate(left_half, right_flipped, cv2.TM_CCOEFF_NORMED)[0][0]
                    symmetry_score = max(0, symmetry)
                else:
                    symmetry_score = 0.5
            else:
                symmetry_score = 0.5
            
            # Combined quality score
            quality = (
                sharpness * 0.3 +
                contrast_score * 0.2 +
                brightness_score * 0.2 +
                size_score * 0.2 +
                symmetry_score * 0.1
            )
            
            return float(min(max(quality, 0.0), 1.0))
            
        except Exception:
            return 0.5

# Global variables
gpu_manager = GPUMemoryManager(memory_limit_mb=1000)
mtcnn_model = None
mtcnn_detector = None

# Metrics
metrics = {
    "total_requests": 0,
    "total_faces_detected": 0,
    "total_processing_time": 0.0,
    "start_time": time.time()
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global mtcnn_model, mtcnn_detector
    
    # Startup
    logger.info("Starting Enhanced MTCNN Face Detection Service...")
    
    try:
        device = gpu_manager.setup_gpu_device()
        mtcnn_model = gpu_manager.setup_mtcnn()
        mtcnn_detector = EnhancedMTCNNDetector(mtcnn_model)
        logger.info("Enhanced MTCNN model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Enhanced MTCNN Face Detection Service...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app = FastAPI(
    title="Enhanced MTCNN Face Detection Service",
    description="AI microservice for face detection and landmark extraction using MTCNN",
    version="1.1.0",
    lifespan=lifespan
)

@app.post("/detect")
async def detect_faces(file: UploadFile = File(...)):
    """Enhanced face detection with landmarks"""
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        image_data = await file.read()
        image_pil = Image.open(io.BytesIO(image_data))
        
        if image_pil.mode != 'RGB':
            image_pil = image_pil.convert('RGB')
        
        # Detect faces
        result = mtcnn_detector.detect_faces(image_pil)
        
        # Update metrics
        metrics["total_requests"] += 1
        metrics["total_faces_detected"] += result["total_faces"]
        metrics["total_processing_time"] += result["processing_time"]
        
        return {
            "success": True,
            **result
        }
        
    except Exception as e:
        logger.error(f"Error in detect endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect/batch")
async def detect_faces_batch(files: List[UploadFile] = File(...)):
    """Enhanced batch face detection with landmarks"""
    if len(files) > 8:
        raise HTTPException(status_code=400, detail="Maximum 8 images per batch")
    
    start_time = time.time()
    results = []
    total_faces = 0
    
    for i, file in enumerate(files):
        if not file.content_type.startswith('image/'):
            continue
        
        try:
            image_data = await file.read()
            image_pil = Image.open(io.BytesIO(image_data))
            
            if image_pil.mode != 'RGB':
                image_pil = image_pil.convert('RGB')
            
            result = mtcnn_detector.detect_faces(image_pil)
            result["image_id"] = i
            result["filename"] = file.filename
            results.append(result)
            total_faces += result["total_faces"]
            
        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {e}")
            results.append({
                "image_id": i,
                "filename": file.filename,
                "faces": [],
                "total_faces": 0,
                "error": str(e)
            })
    
    total_time = time.time() - start_time
    
    # Update metrics
    metrics["total_requests"] += 1
    metrics["total_faces_detected"] += total_faces
    metrics["total_processing_time"] += total_time
    
    return {
        "success": True,
        "batch_results": results,
        "summary": {
            "total_images": len(files),
            "processed_images": len(results),
            "total_faces_detected": total_faces,
            "batch_processing_time": total_time
        }
    }

@app.get("/health")
async def health_check():
    """Enhanced health check endpoint"""
    gpu_available = torch.cuda.is_available()
    
    memory_info = {
        "ram_percent": psutil.virtual_memory().percent,
        "ram_available_gb": psutil.virtual_memory().available / (1024**3)
    }
    
    if gpu_available:
        memory_info.update({
            "gpu_memory_allocated_mb": torch.cuda.memory_allocated() / (1024**2),
            "gpu_memory_cached_mb": torch.cuda.memory_reserved() / (1024**2)
        })
        
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                memory_info.update({
                    "gpu_memory_percent": gpu.memoryUtil * 100,
                    "gpu_memory_used_mb": gpu.memoryUsed,
                    "gpu_memory_total_mb": gpu.memoryTotal
                })
        except Exception:
            pass
    
    uptime = time.time() - metrics["start_time"]
    
    return {
        "status": "healthy",
        "service": "MTCNN Face Detection",
        "version": "1.1.0",
        "model_loaded": mtcnn_model is not None,
        "gpu_available": gpu_available,
        "memory_usage": memory_info,
        "uptime": uptime,
        "capabilities": {
            "face_detection": True,
            "landmarks": True,
            "quality_assessment": True,
            "batch_processing": True,
            "precision_detection": True
        }
    }

@app.get("/metrics")
async def get_metrics():
    """Get enhanced service metrics"""
    avg_processing_time = (metrics["total_processing_time"] / metrics["total_requests"] 
                          if metrics["total_requests"] > 0 else 0)
    
    gpu_utilization = 0
    memory_info = {
        "ram_percent": psutil.virtual_memory().percent,
        "ram_available_gb": psutil.virtual_memory().available / (1024**3)
    }
    
    if torch.cuda.is_available():
        memory_info.update({
            "gpu_memory_allocated_mb": torch.cuda.memory_allocated() / (1024**2),
            "gpu_memory_cached_mb": torch.cuda.memory_reserved() / (1024**2)
        })
    
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            gpu_utilization = gpu.load * 100
            memory_info.update({
                "gpu_memory_percent": gpu.memoryUtil * 100,
                "gpu_memory_used_mb": gpu.memoryUsed,
                "gpu_memory_total_mb": gpu.memoryTotal
            })
    except Exception:
        pass
    
    return {
        "total_requests": metrics["total_requests"],
        "total_faces_detected": metrics["total_faces_detected"],
        "average_processing_time": avg_processing_time,
        "gpu_utilization": gpu_utilization,
        "memory_usage": memory_info,
        "model_info": {
            "name": "MTCNN",
            "min_face_size": 20,
            "thresholds": [0.6, 0.7, 0.7],
            "landmarks_type": "5_point",
            "precision": "high"
        }
    }

@app.get("/")
async def root():
    """Enhanced root endpoint"""
    return {
        "message": "Enhanced MTCNN Face Detection Service",
        "version": "1.1.0",
        "status": "running",
        "capabilities": {
            "face_detection": True,
            "landmarks": True,
            "quality_assessment": True,
            "batch_processing": True,
            "precision_detection": True,
            "gpu_acceleration": True
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
