"""
MTCNN Face Detection Service
FastAPI microservice for face detection using MTCNN model with GPU acceleration
"""

import os
import cv2
import numpy as np
import torch
import warnings
from facenet_pytorch import MTCNN
from fastapi import FastAPI, File, UploadFile, HTTPException, status
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

# Suppress PyTorch FutureWarnings about weights_only
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.load.*weights_only.*")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# GPU Memory Manager
class GPUMemoryManager:
    def __init__(self, memory_limit_mb: int = 1000):
        self.memory_limit_mb = memory_limit_mb
        self.device = None
        self.mtcnn = None
        
    def setup_gpu_device(self):
        """Setup GPU device with memory management"""
        try:
            if torch.cuda.is_available():
                self.device = torch.device('cuda:0')
                
                # Set GPU memory fraction
                if hasattr(torch.cuda, 'set_memory_fraction'):
                    memory_fraction = self.memory_limit_mb / 6144  # Assuming 6GB GPU
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
        """Setup MTCNN model"""
        try:
            self.mtcnn = MTCNN(
                image_size=160,
                margin=0,
                min_face_size=20,
                thresholds=[0.6, 0.7, 0.7],  # P-Net, R-Net, O-Net thresholds
                factor=0.709,
                post_process=True,
                device=self.device,
                keep_all=True,
                selection_method='probability'
            )
            
            logger.info("MTCNN model loaded successfully")
            return self.mtcnn
            
        except Exception as e:
            logger.error(f"Failed to setup MTCNN: {e}")
            raise

# Global variables
gpu_manager = GPUMemoryManager(memory_limit_mb=1000)
mtcnn_model = None

# Pydantic models
class FaceDetection(BaseModel):
    x: float
    y: float
    width: float
    height: float
    confidence: float

class FaceLandmarks(BaseModel):
    left_eye: List[float]
    right_eye: List[float]
    nose: List[float]
    mouth_left: List[float]
    mouth_right: List[float]

class DetailedFaceDetection(BaseModel):
    x: float
    y: float
    width: float
    height: float
    confidence: float
    landmarks: Optional[FaceLandmarks] = None

class DetectionResponse(BaseModel):
    faces: List[DetailedFaceDetection]
    processing_time: float
    image_width: int
    image_height: int
    model_info: str

class BatchDetectionResponse(BaseModel):
    results: List[DetectionResponse]
    total_processing_time: float
    total_faces_detected: int

class HealthResponse(BaseModel):
    status: str
    service: str
    model_loaded: bool
    gpu_available: bool
    memory_usage: Dict[str, Any]
    uptime: float

class MetricsResponse(BaseModel):
    total_requests: int
    total_faces_detected: int
    average_processing_time: float
    gpu_utilization: float
    memory_usage: Dict[str, Any]

# Global metrics
metrics = {
    "total_requests": 0,
    "total_faces_detected": 0,
    "total_processing_time": 0.0,
    "start_time": time.time()
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global mtcnn_model
    
    # Startup
    logger.info("Starting MTCNN Face Detection Service...")
    
    try:
        device = gpu_manager.setup_gpu_device()
        mtcnn_model = gpu_manager.setup_mtcnn()
        logger.info("MTCNN model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down MTCNN Face Detection Service...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app = FastAPI(
    title="MTCNN Face Detection Service",
    description="AI microservice for face detection and landmark extraction using MTCNN",
    version="1.0.0",
    lifespan=lifespan
)

def convert_landmarks_to_dict(landmarks: torch.Tensor) -> FaceLandmarks:
    """Convert MTCNN landmarks tensor to structured format"""
    if landmarks is None:
        return None
    
    # Handle both tensor and numpy array cases
    if isinstance(landmarks, torch.Tensor):
        if landmarks.numel() == 0:
            return None
        landmarks_np = landmarks.cpu().numpy()
    else:
        # Already a numpy array
        if landmarks.size == 0:
            return None
        landmarks_np = landmarks
    
    return FaceLandmarks(
        left_eye=[float(landmarks_np[0, 0]), float(landmarks_np[0, 1])],
        right_eye=[float(landmarks_np[1, 0]), float(landmarks_np[1, 1])],
        nose=[float(landmarks_np[2, 0]), float(landmarks_np[2, 1])],
        mouth_left=[float(landmarks_np[3, 0]), float(landmarks_np[3, 1])],
        mouth_right=[float(landmarks_np[4, 0]), float(landmarks_np[4, 1])]
    )

async def process_image(image_data: bytes) -> DetectionResponse:
    """Process a single image for face detection"""
    global metrics
    start_time = time.time()
    
    try:
        # Convert bytes to PIL Image
        image_pil = Image.open(io.BytesIO(image_data))
        if image_pil.mode != 'RGB':
            image_pil = image_pil.convert('RGB')
        
        original_width, original_height = image_pil.size
        
        # Run MTCNN detection
        with torch.no_grad():
            boxes, probs, landmarks = mtcnn_model.detect(image_pil, landmarks=True)
        
        detections = []
        
        if boxes is not None and len(boxes) > 0:
            for i, (box, prob, landmark) in enumerate(zip(boxes, probs, landmarks)):
                if prob > 0.5:  # Confidence threshold
                    x1, y1, x2, y2 = box
                    width = x2 - x1
                    height = y2 - y1
                    
                    # Convert landmarks
                    face_landmarks = None
                    if landmark is not None:
                        face_landmarks = convert_landmarks_to_dict(landmark)
                    
                    detections.append(DetailedFaceDetection(
                        x=float(x1),
                        y=float(y1),
                        width=float(width),
                        height=float(height),
                        confidence=float(prob),
                        landmarks=face_landmarks
                    ))
        
        processing_time = time.time() - start_time
        
        # Update metrics
        metrics["total_requests"] += 1
        metrics["total_faces_detected"] += len(detections)
        metrics["total_processing_time"] += processing_time
        
        return DetectionResponse(
            faces=detections,
            processing_time=processing_time,
            image_width=original_width,
            image_height=original_height,
            model_info="MTCNN Face Detection with Landmarks"
        )
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/detect", response_model=DetectionResponse)
async def detect_faces(file: UploadFile = File(...)):
    """Detect faces and landmarks in uploaded image"""
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        image_data = await file.read()
        result = await process_image(image_data)
        return result
    except Exception as e:
        logger.error(f"Error in detect endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect/batch", response_model=BatchDetectionResponse)
async def detect_faces_batch(files: List[UploadFile] = File(...)):
    """Detect faces and landmarks in multiple images"""
    if len(files) > 8:  # Reduce batch size for MTCNN
        raise HTTPException(status_code=400, detail="Maximum 8 images per batch")
    
    start_time = time.time()
    results = []
    total_faces = 0
    
    for file in files:
        if not file.content_type.startswith('image/'):
            continue
        
        try:
            image_data = await file.read()
            result = await process_image(image_data)
            results.append(result)
            total_faces += len(result.faces)
        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {e}")
            continue
    
    total_time = time.time() - start_time
    
    return BatchDetectionResponse(
        results=results,
        total_processing_time=total_time,
        total_faces_detected=total_faces
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
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
        
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            memory_info.update({
                "gpu_memory_percent": gpu.memoryUtil * 100,
                "gpu_memory_used_mb": gpu.memoryUsed,
                "gpu_memory_total_mb": gpu.memoryTotal
            })
    
    uptime = time.time() - metrics["start_time"]
    
    return HealthResponse(
        status="healthy",
        service="MTCNN Face Detection",
        model_loaded=mtcnn_model is not None,
        gpu_available=gpu_available,
        memory_usage=memory_info,
        uptime=uptime
    )

@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get service metrics"""
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
    
    gpus = GPUtil.getGPUs()
    if gpus:
        gpu = gpus[0]
        gpu_utilization = gpu.load * 100
        memory_info.update({
            "gpu_memory_percent": gpu.memoryUtil * 100,
            "gpu_memory_used_mb": gpu.memoryUsed,
            "gpu_memory_total_mb": gpu.memoryTotal
        })
    
    return MetricsResponse(
        total_requests=metrics["total_requests"],
        total_faces_detected=metrics["total_faces_detected"],
        average_processing_time=avg_processing_time,
        gpu_utilization=gpu_utilization,
        memory_usage=memory_info
    )

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "MTCNN Face Detection Service", "status": "running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
