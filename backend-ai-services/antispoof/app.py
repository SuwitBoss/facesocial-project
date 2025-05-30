"""
Anti-Spoofing Service
FastAPI microservice for face anti-spoofing detection using Silent Face Anti-Spoofing
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simple CNN model for anti-spoofing (placeholder - replace with actual model)
class AntiSpoofingNet(nn.Module):
    def __init__(self, num_classes=2):
        super(AntiSpoofingNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# GPU Memory Manager
class GPUMemoryManager:
    def __init__(self, memory_limit_mb: int = 800):
        self.memory_limit_mb = memory_limit_mb
        self.device = None
        self.model = None
        
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
    
    def setup_model(self):
        """Setup anti-spoofing model"""
        try:
            self.model = AntiSpoofingNet(num_classes=2)
            
            # Load pre-trained weights if available
            model_path = "/app/models/antispoofing.pth"
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint)
                logger.info("Loaded pre-trained anti-spoofing model")
            else:
                logger.warning("No pre-trained model found, using randomly initialized weights")
            
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("Anti-spoofing model loaded successfully")
            return self.model
            
        except Exception as e:
            logger.error(f"Failed to setup anti-spoofing model: {e}")
            raise

# Global variables
gpu_manager = GPUMemoryManager(memory_limit_mb=800)
antispoof_model = None

# Pydantic models
class SpoofingDetection(BaseModel):
    face_id: int
    is_real: bool
    confidence: float
    spoof_score: float
    quality_score: float

class AntiSpoofingResponse(BaseModel):
    detections: List[SpoofingDetection]
    processing_time: float
    total_faces: int
    model_info: str

class BatchAntiSpoofingResponse(BaseModel):
    results: List[AntiSpoofingResponse]
    total_processing_time: float
    total_faces_processed: int
    total_real_faces: int
    total_spoofed_faces: int

class HealthResponse(BaseModel):
    status: str
    service: str
    model_loaded: bool
    gpu_available: bool
    memory_usage: Dict[str, Any]
    uptime: float

class MetricsResponse(BaseModel):
    total_requests: int
    total_faces_processed: int
    total_real_faces: int
    total_spoofed_faces: int
    average_processing_time: float
    gpu_utilization: float
    memory_usage: Dict[str, Any]

# Global metrics
metrics = {
    "total_requests": 0,
    "total_faces_processed": 0,
    "total_real_faces": 0,
    "total_spoofed_faces": 0,
    "total_processing_time": 0.0,
    "start_time": time.time()
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global antispoof_model
    
    # Startup
    logger.info("Starting Anti-Spoofing Service...")
    
    try:
        device = gpu_manager.setup_gpu_device()
        antispoof_model = gpu_manager.setup_model()
        logger.info("Anti-spoofing model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Anti-Spoofing Service...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app = FastAPI(
    title="Anti-Spoofing Service",
    description="AI microservice for face anti-spoofing detection",
    version="1.0.0",
    lifespan=lifespan
)

def preprocess_face(face_image: np.ndarray, input_size: tuple = (256, 256)) -> torch.Tensor:
    """Preprocess face image for anti-spoofing model"""
    try:
        # Resize image
        face_resized = cv2.resize(face_image, input_size)
        
        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        face_normalized = face_rgb.astype(np.float32) / 255.0
        
        # Normalize with ImageNet stats
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        face_normalized = (face_normalized - mean) / std
        
        # Convert to tensor and add batch dimension
        face_tensor = torch.from_numpy(face_normalized).permute(2, 0, 1).unsqueeze(0)
        
        return face_tensor
        
    except Exception as e:
        logger.error(f"Error preprocessing face: {e}")
        raise

def calculate_quality_score(face_image: np.ndarray) -> float:
    """Calculate face quality score based on various metrics"""
    try:
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        
        # Calculate sharpness using Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(laplacian_var / 500.0, 1.0)  # Normalize
        
        # Calculate brightness
        brightness = np.mean(gray) / 255.0
        brightness_score = 1.0 - abs(brightness - 0.5) * 2  # Prefer mid-range brightness
        
        # Calculate contrast
        contrast = gray.std() / 255.0
        contrast_score = min(contrast * 4, 1.0)  # Normalize
        
        # Overall quality score
        quality_score = (sharpness_score + brightness_score + contrast_score) / 3.0
        
        return float(quality_score)
        
    except Exception as e:
        logger.error(f"Error calculating quality score: {e}")
        return 0.5

def detect_spoofing(face_image: np.ndarray) -> Tuple[bool, float]:
    """Detect if face is real or spoofed"""
    try:
        face_tensor = preprocess_face(face_image)
        face_tensor = face_tensor.to(gpu_manager.device)
        
        with torch.no_grad():
            outputs = antispoof_model(face_tensor)
            probabilities = F.softmax(outputs, dim=1)
            
            # Assuming class 0 = spoof, class 1 = real
            spoof_prob = probabilities[0, 0].item()
            real_prob = probabilities[0, 1].item()
            
            is_real = real_prob > spoof_prob
            confidence = max(real_prob, spoof_prob)
            spoof_score = spoof_prob
            
        return is_real, confidence, spoof_score
        
    except Exception as e:
        logger.error(f"Error in spoofing detection: {e}")
        return False, 0.0, 1.0

async def process_anti_spoofing(image_data: bytes, face_boxes: List[Dict] = None) -> AntiSpoofingResponse:
    """Process anti-spoofing detection on image"""
    global metrics
    start_time = time.time()
    
    try:
        # Decode image
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        detections = []
        
        # If no face boxes provided, assume entire image is a face
        if face_boxes is None:
            face_boxes = [{"x": 0, "y": 0, "width": image.shape[1], "height": image.shape[0]}]
        
        for i, box in enumerate(face_boxes):
            try:
                # Extract face region
                x, y, w, h = int(box["x"]), int(box["y"]), int(box["width"]), int(box["height"])
                face_image = image[y:y+h, x:x+w]
                
                if face_image.size == 0:
                    continue
                
                # Detect spoofing
                is_real, confidence, spoof_score = detect_spoofing(face_image)
                
                # Calculate quality score
                quality_score = calculate_quality_score(face_image)
                
                detections.append(SpoofingDetection(
                    face_id=i,
                    is_real=is_real,
                    confidence=float(confidence),
                    spoof_score=float(spoof_score),
                    quality_score=float(quality_score)
                ))
                
            except Exception as e:
                logger.error(f"Error processing face {i}: {e}")
                continue
        
        processing_time = time.time() - start_time
        
        # Update metrics
        metrics["total_requests"] += 1
        metrics["total_faces_processed"] += len(detections)
        metrics["total_real_faces"] += sum(1 for d in detections if d.is_real)
        metrics["total_spoofed_faces"] += sum(1 for d in detections if not d.is_real)
        metrics["total_processing_time"] += processing_time
        
        return AntiSpoofingResponse(
            detections=detections,
            processing_time=processing_time,
            total_faces=len(detections),
            model_info="Silent Face Anti-Spoofing"
        )
        
    except Exception as e:
        logger.error(f"Error in anti-spoofing: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/detect", response_model=AntiSpoofingResponse)
async def detect_spoofing_endpoint(
    file: UploadFile = File(...),
    face_boxes: Optional[str] = Form(None)
):
    """Detect spoofing in uploaded image"""
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        image_data = await file.read()
        
        # Parse face boxes if provided
        boxes = None
        if face_boxes:
            boxes = json.loads(face_boxes)
        
        result = await process_anti_spoofing(image_data, boxes)
        return result
    except Exception as e:
        logger.error(f"Error in detect endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect/batch", response_model=BatchAntiSpoofingResponse)
async def detect_spoofing_batch(files: List[UploadFile] = File(...)):
    """Detect spoofing in multiple images"""
    if len(files) > 5:  # Limit batch size for anti-spoofing
        raise HTTPException(status_code=400, detail="Maximum 5 images per batch")
    
    start_time = time.time()
    results = []
    total_faces = 0
    total_real = 0
    total_spoofed = 0
    
    for file in files:
        if not file.content_type.startswith('image/'):
            continue
        
        try:
            image_data = await file.read()
            result = await process_anti_spoofing(image_data)
            results.append(result)
            total_faces += len(result.detections)
            total_real += sum(1 for d in result.detections if d.is_real)
            total_spoofed += sum(1 for d in result.detections if not d.is_real)
        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {e}")
            continue
    
    total_time = time.time() - start_time
    
    return BatchAntiSpoofingResponse(
        results=results,
        total_processing_time=total_time,
        total_faces_processed=total_faces,
        total_real_faces=total_real,
        total_spoofed_faces=total_spoofed
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
        service="Anti-Spoofing",
        model_loaded=antispoof_model is not None,
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
        total_faces_processed=metrics["total_faces_processed"],
        total_real_faces=metrics["total_real_faces"],
        total_spoofed_faces=metrics["total_spoofed_faces"],
        average_processing_time=avg_processing_time,
        gpu_utilization=gpu_utilization,
        memory_usage=memory_info
    )

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Anti-Spoofing Service", "status": "running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)
