"""
Gender and Age Detection Service
FastAPI microservice for gender and age prediction using deep learning models
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

# Gender and Age prediction network
class GenderAgeNet(nn.Module):
    def __init__(self):
        super(GenderAgeNet, self).__init__()
        
        # Shared feature extractor
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Gender classifier (binary: male/female)
        self.gender_classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 2)  # Male, Female
        )
        
        # Age regressor
        self.age_regressor = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 1)  # Single age value
        )
    
    def forward(self, x):
        features = self.features(x)
        features_flat = features.view(features.size(0), -1)
        
        gender_logits = self.gender_classifier(features_flat)
        age_pred = self.age_regressor(features_flat)
        
        return gender_logits, age_pred

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
        """Setup gender and age model"""
        try:
            self.model = GenderAgeNet()
            
            # Load pre-trained weights if available
            model_path = "/app/models/gender_age.pth"
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint)
                logger.info("Loaded pre-trained gender-age model")
            else:
                logger.warning("No pre-trained model found, using randomly initialized weights")
            
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("Gender-Age model loaded successfully")
            return self.model
            
        except Exception as e:
            logger.error(f"Failed to setup gender-age model: {e}")
            raise

# Global variables
gpu_manager = GPUMemoryManager(memory_limit_mb=800)
gender_age_model = None

# Age group mapping
AGE_GROUPS = [
    "0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"
]

def get_age_group(age: float) -> str:
    """Convert age to age group"""
    if age < 3:
        return "0-2"
    elif age < 10:
        return "3-9"
    elif age < 20:
        return "10-19"
    elif age < 30:
        return "20-29"
    elif age < 40:
        return "30-39"
    elif age < 50:
        return "40-49"
    elif age < 60:
        return "50-59"
    elif age < 70:
        return "60-69"
    else:
        return "70+"

# Pydantic models
class GenderAgePrediction(BaseModel):
    face_id: int
    gender: str
    gender_confidence: float
    age: float
    age_group: str
    age_confidence: float

class GenderAgeResponse(BaseModel):
    predictions: List[GenderAgePrediction]
    processing_time: float
    total_faces: int
    model_info: str

class BatchGenderAgeResponse(BaseModel):
    results: List[GenderAgeResponse]
    total_processing_time: float
    total_faces_processed: int
    gender_distribution: Dict[str, int]
    age_distribution: Dict[str, int]

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
    gender_distribution: Dict[str, int]
    age_distribution: Dict[str, int]
    average_processing_time: float
    gpu_utilization: float
    memory_usage: Dict[str, Any]

# Global metrics
metrics = {
    "total_requests": 0,
    "total_faces_processed": 0,
    "gender_distribution": {"male": 0, "female": 0},
    "age_distribution": {group: 0 for group in AGE_GROUPS},
    "total_processing_time": 0.0,
    "start_time": time.time()
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global gender_age_model
    
    # Startup
    logger.info("Starting Gender and Age Detection Service...")
    
    try:
        device = gpu_manager.setup_gpu_device()
        gender_age_model = gpu_manager.setup_model()
        logger.info("Gender-Age model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Gender and Age Detection Service...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app = FastAPI(
    title="Gender and Age Detection Service",
    description="AI microservice for gender and age prediction",
    version="1.0.0",
    lifespan=lifespan
)

def preprocess_face(face_image: np.ndarray, input_size: tuple = (224, 224)) -> torch.Tensor:
    """Preprocess face image for gender-age model"""
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

def predict_gender_age(face_image: np.ndarray) -> Tuple[str, float, float, float]:
    """Predict gender and age from face image"""
    try:
        face_tensor = preprocess_face(face_image)
        face_tensor = face_tensor.to(gpu_manager.device)
        
        with torch.no_grad():
            gender_logits, age_pred = gender_age_model(face_tensor)
            
            # Process gender prediction
            gender_probs = F.softmax(gender_logits, dim=1)
            gender_confidence, gender_idx = torch.max(gender_probs, 1)
            gender = "male" if gender_idx.item() == 0 else "female"
            
            # Process age prediction
            age = max(0, min(age_pred.item(), 100))  # Clamp age between 0-100
            
            # Age confidence based on how close prediction is to realistic range
            age_confidence = 1.0 - min(abs(age - 35) / 35, 1.0)  # Peak confidence at age 35
            
        return gender, float(gender_confidence), float(age), float(age_confidence)
        
    except Exception as e:
        logger.error(f"Error in gender-age prediction: {e}")
        return "unknown", 0.0, 0.0, 0.0

async def process_gender_age(image_data: bytes, face_boxes: List[Dict] = None) -> GenderAgeResponse:
    """Process gender and age prediction on image"""
    global metrics
    start_time = time.time()
    
    try:
        # Decode image
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        predictions = []
        
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
                
                # Predict gender and age
                gender, gender_conf, age, age_conf = predict_gender_age(face_image)
                age_group = get_age_group(age)
                
                predictions.append(GenderAgePrediction(
                    face_id=i,
                    gender=gender,
                    gender_confidence=gender_conf,
                    age=age,
                    age_group=age_group,
                    age_confidence=age_conf
                ))
                
                # Update metrics
                if gender in metrics["gender_distribution"]:
                    metrics["gender_distribution"][gender] += 1
                if age_group in metrics["age_distribution"]:
                    metrics["age_distribution"][age_group] += 1
                
            except Exception as e:
                logger.error(f"Error processing face {i}: {e}")
                continue
        
        processing_time = time.time() - start_time
        
        # Update metrics
        metrics["total_requests"] += 1
        metrics["total_faces_processed"] += len(predictions)
        metrics["total_processing_time"] += processing_time
        
        return GenderAgeResponse(
            predictions=predictions,
            processing_time=processing_time,
            total_faces=len(predictions),
            model_info="CNN Gender and Age Prediction"
        )
        
    except Exception as e:
        logger.error(f"Error in gender-age prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/predict", response_model=GenderAgeResponse)
async def predict_gender_age_endpoint(
    file: UploadFile = File(...),
    face_boxes: Optional[str] = Form(None)
):
    """Predict gender and age in uploaded image"""
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        image_data = await file.read()
        
        # Parse face boxes if provided
        boxes = None
        if face_boxes:
            boxes = json.loads(face_boxes)
        
        result = await process_gender_age(image_data, boxes)
        return result
    except Exception as e:
        logger.error(f"Error in predict endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch", response_model=BatchGenderAgeResponse)
async def predict_gender_age_batch(files: List[UploadFile] = File(...)):
    """Predict gender and age in multiple images"""
    if len(files) > 8:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 8 images per batch")
    
    start_time = time.time()
    results = []
    total_faces = 0
    gender_dist = {"male": 0, "female": 0}
    age_dist = {group: 0 for group in AGE_GROUPS}
    
    for file in files:
        if not file.content_type.startswith('image/'):
            continue
        
        try:
            image_data = await file.read()
            result = await process_gender_age(image_data)
            results.append(result)
            total_faces += len(result.predictions)
            
            # Aggregate distributions
            for pred in result.predictions:
                if pred.gender in gender_dist:
                    gender_dist[pred.gender] += 1
                if pred.age_group in age_dist:
                    age_dist[pred.age_group] += 1
                    
        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {e}")
            continue
    
    total_time = time.time() - start_time
    
    return BatchGenderAgeResponse(
        results=results,
        total_processing_time=total_time,
        total_faces_processed=total_faces,
        gender_distribution=gender_dist,
        age_distribution=age_dist
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
        service="Gender and Age Detection",
        model_loaded=gender_age_model is not None,
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
        gender_distribution=metrics["gender_distribution"],
        age_distribution=metrics["age_distribution"],
        average_processing_time=avg_processing_time,
        gpu_utilization=gpu_utilization,
        memory_usage=memory_info
    )

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Gender and Age Detection Service", "status": "running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8006)
