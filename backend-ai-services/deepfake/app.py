"""
Deepfake Detection Service
FastAPI microservice for deepfake detection using state-of-the-art models
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
import onnxruntime as ort

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Deepfake Detection Network (EfficientNet-based)
class DeepfakeDetectionNet(nn.Module):
    def __init__(self, num_classes=2):
        super(DeepfakeDetectionNet, self).__init__()
        
        # Feature extractor (simplified EfficientNet-like architecture)
        self.features = nn.Sequential(
            # Stem
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Block 1
            self._make_block(32, 64, 3, stride=1, expand_ratio=1),
            self._make_block(64, 64, 3, stride=2, expand_ratio=6),
            
            # Block 2
            self._make_block(64, 128, 5, stride=2, expand_ratio=6),
            self._make_block(128, 128, 5, stride=1, expand_ratio=6),
            
            # Block 3
            self._make_block(128, 256, 3, stride=2, expand_ratio=6),
            self._make_block(256, 256, 3, stride=1, expand_ratio=6),
            
            # Head
            nn.Conv2d(256, 512, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)  # Real, Fake
        )
        
        # Attention mechanism for temporal consistency
        self.attention = nn.MultiheadAttention(512, 8, dropout=0.1)
        
    def _make_block(self, in_channels, out_channels, kernel_size, stride=1, expand_ratio=1):
        """Create an inverted residual block"""
        expanded_channels = in_channels * expand_ratio
        
        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, expanded_channels, 1),
                nn.BatchNorm2d(expanded_channels),
                nn.ReLU(inplace=True)
            ])
        
        layers.extend([
            nn.Conv2d(expanded_channels, expanded_channels, kernel_size, 
                     stride=stride, padding=kernel_size//2, groups=expanded_channels),
            nn.BatchNorm2d(expanded_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(expanded_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        ])
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        features = self.features(x)
        features = features.view(features.size(0), -1)
        output = self.classifier(features)
        return output

# GPU Memory Manager
class GPUMemoryManager:
    def __init__(self, memory_limit_mb: int = 1400):
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
        """Setup deepfake detection model using ONNX"""
        try:
            # Setup ONNX Runtime with GPU support
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
              # Model path - try different models in order of preference
            model_candidates = [
                "/app/models/model.onnx",       # Full precision model (highest accuracy)
                "/app/models/model_fp16.onnx",  # Best balance of speed and accuracy
                "/app/models/model_int8.onnx",  # Quantized fallback
                "/app/models/model_q4.onnx"     # Most compressed fallback
            ]
            
            model_path = None
            for candidate in model_candidates:
                if os.path.exists(candidate):
                    model_path = candidate
                    break
            
            if model_path is None:
                logger.warning("No ONNX model found, using randomly initialized PyTorch weights")
                # Fallback to PyTorch model
                self.model = DeepfakeDetectionNet(num_classes=2)
                self.model.to(self.device)
                self.model.eval()
                self.is_onnx = False
                return self.model
            
            # Load ONNX model
            logger.info(f"Loading ONNX model: {model_path}")
            self.onnx_session = ort.InferenceSession(model_path, providers=providers)
            self.is_onnx = True
            
            # Get input details
            input_details = self.onnx_session.get_inputs()[0]
            self.input_name = input_details.name
            self.input_shape = input_details.shape
            
            logger.info(f"ONNX deepfake detection model loaded successfully")
            logger.info(f"Model input shape: {self.input_shape}")
            logger.info(f"Available providers: {self.onnx_session.get_providers()}")
            
            return self.onnx_session
            
        except Exception as e:
            logger.error(f"Failed to setup deepfake detection model: {e}")
            raise

# Global variables
gpu_manager = GPUMemoryManager(memory_limit_mb=1400)
deepfake_model = None

# Pydantic models
class DeepfakeDetection(BaseModel):
    face_id: int
    is_real: bool
    confidence: float
    deepfake_score: float
    manipulation_type: Optional[str] = None
    quality_metrics: Dict[str, float]

class DeepfakeResponse(BaseModel):
    detections: List[DeepfakeDetection]
    processing_time: float
    total_faces: int
    overall_authenticity_score: float
    model_info: str

class BatchDeepfakeResponse(BaseModel):
    results: List[DeepfakeResponse]
    total_processing_time: float
    total_faces_processed: int
    total_real_faces: int
    total_fake_faces: int
    authenticity_distribution: Dict[str, int]

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
    total_fake_faces: int
    average_processing_time: float
    average_authenticity_score: float
    gpu_utilization: float
    memory_usage: Dict[str, Any]

# Global metrics
metrics = {
    "total_requests": 0,
    "total_faces_processed": 0,
    "total_real_faces": 0,
    "total_fake_faces": 0,
    "total_authenticity_score": 0.0,
    "total_processing_time": 0.0,
    "start_time": time.time()
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global deepfake_model
    
    # Startup
    logger.info("Starting Deepfake Detection Service...")
    
    try:
        device = gpu_manager.setup_gpu_device()
        deepfake_model = gpu_manager.setup_model()
        logger.info("Deepfake detection model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Deepfake Detection Service...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app = FastAPI(
    title="Deepfake Detection Service",
    description="AI microservice for deepfake and face manipulation detection",
    version="1.0.0",
    lifespan=lifespan
)

def preprocess_face(face_image: np.ndarray, input_size: tuple = (224, 224)) -> torch.Tensor:
    """Preprocess face image for deepfake detection model"""
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

def calculate_quality_metrics(face_image: np.ndarray) -> Dict[str, float]:
    """Calculate various quality metrics for deepfake detection"""
    try:
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        
        metrics_dict = {}
        
        # Sharpness (Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        metrics_dict["sharpness"] = min(laplacian_var / 500.0, 1.0)
        
        # Contrast (standard deviation)
        metrics_dict["contrast"] = gray.std() / 255.0
        
        # Brightness uniformity
        mean_brightness = np.mean(gray)
        brightness_std = np.std(gray)
        metrics_dict["brightness_uniformity"] = 1.0 - (brightness_std / 255.0)
        
        # Edge density (Canny edges)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        metrics_dict["edge_density"] = edge_density
        
        # Texture consistency (Local Binary Pattern variance)
        lbp_var = np.var(gray)
        metrics_dict["texture_consistency"] = min(lbp_var / 10000.0, 1.0)
        
        # Compression artifacts (DCT analysis - simplified)
        dct = cv2.dct(gray.astype(np.float32))
        high_freq_energy = np.sum(np.abs(dct[gray.shape[0]//2:, gray.shape[1]//2:]))
        total_energy = np.sum(np.abs(dct))
        metrics_dict["compression_artifacts"] = 1.0 - (high_freq_energy / total_energy)
        
        return metrics_dict
        
    except Exception as e:
        logger.error(f"Error calculating quality metrics: {e}")
        return {
            "sharpness": 0.5,
            "contrast": 0.5,
            "brightness_uniformity": 0.5,
            "edge_density": 0.5,
            "texture_consistency": 0.5,
            "compression_artifacts": 0.5
        }

def detect_manipulation_type(quality_metrics: Dict[str, float], deepfake_score: float) -> Optional[str]:
    """Detect type of manipulation based on quality metrics"""
    if deepfake_score < 0.3:  # Likely real
        return None
    
    # Analyze quality metrics to determine manipulation type
    if quality_metrics["compression_artifacts"] < 0.3:
        return "compression_based"
    elif quality_metrics["texture_consistency"] < 0.4:
        return "texture_synthesis"
    elif quality_metrics["edge_density"] > 0.8:
        return "edge_enhancement"
    elif quality_metrics["sharpness"] < 0.3:
        return "blur_based"
    else:
        return "ai_generated"

def detect_deepfake(face_image: np.ndarray) -> Tuple[bool, float, Optional[str], Dict[str, float]]:
    """Detect if face is a deepfake using ONNX or PyTorch model"""
    try:        # Calculate quality metrics first
        quality_metrics = calculate_quality_metrics(face_image)
        
        if hasattr(gpu_manager, 'is_onnx') and gpu_manager.is_onnx:
            # Use ONNX model
            face_tensor = preprocess_face(face_image)
            face_numpy = face_tensor.cpu().numpy().astype(np.float32)  # Ensure float32 for ONNX
            
            # Run ONNX inference
            outputs = gpu_manager.onnx_session.run(None, {gpu_manager.input_name: face_numpy})
            
            # Apply sigmoid activation to convert logits to probabilities
            logits = outputs[0][0]  # Get first batch, first output
            probabilities = 1.0 / (1.0 + np.exp(-logits))  # Sigmoid activation
            
            # Handle binary classification (single output) vs multi-class
            if len(probabilities.shape) == 0 or probabilities.size == 1:
                # Binary classification with single output
                fake_prob = float(probabilities)
                real_prob = 1.0 - fake_prob
            else:
                # Multi-class classification
                real_prob = float(probabilities[0])
                fake_prob = float(probabilities[1])
            
            is_real = real_prob > fake_prob
            confidence = max(real_prob, fake_prob)
            deepfake_score = fake_prob
            
        else:
            # Fallback to PyTorch model
            face_tensor = preprocess_face(face_image)
            face_tensor = face_tensor.to(gpu_manager.device)
            
            with torch.no_grad():
                outputs = gpu_manager.model(face_tensor)
                probabilities = F.softmax(outputs, dim=1)
                
                # Assuming class 0 = real, class 1 = fake
                real_prob = probabilities[0, 0].item()
                fake_prob = probabilities[0, 1].item()
                
                is_real = real_prob > fake_prob
                confidence = max(real_prob, fake_prob)
                deepfake_score = fake_prob
        
        # Detect manipulation type
        manipulation_type = detect_manipulation_type(quality_metrics, deepfake_score)
        
        return is_real, confidence, manipulation_type, quality_metrics
        
    except Exception as e:
        logger.error(f"Error in deepfake detection: {e}")
        return True, 0.5, None, {}

async def process_deepfake_detection(image_data: bytes, face_boxes: List[Dict] = None) -> DeepfakeResponse:
    """Process deepfake detection on image"""
    global metrics
    start_time = time.time()
    
    try:
        # Decode image
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        detections = []
        authenticity_scores = []
        
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
                
                # Detect deepfake
                is_real, confidence, manipulation_type, quality_metrics = detect_deepfake(face_image)
                deepfake_score = 1.0 - confidence if is_real else confidence
                
                detections.append(DeepfakeDetection(
                    face_id=i,
                    is_real=is_real,
                    confidence=float(confidence),
                    deepfake_score=float(deepfake_score),
                    manipulation_type=manipulation_type,
                    quality_metrics=quality_metrics
                ))
                
                authenticity_scores.append(1.0 - deepfake_score)
                
            except Exception as e:
                logger.error(f"Error processing face {i}: {e}")
                continue
        
        # Calculate overall authenticity score
        overall_authenticity = np.mean(authenticity_scores) if authenticity_scores else 0.5
        
        processing_time = time.time() - start_time
        
        # Update metrics
        metrics["total_requests"] += 1
        metrics["total_faces_processed"] += len(detections)
        metrics["total_real_faces"] += sum(1 for d in detections if d.is_real)
        metrics["total_fake_faces"] += sum(1 for d in detections if not d.is_real)
        metrics["total_authenticity_score"] += overall_authenticity
        metrics["total_processing_time"] += processing_time
        
        return DeepfakeResponse(
            detections=detections,
            processing_time=processing_time,
            total_faces=len(detections),
            overall_authenticity_score=float(overall_authenticity),
            model_info="EfficientNet-based Deepfake Detection"
        )
        
    except Exception as e:
        logger.error(f"Error in deepfake detection: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/detect", response_model=DeepfakeResponse)
async def detect_deepfake_endpoint(
    file: UploadFile = File(...),
    face_boxes: Optional[str] = Form(None)
):
    """Detect deepfakes in uploaded image"""
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        image_data = await file.read()
        
        # Parse face boxes if provided
        boxes = None
        if face_boxes:
            boxes = json.loads(face_boxes)
        
        result = await process_deepfake_detection(image_data, boxes)
        return result
    except Exception as e:
        logger.error(f"Error in detect endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect/batch", response_model=BatchDeepfakeResponse)
async def detect_deepfake_batch(files: List[UploadFile] = File(...)):
    """Detect deepfakes in multiple images"""
    if len(files) > 5:  # Limit batch size for deepfake detection
        raise HTTPException(status_code=400, detail="Maximum 5 images per batch")
    
    start_time = time.time()
    results = []
    total_faces = 0
    total_real = 0
    total_fake = 0
    authenticity_dist = {"high": 0, "medium": 0, "low": 0}
    
    for file in files:
        if not file.content_type.startswith('image/'):
            continue
        
        try:
            image_data = await file.read()
            result = await process_deepfake_detection(image_data)
            results.append(result)
            total_faces += len(result.detections)
            total_real += sum(1 for d in result.detections if d.is_real)
            total_fake += sum(1 for d in result.detections if not d.is_real)
            
            # Categorize authenticity
            if result.overall_authenticity_score > 0.7:
                authenticity_dist["high"] += 1
            elif result.overall_authenticity_score > 0.4:
                authenticity_dist["medium"] += 1
            else:
                authenticity_dist["low"] += 1
                
        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {e}")
            continue
    
    total_time = time.time() - start_time
    
    return BatchDeepfakeResponse(
        results=results,
        total_processing_time=total_time,
        total_faces_processed=total_faces,
        total_real_faces=total_real,
        total_fake_faces=total_fake,
        authenticity_distribution=authenticity_dist
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
        service="Deepfake Detection",
        model_loaded=deepfake_model is not None,
        gpu_available=gpu_available,
        memory_usage=memory_info,
        uptime=uptime
    )

@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get service metrics"""
    avg_processing_time = (metrics["total_processing_time"] / metrics["total_requests"] 
                          if metrics["total_requests"] > 0 else 0)
    
    avg_authenticity_score = (metrics["total_authenticity_score"] / metrics["total_requests"] 
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
        total_fake_faces=metrics["total_fake_faces"],
        average_processing_time=avg_processing_time,
        average_authenticity_score=avg_authenticity_score,
        gpu_utilization=gpu_utilization,
        memory_usage=memory_info
    )

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Deepfake Detection Service", "status": "running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
