"""
YOLOv10n Face Detection Service
FastAPI microservice for face detection using YOLOv10n model with GPU acceleration
"""

import os
import cv2
import numpy as np
import onnxruntime as ort
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
import psutil
import GPUtil
import time
from contextlib import asynccontextmanager
import asyncio
from PIL import Image
import io
import base64
import os

# Suppress ONNX Runtime warnings about performance
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="onnxruntime")

# Set ONNX Runtime environment variables to suppress warnings
os.environ['ORT_DISABLE_ALL_OPTIMIZATION_WARNINGS'] = '1'
os.environ['ORT_LOG_LEVEL'] = '3'  # ERROR level only

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# GPU Memory Manager
class GPUMemoryManager:
    def __init__(self, memory_limit_mb: int = 800):
        self.memory_limit_mb = memory_limit_mb
        self.session = None
    
    def setup_gpu_session(self, model_path: str):
        """Setup ONNX Runtime session with optimized GPU memory management"""
        try:
            providers = []
            if ort.get_available_providers().__contains__('CUDAExecutionProvider'):
                # Optimized CUDA provider settings to minimize memcpy operations
                cuda_provider = ('CUDAExecutionProvider', {
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': self.memory_limit_mb * 1024 * 1024,
                    'cudnn_conv_algo_search': 'HEURISTIC',
                    'do_copy_in_default_stream': False,
                    'cudnn_conv1d_pad_to_nc1d': True,
                    'enable_cuda_graph': False,  # Disable CUDA graph to avoid partitioning issues
                    'tunable_op_enable': True,
                    'tunable_op_tuning_enable': True,
                })
                providers.append(cuda_provider)
            
            providers.append('CPUExecutionProvider')
              # Optimized session options to reduce memory copies
            session_options = ort.SessionOptions()
            session_options.enable_mem_pattern = True
            session_options.enable_cpu_mem_arena = True
            session_options.enable_mem_reuse = True
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            session_options.inter_op_num_threads = 1
            session_options.intra_op_num_threads = 0
            session_options.log_severity_level = 3  # Suppress all warnings (ERROR level only)
            
            # Add session configuration to optimize CUDA graphs
            session_options.add_session_config_entry('session.use_env_allocators', '1')
            session_options.add_session_config_entry('session.use_device_allocator_for_initializers', '1')
            
            self.session = ort.InferenceSession(
                model_path,
                sess_options=session_options,
                providers=providers
            )
            
            logger.info(f"Model loaded successfully with providers: {self.session.get_providers()}")
            
            return self.session
            
        except Exception as e:
            logger.error(f"Failed to setup GPU session: {e}")
            raise
    
    def _warmup_model(self):
        """Warm up the model with dummy input to initialize CUDA graphs"""
        try:
            if self.session:
                # Create dummy input matching expected shape
                input_name = self.session.get_inputs()[0].name
                input_shape = self.session.get_inputs()[0].shape
                
                # Handle dynamic shapes
                if isinstance(input_shape[0], str) or input_shape[0] is None:
                    input_shape[0] = 1
                if isinstance(input_shape[2], str) or input_shape[2] is None:
                    input_shape[2] = 640
                if isinstance(input_shape[3], str) or input_shape[3] is None:
                    input_shape[3] = 640
                
                dummy_input = np.random.rand(*input_shape).astype(np.float32)
                
                # Run inference a few times to warm up CUDA graphs
                for _ in range(3):
                    self.session.run(None, {input_name: dummy_input})
                    
                logger.info("Model warmed up successfully - CUDA graphs initialized")
                
        except Exception as e:
            logger.warning(f"Model warmup failed (non-critical): {e}")

# Global variables
gpu_manager = GPUMemoryManager(memory_limit_mb=800)
model_session = None

# Pydantic models
class FaceDetection(BaseModel):
    x: float
    y: float
    width: float
    height: float
    confidence: float

class DetectionResponse(BaseModel):
    faces: List[FaceDetection]
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
    global model_session
    
    # Startup
    logger.info("Starting YOLOv10n Face Detection Service...")
    
    model_path = "/app/models/yolov10n-face.onnx"
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        model_session = gpu_manager.setup_gpu_session(model_path)
        logger.info("YOLOv10n model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down YOLOv10n Face Detection Service...")
    if model_session:
        del model_session

app = FastAPI(
    title="YOLOv10n Face Detection Service",
    description="AI microservice for face detection using YOLOv10n model",
    version="1.0.0",
    lifespan=lifespan
)

def preprocess_image(image: np.ndarray, input_size: tuple = (640, 640)):
    """Preprocess image for YOLOv10n model"""
    original_height, original_width = image.shape[:2]
    
    # Resize image while maintaining aspect ratio
    scale = min(input_size[0] / original_width, input_size[1] / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    
    resized_image = cv2.resize(image, (new_width, new_height))
    
    # Pad image to input size
    padded_image = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
    y_offset = (input_size[1] - new_height) // 2
    x_offset = (input_size[0] - new_width) // 2
    padded_image[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_image
    
    # Convert to RGB and normalize
    rgb_image = cv2.cvtColor(padded_image, cv2.COLOR_BGR2RGB)
    normalized_image = rgb_image.astype(np.float32) / 255.0
    
    # Transpose to NCHW format
    input_tensor = np.transpose(normalized_image, (2, 0, 1))
    input_tensor = np.expand_dims(input_tensor, axis=0)
    
    return input_tensor, scale, x_offset, y_offset

def postprocess_detections(outputs, scale, x_offset, y_offset, original_width, original_height, conf_threshold=0.5):
    """Post-process YOLOv10n outputs to extract face detections"""
    detections = []
    
    if len(outputs) > 0:
        output = outputs[0]
        
        # YOLOv10n output format: [batch, num_detections, 6] where 6 = [x1, y1, x2, y2, confidence, class]
        if len(output.shape) == 3:
            for detection in output[0]:
                if len(detection) >= 5:
                    x1, y1, x2, y2, confidence = detection[:5]
                    
                    if confidence > conf_threshold:
                        # Convert back to original image coordinates
                        x1 = (x1 - x_offset) / scale
                        y1 = (y1 - y_offset) / scale
                        x2 = (x2 - x_offset) / scale
                        y2 = (y2 - y_offset) / scale
                        
                        # Ensure coordinates are within image bounds
                        x1 = max(0, min(x1, original_width))
                        y1 = max(0, min(y1, original_height))
                        x2 = max(0, min(x2, original_width))
                        y2 = max(0, min(y2, original_height))
                        
                        width = x2 - x1
                        height = y2 - y1
                        
                        if width > 0 and height > 0:
                            detections.append(FaceDetection(
                                x=float(x1),
                                y=float(y1),
                                width=float(width),
                                height=float(height),
                                confidence=float(confidence)
                            ))
    
    return detections

async def process_image(image_data: bytes) -> DetectionResponse:
    """Process a single image for face detection"""
    global metrics
    start_time = time.time()
    
    try:
        # Decode image
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        original_height, original_width = image.shape[:2]
        
        # Preprocess image
        input_tensor, scale, x_offset, y_offset = preprocess_image(image)
        
        # Run inference
        input_name = model_session.get_inputs()[0].name
        outputs = model_session.run(None, {input_name: input_tensor})
        
        # Post-process results
        detections = postprocess_detections(
            outputs, scale, x_offset, y_offset, 
            original_width, original_height
        )
        
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
            model_info="YOLOv10n Face Detection"
        )
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/detect", response_model=DetectionResponse)
async def detect_faces(file: UploadFile = File(...)):
    """Detect faces in uploaded image"""
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
    """Detect faces in multiple images"""
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 images per batch")
    
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
    # Check both ONNX Runtime and PyTorch CUDA availability
    onnx_gpu_available = 'CUDAExecutionProvider' in ort.get_available_providers()
    torch_gpu_available = False
    
    try:
        torch_gpu_available = torch.cuda.is_available()
    except Exception:
        pass
    
    gpu_available = onnx_gpu_available or torch_gpu_available
    
    memory_info = {
        "ram_percent": psutil.virtual_memory().percent,
        "ram_available_gb": psutil.virtual_memory().available / (1024**3)
    }
    
    # Add PyTorch GPU memory info if available
    if torch_gpu_available:
        try:
            memory_info.update({
                "gpu_memory_allocated_mb": torch.cuda.memory_allocated() / (1024**2),
                "gpu_memory_cached_mb": torch.cuda.memory_reserved() / (1024**2)
            })
        except Exception:
            pass
    
    # Add GPUtil info if available
    if gpu_available:
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
        service="YOLOv10n Face Detection",
        model_loaded=model_session is not None,
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
    return {"message": "YOLOv10n Face Detection Service", "status": "running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
