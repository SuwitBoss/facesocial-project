"""
YOLOv10n Face Detection Service - Enhanced Response Format  
FastAPI microservice for face detection using YOLOv10n model with standardized response
"""

import os
import cv2
import numpy as np
import onnxruntime as ort
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, status, Form
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
import json

# Suppress ONNX Runtime warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="onnxruntime")

os.environ['ORT_DISABLE_ALL_OPTIMIZATION_WARNINGS'] = '1'
os.environ['ORT_LOG_LEVEL'] = '3'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GPUMemoryManager:
    """GPU Memory Manager with enhanced monitoring"""
    
    def __init__(self, memory_limit_mb: int = 800):
        self.memory_limit_mb = memory_limit_mb
        self.session = None
    
    def setup_gpu_session(self, model_path: str):
        """Setup ONNX Runtime session with optimized GPU memory management"""
        try:
            providers = []
            if ort.get_available_providers().__contains__('CUDAExecutionProvider'):
                cuda_provider = ('CUDAExecutionProvider', {
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': self.memory_limit_mb * 1024 * 1024,
                    'cudnn_conv_algo_search': 'HEURISTIC',
                    'do_copy_in_default_stream': False,
                    'cudnn_conv1d_pad_to_nc1d': True,
                    'enable_cuda_graph': False,
                    'tunable_op_enable': True,
                    'tunable_op_tuning_enable': True,
                })
                providers.append(cuda_provider)
            
            providers.append('CPUExecutionProvider')
            
            session_options = ort.SessionOptions()
            session_options.enable_mem_pattern = True
            session_options.enable_cpu_mem_arena = True
            session_options.enable_mem_reuse = True
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            session_options.inter_op_num_threads = 1
            session_options.intra_op_num_threads = 0
            session_options.log_severity_level = 3
            
            session_options.add_session_config_entry('session.use_env_allocators', '1')
            session_options.add_session_config_entry('session.use_device_allocator_for_initializers', '1')
            
            self.session = ort.InferenceSession(
                model_path,
                sess_options=session_options,
                providers=providers
            )
            
            logger.info(f"YOLO model loaded successfully with providers: {self.session.get_providers()}")
            return self.session
            
        except Exception as e:
            logger.error(f"Failed to setup GPU session: {e}")
            raise

# Enhanced Face Detection with Quality Assessment
class EnhancedYOLODetector:
    """Enhanced YOLO detector with quality assessment and standardized response"""
    
    def __init__(self, session: ort.InferenceSession):
        self.session = session
        self.input_name = session.get_inputs()[0].name
        self.output_names = [o.name for o in session.get_outputs()]
        self.input_size = (640, 640)
        self.confidence_threshold = 0.25
        self.iou_threshold = 0.45
    
    def detect_faces(self, image: np.ndarray) -> Dict[str, Any]:
        """Enhanced face detection with quality metrics"""
        try:
            start_time = time.time()
            original_height, original_width = image.shape[:2]
            
            # Preprocess image
            input_tensor, scale_x, scale_y = self._preprocess_image(image)
            
            # Run inference
            inference_start = time.time()
            outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
            inference_time = (time.time() - inference_start) * 1000
            
            # Post-process results
            faces = self._postprocess_detections(outputs, scale_x, scale_y, original_width, original_height)
            
            # Add quality assessment to each face
            for i, face in enumerate(faces):
                face_crop = self._extract_face_crop(image, face)
                face["quality_score"] = self._assess_face_quality(face_crop)
                face["face_id"] = f"yolo_face_{i}"
                face["face_index"] = i
                face["detection_method"] = "yolo"
            
            processing_time = (time.time() - start_time) * 1000
            
            return {
                "faces": faces,
                "total_faces": len(faces),
                "processing_time": processing_time / 1000,  # Convert to seconds
                "inference_time_ms": inference_time,
                "image_info": {
                    "width": original_width,
                    "height": original_height,
                    "channels": image.shape[2] if len(image.shape) > 2 else 1
                },
                "model_info": {
                    "detector": "yolov10n",
                    "version": "1.0",
                    "confidence_threshold": self.confidence_threshold,
                    "iou_threshold": self.iou_threshold
                }
            }
            
        except Exception as e:
            logger.error(f"YOLO detection failed: {e}")
            return {
                "faces": [],
                "total_faces": 0,
                "processing_time": 0.0,
                "error": str(e)
            }
    
    def _preprocess_image(self, image: np.ndarray):
        """Preprocess image for YOLO model"""
        original_height, original_width = image.shape[:2]
        
        # Calculate scale
        scale = min(self.input_size[0] / original_width, self.input_size[1] / original_height)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        
        resized_image = cv2.resize(image, (new_width, new_height))
        
        # Pad image
        padded_image = np.zeros((self.input_size[1], self.input_size[0], 3), dtype=np.uint8)
        y_offset = (self.input_size[1] - new_height) // 2
        x_offset = (self.input_size[0] - new_width) // 2
        padded_image[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_image
        
        # Convert to RGB and normalize
        rgb_image = cv2.cvtColor(padded_image, cv2.COLOR_BGR2RGB)
        normalized_image = rgb_image.astype(np.float32) / 255.0
        
        # Transpose to NCHW format
        input_tensor = np.transpose(normalized_image, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        return input_tensor, scale, scale
    
    def _postprocess_detections(self, outputs, scale_x, scale_y, original_width, original_height):
        """Post-process YOLO detections with enhanced format"""
        faces = []
        
        if len(outputs) > 0:
            output = outputs[0]
            
            if len(output.shape) == 3:
                for detection in output[0]:
                    if len(detection) >= 5:
                        x1, y1, x2, y2, confidence = detection[:5]
                        
                        if confidence > self.confidence_threshold:
                            # Scale back to original coordinates
                            x1 = max(0, x1 / scale_x)
                            y1 = max(0, y1 / scale_y)
                            x2 = min(original_width, x2 / scale_x)
                            y2 = min(original_height, y2 / scale_y)
                            
                            width = x2 - x1
                            height = y2 - y1
                            
                            if width > 0 and height > 0:
                                # Create enhanced face object
                                face_obj = {
                                    "x": float(x1),  # Legacy format
                                    "y": float(y1),
                                    "width": float(width),
                                    "height": float(height),
                                    "confidence": float(confidence),
                                    "bounding_box": {  # Standardized format
                                        "x": float(x1),
                                        "y": float(y1),
                                        "width": float(width),
                                        "height": float(height),
                                        "confidence": float(confidence)
                                    },
                                    "landmarks": None,  # YOLO doesn't provide landmarks
                                    "area": float(width * height)
                                }
                                faces.append(face_obj)
        
        # Apply NMS if needed
        if len(faces) > 1:
            faces = self._apply_nms(faces)
        
        return faces
    
    def _apply_nms(self, faces: List[Dict]) -> List[Dict]:
        """Apply Non-Maximum Suppression"""
        if len(faces) <= 1:
            return faces
        
        boxes = []
        confidences = []
        
        for face in faces:
            boxes.append([face["x"], face["y"], face["width"], face["height"]])
            confidences.append(face["confidence"])
        
        indices = cv2.dnn.NMSBoxes(
            boxes, confidences, 
            self.confidence_threshold, 
            self.iou_threshold
        )
        
        if len(indices) > 0:
            return [faces[i] for i in indices.flatten()]
        
        return []
    
    def _extract_face_crop(self, image: np.ndarray, face: Dict) -> np.ndarray:
        """Extract face crop for quality assessment"""
        try:
            x, y, w, h = int(face["x"]), int(face["y"]), int(face["width"]), int(face["height"])
            
            # Ensure coordinates are within bounds
            height, width = image.shape[:2]
            x = max(0, min(x, width - 1))
            y = max(0, min(y, height - 1))
            w = max(1, min(w, width - x))
            h = max(1, min(h, height - y))
            
            return image[y:y+h, x:x+w]
        except Exception:
            return np.zeros((50, 50, 3), dtype=np.uint8)
    
    def _assess_face_quality(self, face_crop: np.ndarray) -> float:
        """Assess face quality"""
        try:
            if face_crop.size == 0:
                return 0.0
            
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            
            # Sharpness
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness = min(laplacian_var / 500.0, 1.0)
            
            # Size adequacy
            area = face_crop.shape[0] * face_crop.shape[1]
            size_score = min(area / (80 * 80), 1.0)
            
            # Brightness
            brightness = np.mean(gray) / 255.0
            brightness_score = 1.0 - abs(brightness - 0.5) * 2
            
            # Combined quality
            quality = (sharpness * 0.5 + size_score * 0.3 + brightness_score * 0.2)
            return float(min(max(quality, 0.0), 1.0))
            
        except Exception:
            return 0.5

# Global variables
gpu_manager = GPUMemoryManager(memory_limit_mb=800)
model_session = None
yolo_detector = None

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
    global model_session, yolo_detector
    
    # Startup
    logger.info("Starting Enhanced YOLOv10n Face Detection Service...")
    
    model_path = "/app/models/yolov10n-face.onnx"
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        model_session = gpu_manager.setup_gpu_session(model_path)
        yolo_detector = EnhancedYOLODetector(model_session)
        logger.info("Enhanced YOLOv10n model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Enhanced YOLOv10n Face Detection Service...")
    if model_session:
        del model_session

app = FastAPI(
    title="Enhanced YOLOv10n Face Detection Service",
    description="AI microservice for face detection using YOLOv10n with quality assessment",
    version="1.1.0",
    lifespan=lifespan
)

@app.post("/detect")
async def detect_faces(file: UploadFile = File(...)):
    """Enhanced face detection in uploaded image"""
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and decode image
        image_data = await file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Detect faces
        result = yolo_detector.detect_faces(image)
        
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
    """Enhanced batch face detection"""
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 images per batch")
    
    start_time = time.time()
    results = []
    total_faces = 0
    
    for i, file in enumerate(files):
        if not file.content_type.startswith('image/'):
            continue
        
        try:
            image_data = await file.read()
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is not None:
                result = yolo_detector.detect_faces(image)
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
    gpu_available = 'CUDAExecutionProvider' in ort.get_available_providers()
    
    memory_info = {
        "ram_percent": psutil.virtual_memory().percent,
        "ram_available_gb": psutil.virtual_memory().available / (1024**3)
    }
    
    if gpu_available:
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
        "service": "YOLOv10n Face Detection",
        "version": "1.1.0",
        "model_loaded": model_session is not None,
        "gpu_available": gpu_available,
        "memory_usage": memory_info,
        "uptime": uptime,
        "capabilities": {
            "face_detection": True,
            "quality_assessment": True,
            "batch_processing": True,
            "nms": True
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
            "name": "YOLOv10n",
            "input_size": "640x640",
            "confidence_threshold": yolo_detector.confidence_threshold if yolo_detector else 0.25,
            "iou_threshold": yolo_detector.iou_threshold if yolo_detector else 0.45
        }
    }

@app.get("/")
async def root():
    """Enhanced root endpoint"""
    return {
        "message": "Enhanced YOLOv10n Face Detection Service",
        "version": "1.1.0", 
        "status": "running",
        "capabilities": {
            "face_detection": True,
            "quality_assessment": True,
            "batch_processing": True,
            "nms": True,
            "gpu_acceleration": True
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
