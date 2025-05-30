'''
AI Services API Gateway - Enhanced with Unified Face Detection
FastAPI gateway for orchestrating all FaceSocial AI microservices
Unified Face Detection Interface according to documentation specifications
'''

import os
import asyncio
import aiohttp
import cv2
import numpy as np
import base64
import time
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException, status, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union
import psutil
from contextlib import asynccontextmanager
import json
import uuid
from datetime import datetime

# Import our unified models and processor
from .face_models import (
    FaceDetectionRequest, FaceDetectionResponse, BatchDetectionRequest, BatchDetectionResponse,
    DetectionOptions, ProcessingOptions, FaceDetectionMethod, GatewayHealth, ServiceStatus,
    create_face_detection_response, create_error_response
)
from .face_processor import FaceProcessingPipeline, ServiceHealthChecker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Legacy service configurations (maintained for backward compatibility)
SERVICES = {
    "mediapipe": {"url": "http://mediapipe-predetection:5000", "timeout": 30},
    "yolo": {"url": "http://yolo10-main-detection:5000", "timeout": 30},
    "mtcnn": {"url": "http://mtcnn-precision:5000", "timeout": 30},
    "face_recognition": {"url": "http://face-recognition:5000", "timeout": 30},
    "antispoof": {"url": "http://antispoof-service:5000", "timeout": 30},
    "gender_age": {"url": "http://gender-age-service:5000", "timeout": 30},
    "deepfake": {"url": "http://deepfake-detection:5000", "timeout": 30},
    "face_quality": {"url": "http://face-quality:5000", "timeout": 30}
}

# Global HTTP session and processors
http_session = None
face_processor = None
health_checker = None

# Global metrics
metrics = {
    "total_requests": 0,
    "face_detection_requests": 0,
    "service_calls": {service: 0 for service in SERVICES.keys()},
    "service_errors": {service: 0 for service in SERVICES.keys()},
    "start_time": time.time()
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global http_session, face_processor, health_checker
    
    # Startup
    logger.info("Starting AI Services API Gateway with Unified Face Detection...")
    
    # Create HTTP session with connection pooling
    connector = aiohttp.TCPConnector(
        limit=100,
        limit_per_host=20,
        keepalive_timeout=30,
        enable_cleanup_closed=True
    )
    
    timeout = aiohttp.ClientTimeout(total=60, connect=10)
    http_session = aiohttp.ClientSession(
        connector=connector,
        timeout=timeout,
        headers={"User-Agent": "FaceSocial-Gateway/1.0"}
    )
    
    # Initialize processors
    face_processor = FaceProcessingPipeline(http_session)
    health_checker = ServiceHealthChecker(http_session)
    
    logger.info("API Gateway started successfully with unified face detection")
    
    yield
    
    # Shutdown
    logger.info("Shutting down API Gateway...")
    if http_session:
        await http_session.close()

app = FastAPI(
    title="FaceSocial AI Services Gateway",
    description="Unified API Gateway for Face Detection and AI microservices",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== UNIFIED FACE DETECTION API v1 =====

@app.post("/api/v1/face-detection/detect")
async def unified_face_detection(
    file: UploadFile = File(...),
    detection_method: str = Form(default="mediapipe"),
    min_face_size: int = Form(default=40),
    max_faces: int = Form(default=100),
    detection_confidence: float = Form(default=0.7),
    return_landmarks: bool = Form(default=True),
    return_attributes: bool = Form(default=False),
    quality_assessment: bool = Form(default=True),
    face_alignment: bool = Form(default=True),
    crop_faces: bool = Form(default=True),
    enhance_quality: bool = Form(default=False)
):
    """
    Unified Face Detection API
    Supports multi-face detection with quality assessment, landmarks, and attributes
    """
    
    if not file.content_type.startswith('image/'):
        return create_error_response(
            "INVALID_INPUT",
            "File must be an image",
            {"content_type": file.content_type}
        )
    
    try:
        # Read image data
        image_data = await file.read()
        
        # Validate detection method
        try:
            method = FaceDetectionMethod(detection_method.lower())
        except ValueError:
            return create_error_response(
                "INVALID_DETECTION_METHOD",
                f"Invalid detection method: {detection_method}",
                {"available_methods": [m.value for m in FaceDetectionMethod]}
            )
        
        # Create options
        detection_options = DetectionOptions(
            min_face_size=min_face_size,
            max_faces=max_faces,
            detection_confidence=detection_confidence,
            return_landmarks=return_landmarks,
            return_attributes=return_attributes,
            quality_assessment=quality_assessment
        )
        
        processing_options = ProcessingOptions(
            face_alignment=face_alignment,
            crop_faces=crop_faces,
            enhance_quality=enhance_quality
        )
        
        # Process image
        faces, image_info, processing_stats = await face_processor.process_image(
            image_data, detection_options, processing_options, method
        )
        
        # Update metrics
        metrics["total_requests"] += 1
        metrics["face_detection_requests"] += 1
        
        # Create response
        request_id = f"req_{uuid.uuid4().hex[:8]}"
        response = create_face_detection_response(
            faces, image_info, processing_stats, request_id
        )
        
        logger.info(f"Face detection completed: {len(faces)} faces detected in {processing_stats.total_time_ms:.0f}ms")
        
        return response
        
    except Exception as e:
        logger.error(f"Face detection error: {e}")
        return create_error_response(
            "PROCESSING_ERROR",
            "Face detection processing failed",
            {"error": str(e)}
        )

@app.post("/api/v1/face-detection/batch-detect")
async def batch_face_detection(
    files: List[UploadFile] = File(...),
    detection_method: str = Form(default="mediapipe"),
    min_face_size: int = Form(default=40),
    max_faces: int = Form(default=100),
    detection_confidence: float = Form(default=0.7),
    return_landmarks: bool = Form(default=True),
    return_attributes: bool = Form(default=False),
    quality_assessment: bool = Form(default=True),
    max_concurrent: int = Form(default=3)
):
    """
    Batch Face Detection API
    Process multiple images with controlled concurrency
    """
    
    if len(files) > 20:
        return create_error_response(
            "BATCH_SIZE_EXCEEDED",
            "Maximum 20 images per batch",
            {"provided": len(files), "maximum": 20}
        )
    
    start_time = time.time()
    
    try:
        # Validate detection method
        try:
            method = FaceDetectionMethod(detection_method.lower())
        except ValueError:
            return create_error_response(
                "INVALID_DETECTION_METHOD",
                f"Invalid detection method: {detection_method}",
                {"available_methods": [m.value for m in FaceDetectionMethod]}
            )
        
        # Create options
        detection_options = DetectionOptions(
            min_face_size=min_face_size,
            max_faces=max_faces,
            detection_confidence=detection_confidence,
            return_landmarks=return_landmarks,
            return_attributes=return_attributes,
            quality_assessment=quality_assessment
        )
        
        processing_options = ProcessingOptions()
        
        # Process images with controlled concurrency
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single_image(file: UploadFile):
            async with semaphore:
                if not file.content_type.startswith('image/'):
                    return None
                
                image_data = await file.read()
                return await face_processor.process_image(
                    image_data, detection_options, processing_options, method
                )
        
        # Process all images
        tasks = [process_single_image(file) for file in files]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Create batch response
        batch_id = f"batch_{uuid.uuid4().hex[:8]}"
        successful_results = []
        total_faces = 0
        
        for i, result in enumerate(results):
            if isinstance(result, tuple) and len(result) == 3:
                faces, image_info, processing_stats = result
                
                # Create individual response
                request_id = f"{batch_id}_img_{i}"
                response = create_face_detection_response(
                    faces, image_info, processing_stats, request_id
                )
                successful_results.append(response.data)
                total_faces += len(faces)
            elif isinstance(result, Exception):
                logger.error(f"Batch processing error for image {i}: {result}")
        
        total_time = (time.time() - start_time) * 1000
        success_rate = len(successful_results) / len(files) if files else 0
        
        # Update metrics
        metrics["total_requests"] += 1
        metrics["face_detection_requests"] += 1
        
        batch_response = BatchDetectionResponse(
            success=True,
            data={
                "batch_id": batch_id,
                "status": "completed",
                "total_images": len(files),
                "completed_images": len(successful_results),
                "failed_images": len(files) - len(successful_results),
                "total_faces_detected": total_faces,
                "results": successful_results,
                "success_rate": success_rate,
                "processing_time_ms": total_time
            },
            metadata={
                "batch_id": batch_id,
                "timestamp": datetime.now().isoformat(),
                "processing_time": f"{total_time:.0f}ms"
            }
        )
        
        logger.info(f"Batch detection completed: {len(successful_results)}/{len(files)} images, {total_faces} total faces")
        
        return batch_response
        
    except Exception as e:
        logger.error(f"Batch detection error: {e}")
        return create_error_response(
            "BATCH_PROCESSING_ERROR",
            "Batch face detection processing failed",
            {"error": str(e)}
        )

@app.get("/api/v1/face-detection/methods")
async def get_detection_methods():
    """Get available face detection methods and their capabilities"""
    
    # Check service health
    health_status = await health_checker.check_all_services()
    
    methods = {}
    for method in FaceDetectionMethod:
        service_health = health_status.get(method.value, {})
        
        methods[method.value] = {
            "name": method.value.upper(),
            "available": service_health.get('available', False),
            "response_time_ms": service_health.get('response_time_ms'),
            "capabilities": {
                "landmarks": method == FaceDetectionMethod.MTCNN,
                "quality_assessment": True,
                "batch_processing": True,
                "real_time": method in [FaceDetectionMethod.MEDIAPIPE, FaceDetectionMethod.YOLO]
            },
            "recommended_use": {
                FaceDetectionMethod.MEDIAPIPE: "Fast real-time detection, good for video streams",
                FaceDetectionMethod.YOLO: "Balanced speed and accuracy, good for general use", 
                FaceDetectionMethod.MTCNN: "High precision with landmarks, good for face recognition prep"
            }.get(method, "General purpose detection")
        }
    
    return {
        "success": True,
        "data": {
            "available_methods": methods,
            "default_method": "mediapipe",
            "recommended_method": "yolo"
        },
        "metadata": {
            "timestamp": datetime.now().isoformat()
        }
    }

# ===== LEGACY ENDPOINTS (Backward Compatibility) =====

async def call_service(service_name: str, endpoint: str, data: Dict = None, files: Dict = None, method: str = "POST") -> Dict:
    """Call a microservice endpoint (legacy function)"""
    global metrics
    
    if service_name not in SERVICES:
        raise HTTPException(status_code=400, detail=f"Unknown service: {service_name}")
    
    service_config = SERVICES[service_name]
    url = f"{service_config['url']}/{endpoint}"
    timeout = service_config['timeout']
    
    try:
        metrics["service_calls"][service_name] += 1
        
        # Prepare request
        request_kwargs = {
            "timeout": aiohttp.ClientTimeout(total=timeout)
        }
        
        if method.upper() == "GET":
            async with http_session.get(url, **request_kwargs) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    logger.error(f"Service {service_name} error {response.status}: {error_text}")
                    metrics["service_errors"][service_name] += 1
                    raise HTTPException(status_code=response.status, detail=f"Service {service_name} error: {error_text}")
        else:
            if files:
                form_data = aiohttp.FormData()
                for key, value in files.items():
                    form_data.add_field(key, value[1], filename=value[0], content_type=value[2])
                if data:
                    for key, value in data.items():
                        form_data.add_field(key, str(value))
                request_kwargs["data"] = form_data
            elif data:
                request_kwargs["json"] = data
            
            async with http_session.post(url, **request_kwargs) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    logger.error(f"Service {service_name} error {response.status}: {error_text}")
                    metrics["service_errors"][service_name] += 1
                    raise HTTPException(status_code=response.status, detail=f"Service {service_name} error: {error_text}")
                
    except asyncio.TimeoutError:
        logger.error(f"Timeout calling service {service_name}")
        metrics["service_errors"][service_name] += 1
        raise HTTPException(status_code=504, detail=f"Service {service_name} timeout")
    except Exception as e:
        logger.error(f"Error calling service {service_name}: {e}")
        metrics["service_errors"][service_name] += 1
        raise HTTPException(status_code=503, detail=f"Service {service_name} unavailable")

# Legacy face detection endpoints
@app.post("/detect/faces")
async def legacy_detect_faces(
    file: UploadFile = File(...),
    service: str = Form(default="mediapipe")
):
    """Legacy face detection endpoint (backward compatibility)"""
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        image_data = await file.read()
        file_data = ("image.jpg", image_data, "image/jpeg")
        
        metrics["total_requests"] += 1
        
        result = await call_service(service, "detect", files={"file": file_data})
        return result
        
    except Exception as e:
        logger.error(f"Error in legacy face detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== COMPREHENSIVE ANALYSIS (Enhanced) =====

@app.post("/analyze/comprehensive")
async def comprehensive_analysis(
    file: UploadFile = File(...),
    detect_faces: bool = Form(default=True),
    face_service: str = Form(default="mediapipe"),
    recognize_faces: bool = Form(default=False),
    detect_spoofing: bool = Form(default=False),
    predict_gender_age: bool = Form(default=False),
    detect_deepfake: bool = Form(default=False),
    assess_quality: bool = Form(default=True)
):
    """Enhanced comprehensive analysis with quality assessment"""
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    start_time = time.time()
    services_used = []
    
    try:
        image_data = await file.read()
        file_data = ("image.jpg", image_data, "image/jpeg")
        
        # Step 1: Face Detection (always performed)
        face_results = None
        if detect_faces:
            face_results = await call_service(face_service, "detect", files={"file": file_data})
            services_used.append(face_service)
        
        # Step 2: Quality Assessment (if requested)
        quality_results = None
        if assess_quality and face_results and face_results.get('faces'):
            try:
                quality_results = await call_service("face_quality", "assess", files={"file": file_data})
                services_used.append("face_quality")
            except Exception as e:
                logger.warning(f"Quality assessment failed: {e}")
        
        # Step 3: Other analyses (parallel execution)
        tasks = []
        
        if recognize_faces:
            tasks.append(("recognition", call_service("face_recognition", "recognize", files={"file": file_data})))
        
        if detect_spoofing:
            tasks.append(("antispoofing", call_service("antispoof", "detect_spoof", files={"file": file_data})))
        
        if predict_gender_age:
            tasks.append(("gender_age", call_service("gender_age", "predict", files={"file": file_data})))
        
        if detect_deepfake:
            tasks.append(("deepfake", call_service("deepfake", "detect_deepfake", files={"file": file_data})))
        
        # Execute parallel tasks
        analysis_results = {}
        if tasks:
            task_results = await asyncio.gather(*[task[1] for task in tasks], return_exceptions=True)
            
            for (name, _), result in zip(tasks, task_results):
                if not isinstance(result, Exception):
                    analysis_results[name] = result
                    services_used.append(name)
                else:
                    logger.error(f"Error in {name} analysis: {result}")
                    analysis_results[name] = None
        
        # Combine results
        comprehensive_result = {
            "face_detection": face_results,
            "quality_assessment": quality_results,
            **analysis_results
        }
        
        processing_time = (time.time() - start_time) * 1000
        
        # Update metrics
        metrics["total_requests"] += 1
        
        return {
            "success": True,
            "data": comprehensive_result,
            "metadata": {
                "request_id": f"comp_{uuid.uuid4().hex[:8]}",
                "processing_time": f"{processing_time:.0f}ms",
                "services_used": services_used,
                "timestamp": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error in comprehensive analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== HEALTH AND MONITORING =====

@app.get("/health")
async def gateway_health():
    """Enhanced health check with unified services"""
    
    start_time = time.time()
    
    # Check all services including new face-quality service
    health_status = await health_checker.check_all_services()
    
    # Also check legacy services
    legacy_services = ["face_recognition", "antispoof", "deepfake"]
    for service_name in legacy_services:
        try:
            result = await call_service(service_name, "health")
            health_status[service_name] = {
                'service': service_name,
                'status': 'healthy',
                'available': True,
                'response_time_ms': (time.time() - start_time) * 1000
            }
        except Exception as e:
            health_status[service_name] = {
                'service': service_name,
                'status': 'error',
                'available': False,
                'error': str(e)
            }
    
    # Convert to ServiceStatus objects
    service_statuses = []
    healthy_count = 0
    
    for service_name, status_data in health_status.items():
        service_status = ServiceStatus(
            service_name=service_name,
            status=status_data.get('status', 'unknown'),
            response_time_ms=status_data.get('response_time_ms'),
            available=status_data.get('available', False)
        )
        service_statuses.append(service_status)
        
        if service_status.available:
            healthy_count += 1
    
    overall_status = "healthy" if healthy_count == len(service_statuses) else "degraded"
    uptime = time.time() - metrics["start_time"]
    
    gateway_health_response = GatewayHealth(
        status=overall_status,
        total_services=len(service_statuses),
        healthy_services=healthy_count,
        unhealthy_services=len(service_statuses) - healthy_count,
        services=service_statuses,
        uptime=uptime
    )
    
    return gateway_health_response

@app.get("/metrics")
async def get_gateway_metrics():
    """Enhanced gateway metrics"""
    
    uptime = time.time() - metrics["start_time"]
    
    return {
        "uptime": uptime,
        "total_requests": metrics["total_requests"],
        "face_detection_requests": metrics["face_detection_requests"],
        "service_calls": metrics["service_calls"],
        "service_errors": metrics["service_errors"],
        "error_rates": {
            service: (errors / max(calls, 1)) * 100 
            for service, (calls, errors) in zip(
                SERVICES.keys(), 
                zip(metrics["service_calls"].values(), metrics["service_errors"].values())
            )
        },
        "memory_usage": {
            "ram_percent": psutil.virtual_memory().percent,
            "ram_available_gb": psutil.virtual_memory().available / (1024**3)
        },
        "api_version": "2.0.0",
        "features": {
            "unified_face_detection": True,
            "quality_assessment": True,
            "batch_processing": True,
            "legacy_compatibility": True
        }
    }

@app.get("/services")
async def list_services():
    """List all available services with capabilities"""
    
    health_status = await health_checker.check_all_services()
    
    services_info = {}
    
    # Face detection services
    for method in FaceDetectionMethod:
        service_health = health_status.get(method.value, {})
        services_info[f"face_detection_{method.value}"] = {
            "type": "face_detection",
            "method": method.value,
            "endpoint": f"/api/v1/face-detection/detect?detection_method={method.value}",
            "available": service_health.get('available', False),
            "capabilities": ["detection", "landmarks" if method == FaceDetectionMethod.MTCNN else None],
            "description": f"Face detection using {method.value.upper()}"
        }
    
    # Other services
    other_services = {
        "face_quality": {
            "type": "quality_assessment",
            "endpoint": "/face-quality/assess",
            "capabilities": ["quality_metrics", "recommendations"],
            "description": "Comprehensive face quality assessment"
        },
        "face_recognition": {
            "type": "face_recognition", 
            "endpoint": "/face-recognition/recognize",
            "capabilities": ["recognition", "registration", "verification"],
            "description": "Face recognition and identity matching"
        },
        "antispoof": {
            "type": "security",
            "endpoint": "/antispoof/detect_spoof",
            "capabilities": ["liveness_detection", "spoof_prevention"],
            "description": "Anti-spoofing and liveness detection"
        },
        "gender_age": {
            "type": "attributes",
            "endpoint": "/gender-age/predict",
            "capabilities": ["gender_prediction", "age_estimation"],
            "description": "Gender and age analysis"
        },
        "deepfake": {
            "type": "security",
            "endpoint": "/deepfake/detect_deepfake", 
            "capabilities": ["deepfake_detection", "manipulation_detection"],
            "description": "Deepfake and manipulation detection"
        }
    }
    
    for service_name, service_info in other_services.items():
        service_health = health_status.get(service_name, {})
        service_info["available"] = service_health.get('available', False)
        services_info[service_name] = service_info
    
    return {
        "success": True,
        "data": {
            "services": services_info,
            "total_services": len(services_info),
            "api_version": "2.0.0"
        },
        "metadata": {
            "timestamp": datetime.now().isoformat()
        }
    }

@app.get("/")
async def root():
    """Enhanced root endpoint"""
    return {
        "message": "FaceSocial AI Services Gateway",
        "version": "2.0.0",
        "status": "running",
        "features": {
            "unified_face_detection": True,
            "quality_assessment": True,
            "batch_processing": True,
            "comprehensive_analysis": True,
            "legacy_compatibility": True
        },
        "api_docs": "/docs",
        "health_check": "/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
