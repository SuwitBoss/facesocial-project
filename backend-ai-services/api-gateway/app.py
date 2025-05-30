"""
AI Services API Gateway
FastAPI gateway for orchestrating all FaceSocial AI microservices
"""

import os
import asyncio
import aiohttp
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, status, Form, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union
import logging
import psutil
import time
from contextlib import asynccontextmanager
import json
import base64
import io
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Service configurations
SERVICES = {
    "mediapipe": {"url": "http://mediapipe-predetection:5000", "timeout": 30},
    "yolo": {"url": "http://yolo10-main-detection:5000", "timeout": 30},
    "mtcnn": {"url": "http://mtcnn-precision:5000", "timeout": 30},
    "face_recognition": {"url": "http://face-recognition:5000", "timeout": 30},
    "antispoof": {"url": "http://antispoof-service:5000", "timeout": 30},
    "gender_age": {"url": "http://gender-age-service:5000", "timeout": 30},
    "deepfake": {"url": "http://deepfake-detection:5000", "timeout": 30}
}

# Global HTTP session for service communication
http_session = None

# Pydantic models
class ServiceHealth(BaseModel):
    service: str
    status: str
    response_time: float
    available: bool

class GatewayHealth(BaseModel):
    status: str
    total_services: int
    healthy_services: int
    unhealthy_services: int
    services: List[ServiceHealth]
    uptime: float

class FaceDetectionRequest(BaseModel):
    service: str = "mediapipe"  # mediapipe, yolo, mtcnn
    include_landmarks: bool = False

class ComprehensiveAnalysisRequest(BaseModel):
    detect_faces: bool = True
    face_service: str = "mediapipe"
    recognize_faces: bool = False
    detect_spoofing: bool = False
    predict_gender_age: bool = False
    detect_deepfake: bool = False
    include_landmarks: bool = False

class FaceBox(BaseModel):
    x: float
    y: float
    width: float
    height: float
    confidence: float

class ComprehensiveResult(BaseModel):
    face_id: int
    detection: Optional[Dict] = None
    recognition: Optional[Dict] = None
    antispoofing: Optional[Dict] = None
    gender_age: Optional[Dict] = None
    deepfake: Optional[Dict] = None

class ComprehensiveResponse(BaseModel):
    faces: List[ComprehensiveResult]
    processing_time: float
    services_used: List[str]
    total_faces_detected: int
    summary: Dict[str, Any]

class BatchProcessingRequest(BaseModel):
    analysis_config: ComprehensiveAnalysisRequest
    max_concurrent: int = 3

class BatchProcessingResponse(BaseModel):
    results: List[ComprehensiveResponse]
    total_processing_time: float
    total_images: int
    total_faces: int
    success_rate: float

# Global metrics
metrics = {
    "total_requests": 0,
    "service_calls": {service: 0 for service in SERVICES.keys()},
    "service_errors": {service: 0 for service in SERVICES.keys()},
    "start_time": time.time()
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global http_session
    
    # Startup
    logger.info("Starting AI Services API Gateway...")
    
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
    
    logger.info("API Gateway started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down API Gateway...")
    if http_session:
        await http_session.close()

app = FastAPI(
    title="FaceSocial AI Services Gateway",
    description="API Gateway for orchestrating AI microservices",
    version="1.0.0",
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

async def call_service(service_name: str, endpoint: str, data: Dict = None, files: Dict = None, method: str = "POST") -> Dict:
    """Call a microservice endpoint"""
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
        
        # For health checks and metrics, use GET method
        if endpoint in ["health", "metrics", ""]:
            method = "GET"
        
        if method.upper() == "GET":
            # GET request - no body
            async with http_session.get(url, **request_kwargs) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    logger.error(f"Service {service_name} error {response.status}: {error_text}")
                    metrics["service_errors"][service_name] += 1
                    raise HTTPException(
                        status_code=response.status,
                        detail=f"Service {service_name} error: {error_text}"
                    )
        else:
            # POST request - with data/files
            if files:
                # Multipart form data
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
                    raise HTTPException(
                        status_code=response.status,
                        detail=f"Service {service_name} error: {error_text}"
                    )
                
    except asyncio.TimeoutError:
        logger.error(f"Timeout calling service {service_name}")
        metrics["service_errors"][service_name] += 1
        raise HTTPException(status_code=504, detail=f"Service {service_name} timeout")
    except Exception as e:
        logger.error(f"Error calling service {service_name}: {e}")
        metrics["service_errors"][service_name] += 1
        raise HTTPException(status_code=503, detail=f"Service {service_name} unavailable")

async def check_service_health(service_name: str) -> ServiceHealth:
    """Check health of a specific service"""
    start_time = time.time()
    
    try:
        result = await call_service(service_name, "health")
        response_time = time.time() - start_time
        
        return ServiceHealth(
            service=service_name,
            status="healthy",
            response_time=response_time,
            available=True
        )
    except Exception as e:
        response_time = time.time() - start_time
        return ServiceHealth(
            service=service_name,
            status=f"unhealthy: {str(e)}",
            response_time=response_time,
            available=False
        )

def extract_face_boxes(detection_result: Dict, service_name: str) -> List[Dict]:
    """Extract face bounding boxes from detection result"""
    boxes = []
    
    if service_name == "mediapipe":
        for i, face in enumerate(detection_result.get("faces", [])):
            bbox = face.get("bbox", {})
            boxes.append({
                "x": bbox.get("x", 0),
                "y": bbox.get("y", 0),
                "width": bbox.get("width", 0),
                "height": bbox.get("height", 0),
                "confidence": face.get("confidence", 0)
            })
    elif service_name in ["yolo", "mtcnn"]:
        for face in detection_result.get("faces", []):
            boxes.append({
                "x": face.get("x", 0),
                "y": face.get("y", 0),
                "width": face.get("width", 0),
                "height": face.get("height", 0),
                "confidence": face.get("confidence", 0)
            })
    
    return boxes

async def process_comprehensive_analysis(
    image_data: bytes, 
    config: ComprehensiveAnalysisRequest
) -> ComprehensiveResponse:
    """Process comprehensive face analysis using multiple services"""
    start_time = time.time()
    services_used = []
    face_boxes = []
    
    # Prepare file data for service calls
    file_data = ("image.jpg", image_data, "image/jpeg")
    
    try:
        # Step 1: Face Detection
        detection_result = None
        if config.detect_faces:
            detection_result = await call_service(
                config.face_service, 
                "detect", 
                files={"file": file_data}
            )
            services_used.append(config.face_service)
            face_boxes = extract_face_boxes(detection_result, config.face_service)
        
        # If no faces detected, return empty result
        if not face_boxes:
            return ComprehensiveResponse(
                faces=[],
                processing_time=time.time() - start_time,
                services_used=services_used,
                total_faces_detected=0,
                summary={"message": "No faces detected"}
            )
        
        # Prepare face boxes for other services
        face_boxes_json = json.dumps(face_boxes)
        
        # Step 2: Parallel processing of other analyses
        tasks = []
        
        if config.recognize_faces:
            tasks.append(("recognition", call_service(
                "face_recognition", 
                "recognize", 
                data={"face_boxes": face_boxes_json},
                files={"file": file_data}
            )))
        
        if config.detect_spoofing:
            tasks.append(("antispoofing", call_service(
                "antispoof", 
                "detect", 
                data={"face_boxes": face_boxes_json},
                files={"file": file_data}
            )))
        
        if config.predict_gender_age:
            tasks.append(("gender_age", call_service(
                "gender_age", 
                "predict", 
                data={"face_boxes": face_boxes_json},
                files={"file": file_data}
            )))
        
        if config.detect_deepfake:
            tasks.append(("deepfake", call_service(
                "deepfake", 
                "detect", 
                data={"face_boxes": face_boxes_json},
                files={"file": file_data}
            )))
        
        # Execute parallel tasks
        results = {}
        if tasks:
            task_results = await asyncio.gather(
                *[task[1] for task in tasks], 
                return_exceptions=True
            )
            
            for (name, _), result in zip(tasks, task_results):
                if not isinstance(result, Exception):
                    results[name] = result
                    services_used.append(name)
                else:
                    logger.error(f"Error in {name} service: {result}")
                    results[name] = None
        
        # Step 3: Combine results per face
        faces = []
        for i, box in enumerate(face_boxes):
            face_result = ComprehensiveResult(
                face_id=i,
                detection={"bbox": box, "service": config.face_service}
            )
            
            # Add recognition results
            if "recognition" in results and results["recognition"]:
                recognition_data = results["recognition"]
                if i < len(recognition_data.get("faces", [])):
                    face_result.recognition = recognition_data["faces"][i]
            
            # Add antispoofing results
            if "antispoofing" in results and results["antispoofing"]:
                antispoofing_data = results["antispoofing"]
                if i < len(antispoofing_data.get("detections", [])):
                    face_result.antispoofing = antispoofing_data["detections"][i]
            
            # Add gender/age results
            if "gender_age" in results and results["gender_age"]:
                gender_age_data = results["gender_age"]
                if i < len(gender_age_data.get("predictions", [])):
                    face_result.gender_age = gender_age_data["predictions"][i]
            
            # Add deepfake results
            if "deepfake" in results and results["deepfake"]:
                deepfake_data = results["deepfake"]
                if i < len(deepfake_data.get("detections", [])):
                    face_result.deepfake = deepfake_data["detections"][i]
            
            faces.append(face_result)
        
        # Step 4: Generate summary
        summary = {
            "total_faces": len(faces),
            "detection_service": config.face_service,
            "services_used": services_used
        }
        
        if config.recognize_faces and "recognition" in results:
            known_faces = sum(1 for f in faces if f.recognition and f.recognition.get("is_known", False))
            summary["known_faces"] = known_faces
        
        if config.detect_spoofing and "antispoofing" in results:
            real_faces = sum(1 for f in faces if f.antispoofing and f.antispoofing.get("is_real", False))
            summary["real_faces"] = real_faces
        
        if config.predict_gender_age and "gender_age" in results:
            genders = [f.gender_age.get("gender") for f in faces if f.gender_age]
            summary["gender_distribution"] = {
                "male": genders.count("male"),
                "female": genders.count("female")
            }
        
        if config.detect_deepfake and "deepfake" in results:
            authentic_faces = sum(1 for f in faces if f.deepfake and f.deepfake.get("is_real", False))
            summary["authentic_faces"] = authentic_faces
        
        processing_time = time.time() - start_time
        
        return ComprehensiveResponse(
            faces=faces,
            processing_time=processing_time,
            services_used=services_used,
            total_faces_detected=len(faces),
            summary=summary
        )
        
    except Exception as e:
        logger.error(f"Error in comprehensive analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")

@app.post("/detect/faces", response_model=Dict)
async def detect_faces(
    file: UploadFile = File(...),
    service: str = Form(default="mediapipe"),
    include_landmarks: bool = Form(default=False)
):
    """Detect faces using specified service"""
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        image_data = await file.read()
        file_data = ("image.jpg", image_data, "image/jpeg")
        
        metrics["total_requests"] += 1
        
        result = await call_service(
            service, 
            "detect", 
            files={"file": file_data}
        )
        
        return result
    except Exception as e:
        logger.error(f"Error in face detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/comprehensive", response_model=ComprehensiveResponse)
async def comprehensive_analysis(
    file: UploadFile = File(...),
    config: str = Form(...)
):
    """Perform comprehensive face analysis"""
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        image_data = await file.read()
        analysis_config = ComprehensiveAnalysisRequest.parse_raw(config)
        
        metrics["total_requests"] += 1
        
        result = await process_comprehensive_analysis(image_data, analysis_config)
        return result
        
    except Exception as e:
        logger.error(f"Error in comprehensive analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/batch", response_model=BatchProcessingResponse)
async def batch_analysis(
    files: List[UploadFile] = File(...),
    config: str = Form(...)
):
    """Perform batch analysis on multiple images"""
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 images per batch")
    
    try:
        batch_config = BatchProcessingRequest.parse_raw(config)
        start_time = time.time()
        
        # Process images with controlled concurrency
        semaphore = asyncio.Semaphore(batch_config.max_concurrent)
        
        async def process_single_image(file: UploadFile):
            async with semaphore:
                if not file.content_type.startswith('image/'):
                    return None
                image_data = await file.read()
                return await process_comprehensive_analysis(
                    image_data, 
                    batch_config.analysis_config
                )
        
        # Process all images
        tasks = [process_single_image(file) for file in files]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful results
        successful_results = []
        for result in results:
            if isinstance(result, ComprehensiveResponse):
                successful_results.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Batch processing error: {result}")
        
        total_time = time.time() - start_time
        total_faces = sum(len(r.faces) for r in successful_results)
        success_rate = len(successful_results) / len(files) if files else 0
        
        metrics["total_requests"] += 1
        
        return BatchProcessingResponse(
            results=successful_results,
            total_processing_time=total_time,
            total_images=len(files),
            total_faces=total_faces,
            success_rate=success_rate
        )
        
    except Exception as e:
        logger.error(f"Error in batch analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", response_model=GatewayHealth)
async def health_check():
    """Check health of gateway and all services"""
    start_time = time.time()
    
    # Check all services in parallel
    health_tasks = [check_service_health(service) for service in SERVICES.keys()]
    service_healths = await asyncio.gather(*health_tasks, return_exceptions=True)
    
    # Process results
    healthy_services = []
    for health in service_healths:
        if isinstance(health, ServiceHealth):
            healthy_services.append(health)
        else:
            # Handle exceptions
            logger.error(f"Health check error: {health}")
    
    healthy_count = sum(1 for h in healthy_services if h.available)
    unhealthy_count = len(healthy_services) - healthy_count
    
    overall_status = "healthy" if healthy_count == len(SERVICES) else "degraded"
    uptime = time.time() - metrics["start_time"]
    
    return GatewayHealth(
        status=overall_status,
        total_services=len(SERVICES),
        healthy_services=healthy_count,
        unhealthy_services=unhealthy_count,
        services=healthy_services,
        uptime=uptime
    )

@app.get("/metrics")
async def get_metrics():
    """Get gateway metrics"""
    uptime = time.time() - metrics["start_time"]
    
    return {
        "uptime": uptime,
        "total_requests": metrics["total_requests"],
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
        }
    }

@app.get("/services")
async def list_services():
    """List all available services"""
    return {
        "services": list(SERVICES.keys()),
        "endpoints": {
            service: f"{config['url']}" 
            for service, config in SERVICES.items()
        }
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "FaceSocial AI Services Gateway",
        "version": "1.0.0",
        "status": "running",
        "services": list(SERVICES.keys())
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
