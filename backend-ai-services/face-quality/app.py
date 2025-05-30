"""
Face Quality Assessment Service
FastAPI microservice for comprehensive face quality analysis
Supports sharpness, brightness, contrast, face angle, occlusion detection
"""

import os
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, status, Form, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Tuple, Union
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

# Import the new quality models
from .quality_models import (
    FaceQualityONNX, 
    AdvancedQualityAssessor, 
    QualityMetrics as ModelQualityMetrics, # Renamed to avoid conflict
    FacePose
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Pydantic Models ---
class FaceQualityMetrics(BaseModel): # Keep this for API response consistency if needed, or adapt
    overall_score: float = Field(..., example=0.85)
    sharpness: float = Field(..., example=0.9)
    brightness: float = Field(..., example=0.7)
    contrast: float = Field(..., example=0.75)
    face_angle: Dict[str, float] = Field(..., example={"yaw": 5.0, "pitch": -2.0, "roll": 1.0})
    occlusion: Dict[str, float] = Field(..., example={"left_eye": 0.05, "right_eye": 0.03, "nose": 0.0, "mouth": 0.01})
    blur_score: Optional[float] = Field(None, example=0.92) # From original, can be mapped from new model
    noise_level: Optional[float] = Field(None, example=0.05) # From original, can be mapped from new model
    illumination_quality: Optional[float] = Field(None, example=0.8) # From original, can be mapped from new model
    resolution_adequacy: Optional[float] = Field(None, example=0.95) # From original, can be mapped from new model
    # New fields from AdvancedQualityAssessor
    head_pose: Optional[FacePose] = Field(None, example={"yaw": 5.0, "pitch": -2.0, "roll": 1.0, "is_frontal": True})
    eye_status: Optional[Dict[str, bool]] = Field(None, example={"left_eye_open": True, "right_eye_open": True})
    mouth_status: Optional[Dict[str, bool]] = Field(None, example={"mouth_open": False, "mouth_occluded": False})
    glasses_detected: Optional[bool] = Field(None, example=False)
    hat_detected: Optional[bool] = Field(None, example=False)
    mask_detected: Optional[bool] = Field(None, example=False)
    skin_tone_uniformity: Optional[float] = Field(None, example=0.88)
    detail_level: Optional[float] = Field(None, example=0.9)
    color_balance: Optional[Dict[str, float]] = Field(None, example={"red": 0.5, "green": 0.5, "blue": 0.5})


class QualityAssessmentResult(BaseModel):
    face_id: int = Field(..., example=0)
    bbox: Dict[str, float] = Field(..., example={"x": 100.0, "y": 150.0, "width": 80.0, "height": 80.0})
    quality_metrics: FaceQualityMetrics
    quality_category: str = Field(..., example="excellent") # "excellent", "good", "fair", "poor"
    recommendations: List[str] = Field(..., example=["Face quality is good for recognition"])

class QualityResponse(BaseModel):
    faces: List[QualityAssessmentResult]
    processing_time: float = Field(..., example=0.15)
    average_quality: float = Field(..., example=0.85)
    model_info: str = Field(..., example="Advanced Face Quality Assessment v2.0 (ONNX)")

class BatchQualityResponse(BaseModel):
    results: List[QualityResponse]
    total_processing_time: float = Field(..., example=1.5)
    total_faces_processed: int = Field(..., example=10)
    quality_distribution: Dict[str, int] = Field(..., example={"excellent": 8, "good": 2, "fair": 0, "poor": 0})

class HealthResponse(BaseModel):
    status: str = Field(..., example="healthy")
    service: str = Field(..., example="Advanced Face Quality Assessment")
    gpu_available: bool = Field(..., example=True)
    memory_usage: Dict[str, Any] = Field(..., example={"ram_percent": 25.5, "ram_available_gb": 10.2})
    uptime: float = Field(..., example=3600.5)
    model_loaded: bool = Field(..., example=True)
    model_type: Optional[str] = Field(None, example="ONNX")

class ModelInfoResponse(BaseModel):
    model_name: str = Field(..., example="AdvancedQualityAssessor")
    model_version: str = Field(..., example="2.0.0")
    model_type: str = Field(..., example="ONNX-based with rule-engine")
    supported_metrics: List[str] = Field(..., example=["sharpness", "brightness", "contrast", "head_pose", "occlusion", "eye_status"])
    onnx_model_path: Optional[str] = Field(None, example="/models/face_quality.onnx")
    onnx_providers: Optional[List[str]] = Field(None, example=["CUDAExecutionProvider", "CPUExecutionProvider"])

# Global metrics
metrics = {
    "total_requests": 0,
    "total_faces_processed": 0,
    "total_processing_time": 0.0,
    "quality_distribution": {"excellent": 0, "good": 0, "fair": 0, "poor": 0},
    "start_time": time.time(),
    "model_loaded": False,
    "model_type": None
}

# --- Face Quality Assessment Engine ---
# Replace the old FaceQualityAssessment class with the new AdvancedQualityAssessor
# The AdvancedQualityAssessor is imported from quality_models.py

# Global quality engine instance
quality_engine: Optional[AdvancedQualityAssessor] = None

def get_onnx_model_path():
    return os.getenv("ONNX_MODEL_PATH", "/models/face_quality_scrfd.onnx") # Default path

def initialize_quality_engine():
    global quality_engine, metrics
    try:
        onnx_model_path = get_onnx_model_path()
        if not os.path.exists(onnx_model_path):
            logger.warning(f"ONNX model not found at {onnx_model_path}. Advanced features might be limited.")
            # Fallback or raise error depending on desired behavior
            # For now, we'll allow it to proceed with rule-based assessment if ONNX fails
            quality_engine = AdvancedQualityAssessor(onnx_model_path=None) # Initialize without ONNX
            metrics["model_type"] = "Rule-based (ONNX model not found)"
        else:
            quality_engine = AdvancedQualityAssessor(onnx_model_path=onnx_model_path)
            metrics["model_type"] = f"ONNX ({quality_engine.face_quality_onnx.providers})"
        
        metrics["model_loaded"] = True
        logger.info(f"AdvancedQualityAssessor initialized. Model type: {metrics['model_type']}")
    except Exception as e:
        logger.error(f"Failed to initialize AdvancedQualityAssessor: {e}", exc_info=True)
        quality_engine = AdvancedQualityAssessor(onnx_model_path=None) # Fallback to rule-based
        metrics["model_loaded"] = True # Still "loaded" but in a degraded state
        metrics["model_type"] = "Rule-based (Initialization error)"
        logger.warning("Fell back to rule-based quality assessment due to initialization error.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting Advanced Face Quality Assessment Service...")
    initialize_quality_engine() # Initialize the engine on startup
    logger.info("Advanced face quality assessment engine initialization process completed.")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Advanced Face Quality Assessment Service...")

app = FastAPI(
    title="Advanced Face Quality Assessment Service",
    description="AI microservice for comprehensive and advanced face quality analysis using ONNX and rule-based systems.",
    version="2.0.0", # Updated version
    lifespan=lifespan
)

# Helper to convert ModelQualityMetrics to API's FaceQualityMetrics
def _convert_metrics_to_response(model_metrics: ModelQualityMetrics, advanced_results: Dict[str, Any]) -> FaceQualityMetrics:
    # Basic mapping
    response_metrics = FaceQualityMetrics(
        overall_score=model_metrics.overall_quality_score,
        sharpness=model_metrics.sharpness,
        brightness=model_metrics.brightness,
        contrast=model_metrics.contrast,
        face_angle={"yaw": advanced_results.get("head_pose", {}).get("yaw", 0), 
                    "pitch": advanced_results.get("head_pose", {}).get("pitch", 0), 
                    "roll": advanced_results.get("head_pose", {}).get("roll", 0)}, # Map from head_pose
        occlusion=advanced_results.get("occlusion_scores", {"left_eye": 0.0, "right_eye": 0.0, "nose": 0.0, "mouth": 0.0}), # Map from advanced_results
        # Fields from original app.py that might need mapping if still relevant
        # These might be directly available or need calculation based on new model_metrics
        blur_score=model_metrics.sharpness, # Example: map blur_score to sharpness or a dedicated blur metric if available
        noise_level=1.0 - model_metrics.noise, # Example: map noise (0-1, higher is more noise) to noise_level (0-1, higher is less noise)
        illumination_quality=model_metrics.brightness, # Example: map to brightness or a dedicated illumination metric
        resolution_adequacy=1.0 if model_metrics.resolution > 100*100 else model_metrics.resolution/(100*100), # Example mapping
        # New fields
        head_pose=advanced_results.get("head_pose"),
        eye_status=advanced_results.get("eye_status"),
        mouth_status=advanced_results.get("mouth_status"),
        glasses_detected=advanced_results.get("glasses_detected"),
        hat_detected=advanced_results.get("hat_detected"),
        mask_detected=advanced_results.get("mask_detected"),
        skin_tone_uniformity=advanced_results.get("skin_tone_uniformity"),
        detail_level=advanced_results.get("detail_level"),
        color_balance=advanced_results.get("color_balance")
    )
    return response_metrics

async def process_quality_assessment_advanced(image_data: bytes, face_boxes: Optional[List[Dict]] = None, rules_config: Optional[Dict] = None) -> QualityResponse:
    """Process face quality assessment using AdvancedQualityAssessor"""
    global metrics, quality_engine
    if quality_engine is None:
        logger.error("Quality engine not initialized.")
        raise HTTPException(status_code=503, detail="Service not ready, quality engine unavailable.")

    start_time = time.time()
    
    try:
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        results = []
        
        # If no face boxes provided, AdvancedQualityAssessor might detect them or use whole image
        # For now, let's assume if face_boxes is None, we process the whole image as one "face"
        # or rely on internal detection if the model supports it.
        # The AdvancedQualityAssessor is designed to take a full image and find faces.
        
        # If face_boxes are provided, iterate through them.
        # Otherwise, assess the whole image.
        if face_boxes:
            # This part needs to align with how AdvancedQualityAssessor expects input.
            # If it processes one face ROI at a time:
            for i, box in enumerate(face_boxes):
                try:
                    x, y, w, h = int(box["x"]), int(box["y"]), int(box["width"]), int(box["height"])
                    face_image_roi = image[y:y+h, x:x+w]
                    
                    if face_image_roi.size == 0:
                        logger.warning(f"Skipping empty face ROI for box: {box}")
                        continue
                    
                    # Assess quality for the ROI
                    # Note: AdvancedQualityAssessor.assess_quality expects full image and bbox.
                    # We might need a different method or adapt. For now, let's assume it can take ROI.
                    # This is a simplification. Ideally, pass full image and bboxes to AdvancedQualityAssessor.
                    model_metrics, advanced_details, recommendations = quality_engine.assess_quality(face_image_roi, bbox=[0,0,w,h], rules_config=rules_config)

                    api_metrics = _convert_metrics_to_response(model_metrics, advanced_details)
                    quality_category = quality_engine.categorize_quality(model_metrics.overall_quality_score)
                    
                    results.append(QualityAssessmentResult(
                        face_id=i,
                        bbox=box,
                        quality_metrics=api_metrics,
                        quality_category=quality_category,
                        recommendations=recommendations
                    ))
                    metrics["quality_distribution"][quality_category] = metrics["quality_distribution"].get(quality_category, 0) + 1
                except Exception as e_face:
                    logger.error(f"Error processing face {i} with box {box}: {e_face}", exc_info=True)
                    # Add a placeholder or skip
                    continue
        else: # No face_boxes provided, assess the whole image
            # This assumes AdvancedQualityAssessor can find faces or assess the dominant face.
            # The `assess_quality` method in `AdvancedQualityAssessor` takes an image and an optional bbox.
            # If bbox is None, it should ideally run detection or assume a single face.
            # For simplicity, let's assume it processes the primary face or the whole image if no bbox.
            # This needs to be robustly handled by AdvancedQualityAssessor.
            # Let's assume it will try to detect one face or use the whole image.
            
            # The provided AdvancedQualityAssessor seems to take a single face image (ROI) or full image + bbox.
            # If we want it to detect faces, that logic should be inside it or called before.
            # For now, let's assume if no boxes, we try to assess the whole image as a single face.
            # This is likely not the ideal scenario for multiple faces.
            # A better approach: if no boxes, run a face detector first.
            # For now, let's call assess_quality with the full image and no bbox.
            # The `AdvancedQualityAssessor`'s `assess_quality` method expects a `bbox` if provided.
            # If not, it might process the whole image. Let's assume it does.
            # This is a placeholder for potentially more complex logic if face detection is needed here.
            
            # Let's refine this: if no face_boxes, we should run face detection first.
            # The AdvancedQualityAssessor has a `face_quality_onnx.detect_faces` method.
            
            detected_faces = quality_engine.face_quality_onnx.detect_faces(image)
            if not detected_faces:
                logger.info("No faces detected in the image by ONNX model.")
                # Fallback to a simpler assessment or return empty if desired
                # For now, let's try a rule-based assessment on the whole image if no faces detected
                # This is a placeholder for better handling
                model_metrics, advanced_details, recommendations = quality_engine.assess_quality_rules_only(image, rules_config=rules_config)
                api_metrics = _convert_metrics_to_response(model_metrics, advanced_details)
                quality_category = quality_engine.categorize_quality(model_metrics.overall_quality_score)
                results.append(QualityAssessmentResult(
                    face_id=0,
                    bbox={"x":0, "y":0, "width": image.shape[1], "height": image.shape[0]}, # Full image
                    quality_metrics=api_metrics,
                    quality_category=quality_category,
                    recommendations=recommendations
                ))
                metrics["quality_distribution"][quality_category] = metrics["quality_distribution"].get(quality_category, 0) + 1

            else:
                for i, (bbox_coords, landmarks, score) in enumerate(detected_faces):
                    try:
                        x1, y1, x2, y2 = map(int, bbox_coords)
                        w, h = x2 - x1, y2 - y1
                        face_image_roi = image[y1:y2, x1:x2]

                        if face_image_roi.size == 0:
                            logger.warning(f"Skipping empty face ROI for detected face: {bbox_coords}")
                            continue
                        
                        # Pass the detected bbox to assess_quality
                        current_bbox_dict = {"x": float(x1), "y": float(y1), "width": float(w), "height": float(h)}
                        model_metrics, advanced_details, recommendations = quality_engine.assess_quality(
                            image, # Pass the full image
                            bbox=bbox_coords, # Pass the detected bounding box
                            landmarks=landmarks, # Pass landmarks if available
                            rules_config=rules_config
                        )

                        api_metrics = _convert_metrics_to_response(model_metrics, advanced_details)
                        quality_category = quality_engine.categorize_quality(model_metrics.overall_quality_score)
                        
                        results.append(QualityAssessmentResult(
                            face_id=i,
                            bbox=current_bbox_dict,
                            quality_metrics=api_metrics,
                            quality_category=quality_category,
                            recommendations=recommendations
                        ))
                        metrics["quality_distribution"][quality_category] = metrics["quality_distribution"].get(quality_category, 0) + 1
                    except Exception as e_face_detected:
                        logger.error(f"Error processing detected face {i} with bbox {bbox_coords}: {e_face_detected}", exc_info=True)
                        continue


        processing_time = time.time() - start_time
        
        average_quality = sum(r.quality_metrics.overall_score for r in results) / len(results) if results else 0.0
        
        metrics["total_requests"] += 1
        metrics["total_faces_processed"] += len(results)
        metrics["total_processing_time"] += processing_time
        
        return QualityResponse(
            faces=results,
            processing_time=processing_time,
            average_quality=average_quality,
            model_info=f"Advanced Face Quality Assessment v2.0 ({metrics.get('model_type', 'Unknown')})"
        )
        
    except HTTPException: # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error in advanced quality assessment: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


# --- API Endpoints ---

@app.post("/assess", response_model=QualityResponse)
async def assess_quality_single_image_legacy( # Renamed to avoid conflict if keeping old one
    file: UploadFile = File(...),
    face_boxes: Optional[str] = Form(None) # JSON string of List[Dict]
):
    """
    Assess face quality in an uploaded image.
    This endpoint is similar to the original /assess but uses the new engine.
    `face_boxes`: Optional JSON string representing a list of bounding boxes.
                  Example: `[{"x": 10, "y": 10, "width": 100, "height": 100}]`
    """
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        image_data = await file.read()
        
        parsed_boxes = None
        if face_boxes:
            try:
                parsed_boxes = json.loads(face_boxes)
                if not isinstance(parsed_boxes, list):
                    raise ValueError("face_boxes must be a list of dictionaries.")
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid JSON format for face_boxes.")
            except ValueError as ve:
                 raise HTTPException(status_code=400, detail=str(ve))

        # Use the new advanced processing function
        result = await process_quality_assessment_advanced(image_data, parsed_boxes)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /assess endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/assess/advanced", response_model=QualityResponse)
async def assess_quality_advanced_endpoint(
    file: UploadFile = File(...),
    face_boxes: Optional[str] = Form(None, description="JSON string of List[Dict] for face bounding boxes. If None, face detection will be attempted."),
    rules_config: Optional[str] = Form(None, description="JSON string for custom rule thresholds and weights.")
):
    """
    Assess face quality with advanced options and ONNX model.
    - `file`: Uploaded image file.
    - `face_boxes`: Optional. JSON string `[{"x":_,"y":_,"width":_,"height":_}, ...]`. If not provided, ONNX model will attempt to detect faces.
    - `rules_config`: Optional. JSON string for custom rule engine parameters. Example: `{"thresholds": {"sharpness": 0.4}, "weights": {"sharpness": 0.3}}`
    """
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_data = await file.read()
    
    parsed_boxes = None
    if face_boxes:
        try:
            parsed_boxes = json.loads(face_boxes)
            if not isinstance(parsed_boxes, list): # Basic validation
                raise ValueError("face_boxes must be a list of dictionaries.")
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON in face_boxes.")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid face_boxes structure: {e}")

    parsed_rules_config = None
    if rules_config:
        try:
            parsed_rules_config = json.loads(rules_config)
            if not isinstance(parsed_rules_config, dict): # Basic validation
                 raise ValueError("rules_config must be a dictionary.")
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON in rules_config.")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid rules_config structure: {e}")
            
    try:
        result = await process_quality_assessment_advanced(image_data, parsed_boxes, parsed_rules_config)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /assess/advanced endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/assess/batch", response_model=BatchQualityResponse)
async def assess_quality_batch(files: List[UploadFile] = File(...)):
    """Assess face quality in multiple images using the advanced engine."""
    if len(files) > 10: # Keep batch limit reasonable
        raise HTTPException(status_code=400, detail="Maximum 10 images per batch")
    
    start_batch_time = time.time()
    batch_results = []
    total_faces_processed_in_batch = 0
    aggregated_quality_dist = {"excellent": 0, "good": 0, "fair": 0, "poor": 0}
    
    for file in files:
        if not file.content_type.startswith('image/'):
            logger.warning(f"Skipping non-image file: {file.filename}")
            # Create a dummy error response or skip
            # For now, skipping.
            continue
        
        try:
            image_data = await file.read()
            # Using advanced assessment for each image in batch
            # No face_boxes or rules_config passed here, so it will use defaults (detect faces)
            single_image_response = await process_quality_assessment_advanced(image_data, None, None)
            batch_results.append(single_image_response)
            total_faces_processed_in_batch += len(single_image_response.faces)
            
            for face_res in single_image_response.faces:
                aggregated_quality_dist[face_res.quality_category] = aggregated_quality_dist.get(face_res.quality_category, 0) + 1
                    
        except Exception as e: # Catch errors per file to not fail the whole batch
            logger.error(f"Error processing file {file.filename} in batch: {e}", exc_info=True)
            # Optionally, add an error placeholder to batch_results
            # For now, just logging and continuing.
            continue 
            
    total_batch_processing_time = time.time() - start_batch_time
    
    return BatchQualityResponse(
        results=batch_results,
        total_processing_time=total_batch_processing_time,
        total_faces_processed=total_faces_processed_in_batch,
        quality_distribution=aggregated_quality_dist
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for the service."""
    global metrics, quality_engine
    memory_info = {
        "ram_percent": psutil.virtual_memory().percent,
        "ram_available_gb": round(psutil.virtual_memory().available / (1024**3), 2)
    }
    
    gpu_available = False
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0] # Assuming single GPU for simplicity
            memory_info.update({
                "gpu_name": gpu.name,
                "gpu_load": f"{gpu.load*100:.2f}%",
                "gpu_memory_percent": f"{gpu.memoryUtil*100:.2f}%",
                "gpu_memory_used_mb": gpu.memoryUsed,
                "gpu_memory_total_mb": gpu.memoryTotal,
                "gpu_temperature": f"{gpu.temperature} C"
            })
            gpu_available = True
    except Exception as e: # GPUtil might not be installed or no GPU
        logger.debug(f"GPUtil error or no GPU found: {e}")
        pass # Silently ignore if GPUtil fails
    
    uptime = time.time() - metrics["start_time"]
    
    return HealthResponse(
        status="healthy" if metrics["model_loaded"] else "degraded",
        service="Advanced Face Quality Assessment",
        gpu_available=gpu_available,
        memory_usage=memory_info,
        uptime=uptime,
        model_loaded=metrics["model_loaded"],
        model_type=metrics.get("model_type", "Unknown")
    )

@app.get("/metrics")
async def get_metrics_endpoint(): # Renamed to avoid conflict
    """Get service operational metrics."""
    global metrics
    avg_processing_time = (metrics["total_processing_time"] / metrics["total_requests"] 
                          if metrics["total_requests"] > 0 else 0)
    
    return {
        "total_requests": metrics["total_requests"],
        "total_faces_processed": metrics["total_faces_processed"],
        "average_processing_time_seconds": round(avg_processing_time, 4),
        "quality_distribution": metrics["quality_distribution"],
        "uptime_seconds": round(time.time() - metrics["start_time"], 2),
        "model_loaded": metrics["model_loaded"],
        "model_type": metrics.get("model_type", "Unknown")
    }

@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_information():
    """Get information about the loaded quality assessment model."""
    global quality_engine, metrics
    if quality_engine is None or not metrics["model_loaded"]:
        raise HTTPException(status_code=503, detail="Model not available or not loaded.")

    onnx_path = None
    onnx_providers = None
    if quality_engine.face_quality_onnx and quality_engine.face_quality_onnx.session:
        onnx_path = quality_engine.face_quality_onnx.model_path
        onnx_providers = quality_engine.face_quality_onnx.providers

    return ModelInfoResponse(
        model_name="AdvancedQualityAssessor",
        model_version="2.0.0", # Corresponds to app version using this model
        model_type=metrics.get("model_type", "ONNX-based with rule-engine"),
        supported_metrics=list(ModelQualityMetrics.__fields__.keys()) + [
            "head_pose", "eye_status", "mouth_status", "glasses_detected", 
            "hat_detected", "mask_detected", "skin_tone_uniformity", 
            "detail_level", "color_balance" # Add other advanced metrics
        ],
        onnx_model_path=onnx_path,
        onnx_providers=onnx_providers
    )

@app.get("/benchmark")
async def benchmark_quality_assessment(
    image_url: Optional[str] = Query(None, description="URL of an image to download and test."),
    iterations: int = Query(10, ge=1, le=100, description="Number of times to run the assessment.")
):
    """
    Run a quick benchmark on the quality assessment process.
    Uses a default test image if no image_url is provided.
    """
    global quality_engine
    if quality_engine is None:
        raise HTTPException(status_code=503, detail="Service not ready, quality engine unavailable.")

    timings = []
    image_to_test = None
    image_source = ""

    if image_url:
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.get(image_url)
                response.raise_for_status()
                image_data = response.content
                nparr = np.frombuffer(image_data, np.uint8)
                image_to_test = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if image_to_test is None:
                    raise ValueError("Could not decode image from URL.")
                image_source = f"URL: {image_url}"
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to load image from URL: {e}")
    else:
        # Create a dummy image for benchmarking if no URL
        image_to_test = np.zeros((640, 480, 3), dtype=np.uint8)
        cv2.putText(image_to_test, "Test", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        image_source = "Default 640x480 blank image"
        # Encode to bytes to simulate file upload
        _, buffer = cv2.imencode('.jpg', image_to_test)
        image_data = buffer.tobytes()


    logger.info(f"Starting benchmark with {iterations} iterations. Image source: {image_source}")

    # Warm-up run (optional, but good for JIT, etc.)
    try:
        await process_quality_assessment_advanced(image_data, None, None)
    except Exception:
        pass # Ignore errors in warm-up

    for i in range(iterations):
        start_iter = time.perf_counter()
        try:
            # Using the advanced process. This will include detection if no boxes are given.
            await process_quality_assessment_advanced(image_data, None, None)
        except Exception as e:
            logger.error(f"Error during benchmark iteration {i}: {e}")
            # Decide if benchmark should fail or just record a failed iteration
            timings.append(float('inf')) # Indicate failure for this iteration
            continue
        end_iter = time.perf_counter()
        timings.append(end_iter - start_iter)

    if not timings:
        return {"error": "Benchmark failed to run any iterations."}
        
    return {
        "iterations": iterations,
        "image_source": image_source,
        "total_time_seconds": sum(t for t in timings if t != float('inf')),
        "average_time_per_iteration_ms": (sum(t for t in timings if t != float('inf')) / len([t for t in timings if t != float('inf')])) * 1000 if len([t for t in timings if t != float('inf')]) > 0 else "N/A",
        "min_time_ms": min(t for t in timings if t != float('inf')) * 1000 if len([t for t in timings if t != float('inf')]) > 0 else "N/A",
        "max_time_ms": max(t for t in timings if t != float('inf')) * 1000 if len([t for t in timings if t != float('inf')]) > 0 else "N/A",
        "successful_iterations": len([t for t in timings if t != float('inf')]),
        "timings_ms": [t * 1000 if t != float('inf') else "Failed" for t in timings]
    }


@app.get("/")
async def root():
    """Root endpoint providing service information."""
    return {
        "message": "Advanced Face Quality Assessment Service", 
        "status": "running",
        "version": app.version,
        "docs_url": "/docs",
        "model_type": metrics.get("model_type", "Unknown")
    }

if __name__ == "__main__":
    import uvicorn
    # Recommended: Read host and port from environment variables for flexibility
    app_host = os.getenv("FACE_QUALITY_HOST", "0.0.0.0")
    app_port = int(os.getenv("FACE_QUALITY_PORT", "8008"))
    log_level = os.getenv("FACE_QUALITY_LOG_LEVEL", "info").lower()
    
    uvicorn.run(app, host=app_host, port=app_port, log_level=log_level)
