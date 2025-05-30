"""
Face Quality Assessment Service
FastAPI microservice for comprehensive face quality analysis
Supports sharpness, brightness, contrast, face angle, occlusion detection
"""

import os
import cv2
import numpy as np
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
from scipy import ndimage
from skimage import feature, measure

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models
class FaceQualityMetrics(BaseModel):
    overall_score: float
    sharpness: float
    brightness: float
    contrast: float
    face_angle: Dict[str, float]  # yaw, pitch, roll
    occlusion: Dict[str, float]  # left_eye, right_eye, nose, mouth
    blur_score: float
    noise_level: float
    illumination_quality: float
    resolution_adequacy: float

class QualityAssessmentResult(BaseModel):
    face_id: int
    bbox: Dict[str, float]
    quality_metrics: FaceQualityMetrics
    quality_category: str  # "excellent", "good", "fair", "poor"
    recommendations: List[str]

class QualityResponse(BaseModel):
    faces: List[QualityAssessmentResult]
    processing_time: float
    average_quality: float
    model_info: str

class BatchQualityResponse(BaseModel):
    results: List[QualityResponse]
    total_processing_time: float
    total_faces_processed: int
    quality_distribution: Dict[str, int]

class HealthResponse(BaseModel):
    status: str
    service: str
    gpu_available: bool
    memory_usage: Dict[str, Any]
    uptime: float

# Global metrics
metrics = {
    "total_requests": 0,
    "total_faces_processed": 0,
    "total_processing_time": 0.0,
    "quality_distribution": {"excellent": 0, "good": 0, "fair": 0, "poor": 0},
    "start_time": time.time()
}

class FaceQualityAssessment:
    """Comprehensive face quality assessment engine"""
    
    def __init__(self):
        self.min_face_size = 50
        self.optimal_face_size = 150
        self.setup_detectors()
        
    def setup_detectors(self):
        """Setup detection components"""
        try:
            # Setup face cascade for basic detection
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            # Setup eye cascade for occlusion detection
            eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
            self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
            
            logger.info("Face quality detectors initialized successfully")
        except Exception as e:
            logger.error(f"Failed to setup detectors: {e}")
            raise
    
    def assess_face_quality(self, face_image: np.ndarray, bbox: Dict[str, float] = None) -> QualityAssessmentResult:
        """Assess comprehensive face quality"""
        try:
            # Basic quality metrics
            sharpness = self._calculate_sharpness(face_image)
            brightness = self._calculate_brightness(face_image)
            contrast = self._calculate_contrast(face_image)
            
            # Advanced quality metrics
            blur_score = self._calculate_blur_score(face_image)
            noise_level = self._calculate_noise_level(face_image)
            illumination_quality = self._calculate_illumination_quality(face_image)
            resolution_adequacy = self._calculate_resolution_adequacy(face_image)
            
            # Face angle estimation
            face_angle = self._estimate_face_angle(face_image)
            
            # Occlusion detection
            occlusion = self._detect_occlusion(face_image)
            
            # Calculate overall quality score
            overall_score = self._calculate_overall_quality(
                sharpness, brightness, contrast, blur_score, 
                noise_level, illumination_quality, resolution_adequacy, face_angle, occlusion
            )
            
            # Create quality metrics object
            quality_metrics = FaceQualityMetrics(
                overall_score=overall_score,
                sharpness=sharpness,
                brightness=brightness,
                contrast=contrast,
                face_angle=face_angle,
                occlusion=occlusion,
                blur_score=blur_score,
                noise_level=noise_level,
                illumination_quality=illumination_quality,
                resolution_adequacy=resolution_adequacy
            )
            
            # Determine quality category
            quality_category = self._determine_quality_category(overall_score)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(quality_metrics)
            
            return QualityAssessmentResult(
                face_id=0,
                bbox=bbox or {"x": 0, "y": 0, "width": face_image.shape[1], "height": face_image.shape[0]},
                quality_metrics=quality_metrics,
                quality_category=quality_category,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Error in face quality assessment: {e}")
            # Return default quality result
            return QualityAssessmentResult(
                face_id=0,
                bbox=bbox or {"x": 0, "y": 0, "width": face_image.shape[1], "height": face_image.shape[0]},
                quality_metrics=FaceQualityMetrics(
                    overall_score=0.0,
                    sharpness=0.0,
                    brightness=0.0,
                    contrast=0.0,
                    face_angle={"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
                    occlusion={"left_eye": 0.0, "right_eye": 0.0, "nose": 0.0, "mouth": 0.0},
                    blur_score=0.0,
                    noise_level=1.0,
                    illumination_quality=0.0,
                    resolution_adequacy=0.0
                ),
                quality_category="poor",
                recommendations=["Error in quality assessment"]
            )
    
    def _calculate_sharpness(self, image: np.ndarray) -> float:
        """Calculate image sharpness using Laplacian variance"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Normalize to 0-1 range (empirically determined threshold)
            normalized_sharpness = min(laplacian_var / 1000.0, 1.0)
            return float(normalized_sharpness)
        except Exception:
            return 0.0
    
    def _calculate_brightness(self, image: np.ndarray) -> float:
        """Calculate image brightness"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            mean_brightness = np.mean(gray) / 255.0
            
            # Optimal brightness is around 0.4-0.7, penalize extremes
            if 0.4 <= mean_brightness <= 0.7:
                return 1.0
            elif mean_brightness < 0.4:
                return mean_brightness / 0.4
            else:
                return (1.0 - mean_brightness) / 0.3
        except Exception:
            return 0.0
    
    def _calculate_contrast(self, image: np.ndarray) -> float:
        """Calculate image contrast"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            contrast = gray.std() / 255.0
            
            # Normalize to 0-1 range (good contrast is around 0.3-0.7)
            normalized_contrast = min(contrast / 0.5, 1.0)
            return float(normalized_contrast)
        except Exception:
            return 0.0
    
    def _calculate_blur_score(self, image: np.ndarray) -> float:
        """Calculate blur score using multiple methods"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Method 1: Laplacian variance
            laplacian_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Method 2: Sobel variance
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            sobel_score = np.sqrt(sobelx**2 + sobely**2).var()
            
            # Combine scores
            combined_score = (laplacian_score + sobel_score) / 2
            normalized_score = min(combined_score / 1000.0, 1.0)
            
            return float(normalized_score)
        except Exception:
            return 0.0
    
    def _calculate_noise_level(self, image: np.ndarray) -> float:
        """Calculate noise level in image"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Use Gaussian blur to estimate noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            noise = cv2.absdiff(gray, blurred)
            noise_level = np.mean(noise) / 255.0
            
            # Return inverted score (lower noise = higher quality)
            return float(1.0 - min(noise_level * 2, 1.0))
        except Exception:
            return 0.0
    
    def _calculate_illumination_quality(self, image: np.ndarray) -> float:
        """Calculate illumination quality"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Calculate histogram
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist_norm = hist.ravel() / hist.sum()
            
            # Check for even distribution (good illumination)
            # Penalize images with too many dark or bright pixels
            dark_pixels = np.sum(hist_norm[:64])  # Very dark
            bright_pixels = np.sum(hist_norm[192:])  # Very bright
            
            illumination_score = 1.0 - (dark_pixels + bright_pixels)
            return float(max(illumination_score, 0.0))
        except Exception:
            return 0.0
    
    def _calculate_resolution_adequacy(self, image: np.ndarray) -> float:
        """Calculate if resolution is adequate for face recognition"""
        try:
            height, width = image.shape[:2]
            total_pixels = height * width
            
            # Minimum resolution thresholds
            if total_pixels < self.min_face_size ** 2:
                return total_pixels / (self.min_face_size ** 2)
            elif total_pixels >= self.optimal_face_size ** 2:
                return 1.0
            else:
                return total_pixels / (self.optimal_face_size ** 2)
        except Exception:
            return 0.0
    
    def _estimate_face_angle(self, image: np.ndarray) -> Dict[str, float]:
        """Estimate face pose angles (simplified method)"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Detect face for angle estimation
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
            
            if len(faces) > 0:
                # Simple angle estimation based on face symmetry
                x, y, w, h = faces[0]
                face_roi = gray[y:y+h, x:x+w]
                
                # Estimate yaw (left-right rotation)
                left_half = face_roi[:, :w//2]
                right_half = face_roi[:, w//2:]
                right_half_flipped = cv2.flip(right_half, 1)
                
                # Calculate symmetry score
                if left_half.shape == right_half_flipped.shape:
                    symmetry = cv2.matchTemplate(left_half, right_half_flipped, cv2.TM_CCOEFF_NORMED)[0][0]
                    yaw = (1.0 - symmetry) * 30  # Convert to approximate degrees
                else:
                    yaw = 15.0  # Default moderate angle
                
                # Simplified pitch and roll estimation
                pitch = abs(h - w) / max(h, w) * 20  # Based on aspect ratio
                roll = 0.0  # Simplified - would need more complex detection
                
                return {
                    "yaw": float(min(abs(yaw), 45.0)),
                    "pitch": float(min(abs(pitch), 30.0)),
                    "roll": float(roll)
                }
            else:
                return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}
        except Exception:
            return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}
    
    def _detect_occlusion(self, image: np.ndarray) -> Dict[str, float]:
        """Detect facial feature occlusion"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Detect eyes
            eyes = self.eye_cascade.detectMultiScale(gray, 1.1, 5, minSize=(10, 10))
            
            # Simple occlusion estimation
            occlusion_scores = {
                "left_eye": 0.0,
                "right_eye": 0.0,
                "nose": 0.0,
                "mouth": 0.0
            }
            
            if len(eyes) >= 2:
                # If we can detect 2 or more eyes, assume low occlusion
                occlusion_scores["left_eye"] = 0.1
                occlusion_scores["right_eye"] = 0.1
            elif len(eyes) == 1:
                # One eye might be occluded
                occlusion_scores["left_eye"] = 0.5
                occlusion_scores["right_eye"] = 0.1
            else:
                # High occlusion likely
                occlusion_scores["left_eye"] = 0.8
                occlusion_scores["right_eye"] = 0.8
            
            # Simplified nose and mouth occlusion (would need more sophisticated detection)
            occlusion_scores["nose"] = 0.1
            occlusion_scores["mouth"] = 0.1
            
            return occlusion_scores
        except Exception:
            return {"left_eye": 0.0, "right_eye": 0.0, "nose": 0.0, "mouth": 0.0}
    
    def _calculate_overall_quality(self, sharpness: float, brightness: float, contrast: float,
                                 blur_score: float, noise_level: float, illumination_quality: float,
                                 resolution_adequacy: float, face_angle: Dict[str, float],
                                 occlusion: Dict[str, float]) -> float:
        """Calculate overall quality score"""
        try:
            # Weight factors for different quality aspects
            weights = {
                "sharpness": 0.20,
                "brightness": 0.15,
                "contrast": 0.15,
                "blur_score": 0.15,
                "noise_level": 0.10,
                "illumination_quality": 0.10,
                "resolution_adequacy": 0.10,
                "pose_quality": 0.05
            }
            
            # Calculate pose quality (lower angles = higher quality)
            max_angle = max(face_angle["yaw"], face_angle["pitch"], face_angle["roll"])
            pose_quality = max(0.0, 1.0 - (max_angle / 45.0))
            
            # Calculate occlusion penalty
            avg_occlusion = np.mean(list(occlusion.values()))
            occlusion_penalty = avg_occlusion * 0.3  # Max penalty of 0.3
            
            # Weighted sum
            overall_score = (
                sharpness * weights["sharpness"] +
                brightness * weights["brightness"] +
                contrast * weights["contrast"] +
                blur_score * weights["blur_score"] +
                noise_level * weights["noise_level"] +
                illumination_quality * weights["illumination_quality"] +
                resolution_adequacy * weights["resolution_adequacy"] +
                pose_quality * weights["pose_quality"]
            )
            
            # Apply occlusion penalty
            overall_score = max(0.0, overall_score - occlusion_penalty)
            
            return float(min(overall_score, 1.0))
        except Exception:
            return 0.0
    
    def _determine_quality_category(self, overall_score: float) -> str:
        """Determine quality category based on overall score"""
        if overall_score >= 0.8:
            return "excellent"
        elif overall_score >= 0.6:
            return "good"
        elif overall_score >= 0.4:
            return "fair"
        else:
            return "poor"
    
    def _generate_recommendations(self, quality_metrics: FaceQualityMetrics) -> List[str]:
        """Generate quality improvement recommendations"""
        recommendations = []
        
        if quality_metrics.sharpness < 0.5:
            recommendations.append("Image is blurry - use better focus or stabilization")
        
        if quality_metrics.brightness < 0.4:
            recommendations.append("Image is too dark - increase lighting")
        elif quality_metrics.brightness > 0.8:
            recommendations.append("Image is too bright - reduce lighting or adjust exposure")
        
        if quality_metrics.contrast < 0.3:
            recommendations.append("Low contrast - improve lighting conditions")
        
        if quality_metrics.resolution_adequacy < 0.6:
            recommendations.append("Face too small - move closer to camera or increase resolution")
        
        max_angle = max(quality_metrics.face_angle["yaw"], 
                       quality_metrics.face_angle["pitch"], 
                       quality_metrics.face_angle["roll"])
        if max_angle > 20:
            recommendations.append("Face angle not optimal - face camera more directly")
        
        avg_occlusion = np.mean(list(quality_metrics.occlusion.values()))
        if avg_occlusion > 0.3:
            recommendations.append("Facial features may be occluded - ensure clear view of face")
        
        if quality_metrics.noise_level < 0.7:
            recommendations.append("Image has noise - use better camera or lighting conditions")
        
        if not recommendations:
            recommendations.append("Face quality is good for recognition")
        
        return recommendations

# Initialize quality assessment engine
quality_engine = FaceQualityAssessment()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting Face Quality Assessment Service...")
    logger.info("Face quality assessment engine initialized")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Face Quality Assessment Service...")

app = FastAPI(
    title="Face Quality Assessment Service",
    description="AI microservice for comprehensive face quality analysis",
    version="1.0.0",
    lifespan=lifespan
)

async def process_quality_assessment(image_data: bytes, face_boxes: List[Dict] = None) -> QualityResponse:
    """Process face quality assessment"""
    global metrics
    start_time = time.time()
    
    try:
        # Decode image
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        results = []
        
        # If no face boxes provided, use whole image
        if face_boxes is None:
            face_boxes = [{"x": 0, "y": 0, "width": image.shape[1], "height": image.shape[0]}]
        
        for i, box in enumerate(face_boxes):
            try:
                # Extract face region
                x, y, w, h = int(box["x"]), int(box["y"]), int(box["width"]), int(box["height"])
                face_image = image[y:y+h, x:x+w]
                
                if face_image.size == 0:
                    continue
                
                # Assess quality
                quality_result = quality_engine.assess_face_quality(face_image, box)
                quality_result.face_id = i
                results.append(quality_result)
                
                # Update metrics
                metrics["quality_distribution"][quality_result.quality_category] += 1
                
            except Exception as e:
                logger.error(f"Error processing face {i}: {e}")
                continue
        
        processing_time = time.time() - start_time
        
        # Calculate average quality
        if results:
            average_quality = sum(r.quality_metrics.overall_score for r in results) / len(results)
        else:
            average_quality = 0.0
        
        # Update metrics
        metrics["total_requests"] += 1
        metrics["total_faces_processed"] += len(results)
        metrics["total_processing_time"] += processing_time
        
        return QualityResponse(
            faces=results,
            processing_time=processing_time,
            average_quality=average_quality,
            model_info="Comprehensive Face Quality Assessment v1.0"
        )
        
    except Exception as e:
        logger.error(f"Error in quality assessment: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@app.post("/assess", response_model=QualityResponse)
async def assess_quality(
    file: UploadFile = File(...),
    face_boxes: Optional[str] = Form(None)
):
    """Assess face quality in uploaded image"""
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        image_data = await file.read()
        
        # Parse face boxes if provided
        boxes = None
        if face_boxes:
            boxes = json.loads(face_boxes)
        
        result = await process_quality_assessment(image_data, boxes)
        return result
    except Exception as e:
        logger.error(f"Error in assess endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/assess/batch", response_model=BatchQualityResponse)
async def assess_quality_batch(files: List[UploadFile] = File(...)):
    """Assess face quality in multiple images"""
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 images per batch")
    
    start_time = time.time()
    results = []
    total_faces = 0
    quality_dist = {"excellent": 0, "good": 0, "fair": 0, "poor": 0}
    
    for file in files:
        if not file.content_type.startswith('image/'):
            continue
        
        try:
            image_data = await file.read()
            result = await process_quality_assessment(image_data)
            results.append(result)
            total_faces += len(result.faces)
            
            # Aggregate quality distribution
            for face in result.faces:
                quality_dist[face.quality_category] += 1
                    
        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {e}")
            continue
    
    total_time = time.time() - start_time
    
    return BatchQualityResponse(
        results=results,
        total_processing_time=total_time,
        total_faces_processed=total_faces,
        quality_distribution=quality_dist
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    memory_info = {
        "ram_percent": psutil.virtual_memory().percent,
        "ram_available_gb": psutil.virtual_memory().available / (1024**3)
    }
    
    # Add GPU info if available
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            memory_info.update({
                "gpu_memory_percent": gpu.memoryUtil * 100,
                "gpu_memory_used_mb": gpu.memoryUsed,
                "gpu_memory_total_mb": gpu.memoryTotal
            })
    except:
        pass
    
    uptime = time.time() - metrics["start_time"]
    
    return HealthResponse(
        status="healthy",
        service="Face Quality Assessment",
        gpu_available=len(GPUtil.getGPUs()) > 0 if GPUtil else False,
        memory_usage=memory_info,
        uptime=uptime
    )

@app.get("/metrics")
async def get_metrics():
    """Get service metrics"""
    avg_processing_time = (metrics["total_processing_time"] / metrics["total_requests"] 
                          if metrics["total_requests"] > 0 else 0)
    
    return {
        "total_requests": metrics["total_requests"],
        "total_faces_processed": metrics["total_faces_processed"],
        "average_processing_time": avg_processing_time,
        "quality_distribution": metrics["quality_distribution"]
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Face Quality Assessment Service", "status": "running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8008)
