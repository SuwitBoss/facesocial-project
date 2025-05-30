"""
Unified Face Detection Response Models
Standardized data models for face detection pipeline
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum

class FaceDetectionMethod(str, Enum):
    MEDIAPIPE = "mediapipe"
    YOLO = "yolo" 
    MTCNN = "mtcnn"
    ENSEMBLE = "ensemble"

class QualityCategory(str, Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"

class ProcessingStage(str, Enum):
    DETECTION = "detection"
    QUALITY_ASSESSMENT = "quality_assessment"
    FACE_ATTRIBUTES = "face_attributes"
    ALIGNMENT = "alignment"
    CROPPING = "cropping"

# Core Face Data Models
class BoundingBox(BaseModel):
    x: float = Field(..., description="X coordinate of top-left corner")
    y: float = Field(..., description="Y coordinate of top-left corner")
    width: float = Field(..., description="Width of bounding box")
    height: float = Field(..., description="Height of bounding box")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")

class FaceLandmark(BaseModel):
    name: str = Field(..., description="Landmark name (e.g., 'left_eye', 'nose')")
    x: float = Field(..., description="X coordinate")
    y: float = Field(..., description="Y coordinate")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)

class FaceLandmarks(BaseModel):
    type: str = Field(..., description="Landmark type: '5_point', '68_point', '106_point'")
    points: List[FaceLandmark] = Field(..., description="Array of landmark points")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall landmarks confidence")

class FaceAngle(BaseModel):
    yaw: float = Field(..., description="Left-right rotation in degrees")
    pitch: float = Field(..., description="Up-down rotation in degrees") 
    roll: float = Field(..., description="Tilt rotation in degrees")

class OcclusionScores(BaseModel):
    left_eye: float = Field(..., ge=0.0, le=1.0, description="Left eye occlusion score")
    right_eye: float = Field(..., ge=0.0, le=1.0, description="Right eye occlusion score")
    nose: float = Field(..., ge=0.0, le=1.0, description="Nose occlusion score")
    mouth: float = Field(..., ge=0.0, le=1.0, description="Mouth occlusion score")

class QualityAssessment(BaseModel):
    overall_score: float = Field(..., ge=0.0, le=1.0, description="Overall quality score")
    sharpness: float = Field(..., ge=0.0, le=1.0, description="Image sharpness score")
    brightness: float = Field(..., ge=0.0, le=1.0, description="Brightness quality score")
    contrast: float = Field(..., ge=0.0, le=1.0, description="Contrast quality score")
    face_angle: FaceAngle = Field(..., description="Face pose angles")
    occlusion: OcclusionScores = Field(..., description="Feature occlusion scores")
    blur_score: float = Field(..., ge=0.0, le=1.0, description="Blur assessment score")
    noise_level: float = Field(..., ge=0.0, le=1.0, description="Noise level (inverted)")
    illumination_quality: float = Field(..., ge=0.0, le=1.0, description="Lighting quality")
    resolution_adequacy: float = Field(..., ge=0.0, le=1.0, description="Resolution adequacy")

class EmotionScores(BaseModel):
    happy: float = Field(..., ge=0.0, le=1.0)
    sad: float = Field(..., ge=0.0, le=1.0)
    angry: float = Field(..., ge=0.0, le=1.0)
    surprised: float = Field(..., ge=0.0, le=1.0)
    fearful: float = Field(..., ge=0.0, le=1.0)
    disgusted: float = Field(..., ge=0.0, le=1.0)
    neutral: float = Field(..., ge=0.0, le=1.0)

class FaceAttributes(BaseModel):
    estimated_age: Optional[float] = Field(None, ge=0.0, le=120.0, description="Estimated age")
    age_range: Optional[Dict[str, int]] = Field(None, description="Age range (min, max)")
    estimated_gender: Optional[str] = Field(None, description="Estimated gender")
    gender_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    dominant_emotion: Optional[str] = Field(None, description="Dominant emotion")
    emotion_scores: Optional[EmotionScores] = Field(None, description="All emotion scores")

# Main Face Detection Result
class DetectedFace(BaseModel):
    face_id: str = Field(..., description="Unique face identifier")
    face_index: int = Field(..., ge=0, description="Face index in image")
    bounding_box: BoundingBox = Field(..., description="Face bounding box")
    landmarks: Optional[FaceLandmarks] = Field(None, description="Facial landmarks")
    quality_assessment: Optional[QualityAssessment] = Field(None, description="Quality metrics")
    face_attributes: Optional[FaceAttributes] = Field(None, description="Face attributes")
    aligned_face: Optional[str] = Field(None, description="Base64 aligned face image")
    cropped_face: Optional[str] = Field(None, description="Base64 cropped face image")
    detection_method: FaceDetectionMethod = Field(..., description="Detection method used")

# Image Information
class ImageInfo(BaseModel):
    width: int = Field(..., gt=0, description="Image width in pixels")
    height: int = Field(..., gt=0, description="Image height in pixels")
    format: str = Field(..., description="Image format (JPEG, PNG, etc.)")
    file_size_bytes: Optional[int] = Field(None, gt=0, description="File size in bytes")

# Processing Statistics
class ProcessingStats(BaseModel):
    detection_time_ms: float = Field(..., ge=0.0, description="Face detection time")
    landmarks_time_ms: Optional[float] = Field(None, ge=0.0, description="Landmarks detection time")
    quality_assessment_time_ms: Optional[float] = Field(None, ge=0.0, description="Quality assessment time")
    attributes_time_ms: Optional[float] = Field(None, ge=0.0, description="Attributes analysis time")
    alignment_time_ms: Optional[float] = Field(None, ge=0.0, description="Face alignment time")
    total_time_ms: float = Field(..., ge=0.0, description="Total processing time")

# Detection Configuration
class DetectionOptions(BaseModel):
    min_face_size: int = Field(40, gt=0, description="Minimum face size in pixels")
    max_faces: int = Field(100, gt=0, description="Maximum faces to detect")
    detection_confidence: float = Field(0.7, ge=0.0, le=1.0, description="Detection confidence threshold")
    return_landmarks: bool = Field(True, description="Return facial landmarks")
    return_attributes: bool = Field(False, description="Return face attributes")
    quality_assessment: bool = Field(True, description="Perform quality assessment")

class ProcessingOptions(BaseModel):
    face_alignment: bool = Field(True, description="Perform face alignment")
    crop_faces: bool = Field(True, description="Return cropped face images")
    enhance_quality: bool = Field(False, description="Apply quality enhancement")
    parallel_processing: bool = Field(True, description="Use parallel processing")

# Request Models
class FaceDetectionRequest(BaseModel):
    image: str = Field(..., description="Base64 encoded image")
    detection_options: DetectionOptions = Field(default_factory=DetectionOptions)
    processing_options: ProcessingOptions = Field(default_factory=ProcessingOptions)
    metadata: Optional[Dict[str, Any]] = Field(None, description="Request metadata")

class BatchDetectionRequest(BaseModel):
    batch_id: Optional[str] = Field(None, description="Batch identifier")
    images: List[Dict[str, Any]] = Field(..., description="Array of image objects")
    detection_options: DetectionOptions = Field(default_factory=DetectionOptions)
    processing_options: ProcessingOptions = Field(default_factory=ProcessingOptions)

# Response Models
class FaceDetectionResponse(BaseModel):
    success: bool = Field(..., description="Request success status")
    data: Dict[str, Any] = Field(..., description="Response data")
    metadata: Dict[str, Any] = Field(..., description="Response metadata")

class DetectionData(BaseModel):
    total_faces: int = Field(..., ge=0, description="Total faces detected")
    image_info: ImageInfo = Field(..., description="Original image information")
    faces: List[DetectedFace] = Field(..., description="Detected faces array")
    processing_stats: ProcessingStats = Field(..., description="Processing statistics")

class BatchDetectionResponse(BaseModel):
    success: bool = Field(..., description="Batch request success status")
    data: Dict[str, Any] = Field(..., description="Batch response data")
    metadata: Dict[str, Any] = Field(..., description="Batch metadata")

class BatchData(BaseModel):
    batch_id: str = Field(..., description="Batch identifier")
    status: str = Field(..., description="Batch processing status")
    total_images: int = Field(..., ge=0, description="Total images in batch")
    completed_images: int = Field(..., ge=0, description="Completed images")
    failed_images: int = Field(..., ge=0, description="Failed images")
    total_faces_detected: int = Field(..., ge=0, description="Total faces across all images")
    results: List[DetectionData] = Field(..., description="Results for each image")

# Error Models
class ErrorDetail(BaseModel):
    error_code: str = Field(..., description="Error code")
    error_message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")

class ErrorResponse(BaseModel):
    success: bool = Field(False, description="Request success status")
    error: ErrorDetail = Field(..., description="Error information")
    metadata: Dict[str, Any] = Field(..., description="Error metadata")

# Service Health Models
class ServiceStatus(BaseModel):
    service_name: str = Field(..., description="Service name")
    status: str = Field(..., description="Service status")
    response_time_ms: Optional[float] = Field(None, description="Service response time")
    available: bool = Field(..., description="Service availability")

class GatewayHealth(BaseModel):
    status: str = Field(..., description="Overall gateway status")
    total_services: int = Field(..., description="Total services count")
    healthy_services: int = Field(..., description="Healthy services count")
    unhealthy_services: int = Field(..., description="Unhealthy services count")
    services: List[ServiceStatus] = Field(..., description="Individual service status")
    uptime: float = Field(..., description="Gateway uptime in seconds")

# Utility Functions
def create_face_detection_response(
    faces: List[DetectedFace],
    image_info: ImageInfo,
    processing_stats: ProcessingStats,
    request_id: str = None
) -> FaceDetectionResponse:
    """Create standardized face detection response"""
    
    data = DetectionData(
        total_faces=len(faces),
        image_info=image_info,
        faces=faces,
        processing_stats=processing_stats
    )
    
    metadata = {
        "request_id": request_id or f"req_{int(datetime.now().timestamp())}",
        "processing_time": f"{processing_stats.total_time_ms:.0f}ms",
        "model_versions": {
            "detector": "unified_v1.0.0",
            "quality": "quality_v1.0.0",
            "attributes": "attributes_v1.0.0"
        },
        "timestamp": datetime.now().isoformat()
    }
    
    return FaceDetectionResponse(
        success=True,
        data=data.dict(),
        metadata=metadata
    )

def create_error_response(
    error_code: str,
    error_message: str,
    details: Dict[str, Any] = None,
    request_id: str = None
) -> ErrorResponse:
    """Create standardized error response"""
    
    error_detail = ErrorDetail(
        error_code=error_code,
        error_message=error_message,
        details=details or {}
    )
    
    metadata = {
        "request_id": request_id or f"req_{int(datetime.now().timestamp())}",
        "timestamp": datetime.now().isoformat(),
        "error_timestamp": datetime.now().isoformat()
    }
    
    return ErrorResponse(
        error=error_detail,
        metadata=metadata
    )