# Deepfake Detection Service - การตรวจจับภาพปลอมและสื่อสังเคราะห์

## 1. Service Overview

### หน้าที่หลัก
- **Image Deepfake Detection**: ตรวจจับภาพปลอมจากรูปภาพเดียว
- **Video Deepfake Analysis**: วิเคราะห์วิดีโอเพื่อหาการปลอมแปลง
- **Temporal Consistency Check**: ตรวจสอบความสอดคล้องเวลาในวิดีโอ
- **Manipulation Classification**: จำแนกประเภทการปลอมแปลง
- **Batch Processing**: ประมวลผลหลายไฟล์พร้อมกัน

### Integration Flow ใน Frontend
```
Media Upload → Deepfake Detection → Antispoofing Check → Face Recognition
Content Post → Background Analysis → Alert if Suspicious
Live Video → Real-time Detection → Block Suspicious Stream
Admin Review → Manual Verification → Content Moderation
```

### AI Models ที่ใช้
- **EfficientNet-B4**: การตรวจจับภาพปลอมพื้นฐาน
- **XceptionNet**: การจำแนกประเภทการปลอมแปลง
- **MesoNet**: เฉพาะ Deepfake ในวิดีโอ
- **BiT (Big Transfer)**: การวิเคราะห์ความสอดคล้อง

## 2. API Endpoints

### 2.1 Image Deepfake Detection

```http
POST /api/v1/deepfake/analyze-image
Content-Type: application/json
Authorization: Bearer {token}

{
  "image": "base64_encoded_image",
  "detection_level": "standard", // "basic", "standard", "advanced"
  "analysis_options": {
    "detailed_report": true,
    "manipulation_regions": true,
    "confidence_threshold": 0.7
  },
  "metadata": {
    "source": "user_upload",
    "content_type": "social_post",
    "request_priority": "normal"
  }
}
```

**Response (200 OK):**
```json
{
  "success": true,
  "data": {
    "is_deepfake": false,
    "confidence_score": 0.9234,
    "manipulation_score": 0.1456,
    "risk_level": "low", // "low", "medium", "high", "critical"
    "analysis_details": {
      "manipulation_types": [], // ["face_swap", "attribute_manipulation", "expression_transfer"]
      "inconsistency_areas": [],
      "artifacts_detected": [],
      "temporal_analysis": null,
      "quality_metrics": {
        "compression_artifacts": 0.12,
        "blur_score": 0.08,
        "noise_level": 0.05,
        "resolution_quality": 0.92
      }
    },
    "recommendations": {
      "action": "allow", // "allow", "review", "block"
      "reason": "No manipulation detected",
      "suggested_checks": []
    }
  },
  "metadata": {
    "request_id": "req_uuid",
    "processing_time": "1.2s",
    "model_versions": {
      "primary": "deepfake_v3.1.0",
      "secondary": "xception_v2.3.0"
    },
    "timestamp": "2024-01-20T10:30:00Z"
  }
}
```

### 2.2 Video Deepfake Analysis

```http
POST /api/v1/deepfake/analyze-video
Content-Type: application/json
Authorization: Bearer {token}

{
  "video_url": "https://storage.com/video.mp4",
  "detection_level": "advanced",
  "analysis_options": {
    "frame_sampling_rate": 5, // frames per second
    "temporal_analysis": true,
    "motion_analysis": true,
    "audio_visual_sync": true,
    "detailed_timeline": true
  },
  "callback_config": {
    "webhook_url": "https://app.com/webhook/deepfake",
    "notification_email": "admin@facesocial.com"
  },
  "metadata": {
    "content_id": "content_uuid",
    "uploader_id": "user_uuid",
    "priority": "high"
  }
}
```

**Response (202 Accepted):**
```json
{
  "success": true,
  "data": {
    "analysis_id": "analysis_uuid",
    "status": "processing",
    "estimated_completion": "2024-01-20T10:33:00Z",
    "progress": {
      "current_step": "frame_extraction",
      "completion_percentage": 15,
      "frames_processed": 45,
      "total_frames": 300
    },
    "webhook_configured": true
  },
  "metadata": {
    "request_id": "req_uuid",
    "processing_time": "234ms",
    "timestamp": "2024-01-20T10:30:00Z"
  }
}
```

### 2.3 Get Video Analysis Result

```http
GET /api/v1/deepfake/analysis/{analysis_id}
Authorization: Bearer {token}
```

**Response (200 OK):**
```json
{
  "success": true,
  "data": {
    "analysis_id": "analysis_uuid",
    "status": "completed", // "processing", "completed", "failed"
    "is_deepfake": true,
    "overall_confidence": 0.8765,
    "manipulation_score": 0.7892,
    "risk_level": "high",
    "analysis_summary": {
      "total_frames": 300,
      "suspicious_frames": 89,
      "manipulation_percentage": 29.67,
      "consistency_score": 0.34,
      "temporal_artifacts": 23
    },
    "frame_analysis": [
      {
        "frame_number": 45,
        "timestamp": "00:01:30",
        "is_deepfake": true,
        "confidence": 0.89,
        "manipulation_type": "face_swap",
        "bounding_boxes": [
          {
            "x": 120, "y": 80, "width": 150, "height": 180,
            "confidence": 0.92
          }
        ]
      }
    ],
    "timeline_analysis": {
      "suspicious_segments": [
        {
          "start_time": "00:01:25",
          "end_time": "00:02:15",
          "severity": "high",
          "manipulation_type": "face_swap"
        }
      ]
    },
    "recommendations": {
      "action": "block",
      "reason": "High confidence deepfake detected",
      "manual_review_required": true
    },
    "completed_at": "2024-01-20T10:32:45Z"
  },
  "metadata": {
    "request_id": "req_uuid",
    "processing_time": "18ms",
    "timestamp": "2024-01-20T10:35:00Z"
  }
}
```

### 2.4 Real-time Stream Analysis

```http
POST /api/v1/deepfake/analyze-stream
Content-Type: application/json
Authorization: Bearer {token}

{
  "stream_id": "stream_uuid",
  "stream_url": "rtmp://stream.com/live/stream_key",
  "analysis_config": {
    "real_time_alerts": true,
    "frame_skip_interval": 10,
    "alert_threshold": 0.7,
    "buffer_seconds": 30
  },
  "callback_config": {
    "alert_webhook": "https://app.com/webhook/stream-alert",
    "admin_notification": true
  }
}
```

### 2.5 Batch Processing

```http
POST /api/v1/deepfake/batch-analyze
Content-Type: application/json
Authorization: Bearer {token}

{
  "batch_id": "batch_uuid",
  "items": [
    {
      "item_id": "item_001",
      "type": "image",
      "url": "https://storage.com/image1.jpg"
    },
    {
      "item_id": "item_002", 
      "type": "video",
      "url": "https://storage.com/video1.mp4"
    }
  ],
  "processing_options": {
    "detection_level": "standard",
    "priority": "low",
    "parallel_processing": true
  }
}
```

## 3. Service Implementation

### 3.1 Project Structure

```
deepfake-detection-service/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application
│   ├── api/
│   │   ├── __init__.py
│   │   ├── v1/
│   │   │   ├── __init__.py
│   │   │   ├── routes/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── image.py        # Image analysis endpoints
│   │   │   │   ├── video.py        # Video analysis endpoints
│   │   │   │   ├── stream.py       # Stream analysis endpoints
│   │   │   │   ├── batch.py        # Batch processing
│   │   │   │   └── health.py       # Health check endpoints
│   │   │   └── dependencies.py
│   │   └── middleware/
│   │       ├── __init__.py
│   │       ├── auth.py
│   │       ├── rate_limit.py
│   │       └── logging.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── database.py
│   │   ├── security.py
│   │   └── exceptions.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── requests.py
│   │   ├── responses.py
│   │   └── database.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── deepfake/
│   │   │   ├── __init__.py
│   │   │   ├── image_detector.py   # Image deepfake detection
│   │   │   ├── video_analyzer.py   # Video analysis
│   │   │   ├── stream_processor.py # Real-time stream processing
│   │   │   ├── temporal_analyzer.py # Temporal consistency
│   │   │   └── artifact_detector.py # Artifact detection
│   │   ├── ml/
│   │   │   ├── __init__.py
│   │   │   ├── model_manager.py    # Model loading and management
│   │   │   ├── inference.py        # ML inference engine
│   │   │   ├── preprocessing.py    # Data preprocessing
│   │   │   └── postprocessing.py   # Result postprocessing
│   │   ├── video/
│   │   │   ├── __init__.py
│   │   │   ├── frame_extractor.py  # Video frame extraction
│   │   │   ├── codec_analyzer.py   # Video codec analysis
│   │   │   └── quality_assessor.py # Video quality assessment
│   │   ├── database/
│   │   │   ├── __init__.py
│   │   │   ├── postgres.py
│   │   │   └── redis.py
│   │   └── external/
│   │       ├── __init__.py
│   │       ├── storage.py          # Cloud storage client
│   │       └── notification.py     # Webhook notifications
│   └── utils/
│       ├── __init__.py
│       ├── image.py
│       ├── video.py
│       ├── validation.py
│       └── metrics.py
├── models/                     # AI model files
│   ├── deepfake/
│   │   ├── efficientnet_b4.onnx
│   │   ├── xceptionnet.onnx
│   │   ├── mesonet.onnx
│   │   └── bit_model.onnx
│   ├── preprocessing/
│   │   ├── face_detector.onnx
│   │   └── image_enhancer.onnx
│   └── config/
│       ├── model_config.yaml
│       └── thresholds.yaml
├── tests/
│   ├── __init__.py
│   ├── test_api/
│   ├── test_services/
│   └── test_utils/
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── requirements.txt
├── scripts/
│   ├── setup_models.py
│   ├── benchmark.py
│   └── migrate_db.py
├── config/
│   ├── settings.yaml
│   └── logging.yaml
└── README.md
```

### 3.2 Core Implementation

#### Image Deepfake Detector

```python
# app/services/deepfake/image_detector.py
import numpy as np
import cv2
import onnxruntime as ort
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
import asyncio

class ManipulationType(Enum):
    FACE_SWAP = "face_swap"
    ATTRIBUTE_MANIPULATION = "attribute_manipulation"
    EXPRESSION_TRANSFER = "expression_transfer"
    ENTIRE_FACE_SYNTHESIS = "entire_face_synthesis"
    PARTIAL_FACE_MANIPULATION = "partial_face_manipulation"

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ImageDeepfakeDetector:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Load models
        self.primary_model = ort.InferenceSession(config["primary_model_path"])
        self.xception_model = ort.InferenceSession(config["xception_model_path"])
        self.artifact_model = ort.InferenceSession(config["artifact_model_path"])
        
        # Detection thresholds
        self.deepfake_threshold = config.get("deepfake_threshold", 0.5)
        self.high_risk_threshold = config.get("high_risk_threshold", 0.8)
        self.critical_threshold = config.get("critical_threshold", 0.95)
        
    async def analyze_image(self, image: np.ndarray) -> Dict[str, Any]:
        """วิเคราะห์รูปภาพเพื่อตรวจจับ deepfake"""
        
        # 1. Preprocess image
        processed_image = self._preprocess_image(image)
        if processed_image is None:
            raise ValueError("Failed to preprocess image")
            
        # 2. Primary deepfake detection
        primary_result = await self._run_primary_detection(processed_image)
        
        # 3. Secondary analysis with XceptionNet
        xception_result = await self._run_xception_analysis(processed_image)
        
        # 4. Artifact detection
        artifact_result = await self._detect_artifacts(processed_image)
        
        # 5. Quality assessment
        quality_metrics = self._assess_image_quality(processed_image)
        
        # 6. Combine results
        final_result = self._combine_results(
            primary_result, xception_result, artifact_result, quality_metrics
        )
        
        return final_result
        
    async def _run_primary_detection(self, image: np.ndarray) -> Dict[str, Any]:
        """Run primary EfficientNet-B4 detection"""
        try:
            input_name = self.primary_model.get_inputs()[0].name
            output_name = self.primary_model.get_outputs()[0].name
            
            # Prepare input
            input_data = self._prepare_model_input(image, (224, 224))
            
            # Run inference
            result = self.primary_model.run([output_name], {input_name: input_data})
            
            # Get deepfake probability
            deepfake_prob = float(result[0][0][1])  # Assuming [real, fake] output
            
            return {
                "deepfake_probability": deepfake_prob,
                "confidence": abs(deepfake_prob - 0.5) * 2,  # 0-1 scale
                "model_certainty": "high" if abs(deepfake_prob - 0.5) > 0.3 else "low"
            }
            
        except Exception as e:
            raise RuntimeError(f"Primary model inference failed: {e}")
            
    async def _run_xception_analysis(self, image: np.ndarray) -> Dict[str, Any]:
        """Run XceptionNet for manipulation type classification"""
        try:
            input_name = self.xception_model.get_inputs()[0].name
            output_names = [o.name for o in self.xception_model.get_outputs()]
            
            # Prepare input
            input_data = self._prepare_model_input(image, (299, 299))
            
            # Run inference
            result = self.xception_model.run(output_names, {input_name: input_data})
            
            # Parse outputs
            manipulation_probs = result[0][0]  # Manipulation type probabilities
            manipulation_score = float(result[1][0])  # Overall manipulation score
            
            # Get top manipulation types
            manipulation_types = self._get_top_manipulations(manipulation_probs)
            
            return {
                "manipulation_score": manipulation_score,
                "manipulation_types": manipulation_types,
                "type_confidences": {
                    ManipulationType.FACE_SWAP.value: float(manipulation_probs[0]),
                    ManipulationType.ATTRIBUTE_MANIPULATION.value: float(manipulation_probs[1]),
                    ManipulationType.EXPRESSION_TRANSFER.value: float(manipulation_probs[2]),
                    ManipulationType.ENTIRE_FACE_SYNTHESIS.value: float(manipulation_probs[3])
                }
            }
            
        except Exception as e:
            return {
                "manipulation_score": 0.5,
                "manipulation_types": [],
                "type_confidences": {},
                "error": str(e)
            }
            
    async def _detect_artifacts(self, image: np.ndarray) -> Dict[str, Any]:
        """Detect compression and generation artifacts"""
        try:
            # Convert to different color spaces for analysis
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            
            # Analyze frequency domain
            frequency_artifacts = self._analyze_frequency_domain(rgb_image)
            
            # Analyze compression artifacts
            compression_artifacts = self._analyze_compression_artifacts(rgb_image)
            
            # Analyze inconsistencies
            inconsistencies = self._detect_inconsistencies(rgb_image, hsv_image, lab_image)
            
            return {
                "frequency_artifacts": frequency_artifacts,
                "compression_artifacts": compression_artifacts,
                "inconsistencies": inconsistencies,
                "overall_artifact_score": (
                    frequency_artifacts["score"] * 0.4 +
                    compression_artifacts["score"] * 0.3 +
                    inconsistencies["score"] * 0.3
                )
            }
            
        except Exception as e:
            return {
                "frequency_artifacts": {"score": 0.5},
                "compression_artifacts": {"score": 0.5},
                "inconsistencies": {"score": 0.5},
                "overall_artifact_score": 0.5,
                "error": str(e)
            }
            
    def _analyze_frequency_domain(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze frequency domain for generation artifacts"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # DCT analysis
        dct = cv2.dct(np.float32(gray))
        
        # High frequency component analysis
        high_freq_energy = np.sum(np.abs(dct[32:, 32:]))
        total_energy = np.sum(np.abs(dct))
        high_freq_ratio = high_freq_energy / total_energy if total_energy > 0 else 0
        
        # FFT analysis for periodic patterns
        fft = np.fft.fft2(gray)
        fft_magnitude = np.abs(fft)
        
        # Detect periodic patterns (common in GAN artifacts)
        periodic_score = self._detect_periodic_patterns(fft_magnitude)
        
        # Overall frequency artifact score
        artifact_score = (high_freq_ratio * 0.6 + periodic_score * 0.4)
        
        return {
            "score": min(artifact_score, 1.0),
            "high_freq_ratio": high_freq_ratio,
            "periodic_score": periodic_score,
            "suspicious_patterns": artifact_score > 0.7
        }
        
    def _analyze_compression_artifacts(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze JPEG compression artifacts"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Block artifact detection (8x8 JPEG blocks)
        block_artifacts = self._detect_block_artifacts(gray)
        
        # Ringing artifacts around edges
        ringing_artifacts = self._detect_ringing_artifacts(gray)
        
        # Quantization artifacts
        quantization_artifacts = self._detect_quantization_artifacts(gray)
        
        # Overall compression artifact score
        compression_score = (
            block_artifacts * 0.4 +
            ringing_artifacts * 0.3 +
            quantization_artifacts * 0.3
        )
        
        return {
            "score": compression_score,
            "block_artifacts": block_artifacts,
            "ringing_artifacts": ringing_artifacts,
            "quantization_artifacts": quantization_artifacts,
            "quality_estimate": 1.0 - compression_score
        }
        
    def _detect_inconsistencies(
        self, 
        rgb_image: np.ndarray, 
        hsv_image: np.ndarray, 
        lab_image: np.ndarray
    ) -> Dict[str, Any]:
        """Detect lighting and color inconsistencies"""
        
        # Lighting consistency analysis
        lighting_score = self._analyze_lighting_consistency(lab_image)
        
        # Color histogram analysis
        color_score = self._analyze_color_consistency(rgb_image, hsv_image)
        
        # Shadow consistency
        shadow_score = self._analyze_shadow_consistency(rgb_image)
        
        # Edge consistency
        edge_score = self._analyze_edge_consistency(rgb_image)
        
        overall_score = (
            lighting_score * 0.3 +
            color_score * 0.3 +
            shadow_score * 0.2 +
            edge_score * 0.2
        )
        
        return {
            "score": overall_score,
            "lighting_consistency": lighting_score,
            "color_consistency": color_score,
            "shadow_consistency": shadow_score,
            "edge_consistency": edge_score,
            "suspicious_regions": overall_score > 0.6
        }
        
    def _combine_results(
        self,
        primary_result: Dict[str, Any],
        xception_result: Dict[str, Any],
        artifact_result: Dict[str, Any],
        quality_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Combine all analysis results"""
        
        # Calculate weighted final score
        weights = {
            "primary": 0.5,
            "xception": 0.3,
            "artifacts": 0.2
        }
        
        final_score = (
            primary_result["deepfake_probability"] * weights["primary"] +
            xception_result["manipulation_score"] * weights["xception"] +
            artifact_result["overall_artifact_score"] * weights["artifacts"]
        )
        
        # Determine if deepfake
        is_deepfake = final_score >= self.deepfake_threshold
        
        # Determine risk level
        risk_level = self._determine_risk_level(final_score)
        
        # Get manipulation types
        manipulation_types = xception_result.get("manipulation_types", [])
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            is_deepfake, final_score, risk_level, manipulation_types
        )
        
        return {
            "is_deepfake": is_deepfake,
            "confidence_score": final_score,
            "manipulation_score": xception_result["manipulation_score"],
            "risk_level": risk_level.value,
            "analysis_details": {
                "manipulation_types": manipulation_types,
                "inconsistency_areas": [],
                "artifacts_detected": artifact_result.get("suspicious_patterns", []),
                "quality_metrics": quality_metrics,
                "model_scores": {
                    "primary_model": primary_result["deepfake_probability"],
                    "xception_model": xception_result["manipulation_score"],
                    "artifact_detection": artifact_result["overall_artifact_score"]
                }
            },
            "recommendations": recommendations
        }
        
    def _determine_risk_level(self, score: float) -> RiskLevel:
        """Determine risk level based on confidence score"""
        if score >= self.critical_threshold:
            return RiskLevel.CRITICAL
        elif score >= self.high_risk_threshold:
            return RiskLevel.HIGH
        elif score >= self.deepfake_threshold:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
            
    def _generate_recommendations(
        self,
        is_deepfake: bool,
        confidence: float,
        risk_level: RiskLevel,
        manipulation_types: List[str]
    ) -> Dict[str, Any]:
        """Generate action recommendations"""
        
        if risk_level == RiskLevel.CRITICAL:
            action = "block"
            reason = "Critical deepfake detected with high confidence"
        elif risk_level == RiskLevel.HIGH:
            action = "review" if confidence < 0.9 else "block"
            reason = "High confidence deepfake detected"
        elif risk_level == RiskLevel.MEDIUM:
            action = "review"
            reason = "Possible deepfake detected"
        else:
            action = "allow"
            reason = "No significant manipulation detected"
            
        return {
            "action": action,
            "reason": reason,
            "manual_review_required": risk_level.value in ["high", "critical"],
            "suggested_checks": self._get_suggested_checks(manipulation_types)
        }
        
    # Helper methods for preprocessing and utilities
    def _preprocess_image(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Preprocess image for analysis"""
        try:
            # Resize if too large
            height, width = image.shape[:2]
            if max(height, width) > 1024:
                scale = 1024 / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height))
                
            return image
        except Exception:
            return None
            
    def _prepare_model_input(self, image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Prepare image input for model inference"""
        # Resize to target size
        resized = cv2.resize(image, target_size)
        
        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0
        
        # Add batch dimension
        input_data = np.expand_dims(normalized, axis=0)
        
        return input_data
```

#### Video Deepfake Analyzer

```python
# app/services/deepfake/video_analyzer.py
import asyncio
import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import tempfile
import os

@dataclass
class FrameAnalysis:
    frame_number: int
    timestamp: str
    is_deepfake: bool
    confidence: float
    manipulation_type: Optional[str]
    bounding_boxes: List[Dict[str, Any]]

class VideoDeepfakeAnalyzer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.image_detector = ImageDeepfakeDetector(config)
        self.temporal_analyzer = TemporalAnalyzer(config)
        
        # Video processing settings
        self.frame_sampling_rate = config.get("frame_sampling_rate", 5)
        self.max_frames = config.get("max_frames", 1000)
        self.batch_size = config.get("batch_size", 16)
        
    async def analyze_video(self, video_path: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze video for deepfake content"""
        
        # 1. Extract video metadata
        metadata = self._extract_video_metadata(video_path)
        
        # 2. Extract frames
        frames = await self._extract_frames(video_path, options)
        
        # 3. Analyze frames in batches
        frame_results = await self._analyze_frames_batch(frames)
        
        # 4. Temporal consistency analysis
        temporal_result = await self.temporal_analyzer.analyze_temporal_consistency(
            frame_results, metadata
        )
        
        # 5. Generate final result
        final_result = self._generate_video_result(
            frame_results, temporal_result, metadata
        )
        
        return final_result
        
    async def _extract_frames(self, video_path: str, options: Dict[str, Any]) -> List[np.ndarray]:
        """Extract frames from video for analysis"""
        frames = []
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Unable to open video file")
            
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = max(1, int(fps / self.frame_sampling_rate))
            frame_count = 0
            extracted_count = 0
            
            while extracted_count < self.max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_count % frame_interval == 0:
                    frames.append(frame)
                    extracted_count += 1
                    
                frame_count += 1
                
        finally:
            cap.release()
            
        return frames
        
    async def _analyze_frames_batch(self, frames: List[np.ndarray]) -> List[FrameAnalysis]:
        """Analyze frames in batches for efficiency"""
        results = []
        
        for i in range(0, len(frames), self.batch_size):
            batch = frames[i:i + self.batch_size]
            batch_results = await self._process_frame_batch(batch, i)
            results.extend(batch_results)
            
        return results
        
    async def _process_frame_batch(self, batch: List[np.ndarray], start_idx: int) -> List[FrameAnalysis]:
        """Process a batch of frames"""
        tasks = []
        
        for idx, frame in enumerate(batch):
            task = self._analyze_single_frame(frame, start_idx + idx)
            tasks.append(task)
            
        results = await asyncio.gather(*tasks)
        return results
        
    async def _analyze_single_frame(self, frame: np.ndarray, frame_number: int) -> FrameAnalysis:
        """Analyze a single frame"""
        try:
            # Analyze frame with image detector
            result = await self.image_detector.analyze_image(frame)
            
            # Extract relevant information
            is_deepfake = result["is_deepfake"]
            confidence = result["confidence_score"]
            manipulation_types = result["analysis_details"]["manipulation_types"]
            manipulation_type = manipulation_types[0] if manipulation_types else None
            
            # Convert frame number to timestamp (assuming 30fps)
            timestamp = self._frame_to_timestamp(frame_number, 30)
            
            return FrameAnalysis(
                frame_number=frame_number,
                timestamp=timestamp,
                is_deepfake=is_deepfake,
                confidence=confidence,
                manipulation_type=manipulation_type,
                bounding_boxes=[]  # TODO: Implement face detection integration
            )
            
        except Exception as e:
            # Return safe default on error
            return FrameAnalysis(
                frame_number=frame_number,
                timestamp=self._frame_to_timestamp(frame_number, 30),
                is_deepfake=False,
                confidence=0.0,
                manipulation_type=None,
                bounding_boxes=[]
            )
```

### 3.3 Temporal Consistency Analyzer

```python
# app/services/deepfake/temporal_analyzer.py
import numpy as np
from typing import Dict, Any, List
from scipy import signal
from sklearn.cluster import DBSCAN

class TemporalAnalyzer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.consistency_threshold = config.get("consistency_threshold", 0.7)
        
    async def analyze_temporal_consistency(
        self, 
        frame_results: List[FrameAnalysis], 
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze temporal consistency across video frames"""
        
        if len(frame_results) < 10:
            return {
                "consistency_score": 1.0,
                "temporal_artifacts": 0,
                "suspicious_segments": [],
                "analysis_quality": "insufficient_frames"
            }
            
        # 1. Confidence signal analysis
        confidence_analysis = self._analyze_confidence_signal(frame_results)
        
        # 2. Detection pattern analysis
        pattern_analysis = self._analyze_detection_patterns(frame_results)
        
        # 3. Transition analysis
        transition_analysis = self._analyze_transitions(frame_results)
        
        # 4. Clustering suspicious segments
        segments = self._identify_suspicious_segments(frame_results)
        
        # 5. Calculate overall consistency score
        consistency_score = self._calculate_consistency_score(
            confidence_analysis, pattern_analysis, transition_analysis
        )
        
        return {
            "consistency_score": consistency_score,
            "temporal_artifacts": len(segments),
            "suspicious_segments": segments,
            "confidence_analysis": confidence_analysis,
            "pattern_analysis": pattern_analysis,
            "transition_analysis": transition_analysis
        }
        
    def _analyze_confidence_signal(self, results: List[FrameAnalysis]) -> Dict[str, Any]:
        """Analyze the confidence signal over time"""
        confidences = [r.confidence for r in results]
        
        # Signal smoothness
        smoothness = self._calculate_signal_smoothness(confidences)
        
        # Sudden spikes/drops
        spikes = self._detect_sudden_changes(confidences)
        
        # Frequency analysis
        frequencies = self._analyze_frequency_components(confidences)
        
        return {
            "smoothness": smoothness,
            "sudden_changes": len(spikes),
            "change_points": spikes,
            "frequency_analysis": frequencies
        }
        
    def _calculate_signal_smoothness(self, signal: List[float]) -> float:
        """Calculate smoothness of confidence signal"""
        if len(signal) < 3:
            return 1.0
            
        # Calculate first derivative
        derivatives = np.diff(signal)
        
        # Calculate variance of derivatives (lower = smoother)
        variance = np.var(derivatives)
        
        # Convert to smoothness score (0-1, higher = smoother)
        smoothness = 1.0 / (1.0 + variance * 10)
        
        return smoothness
```

## 4. Database Integration

### 4.1 Analysis Results Storage

```sql
-- Deepfake analysis results
CREATE TABLE deepfake_detection.analysis_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content_id UUID NOT NULL,
    content_type VARCHAR(20) NOT NULL, -- 'image', 'video'
    is_deepfake BOOLEAN NOT NULL,
    confidence_score FLOAT NOT NULL,
    manipulation_score FLOAT NOT NULL,
    risk_level VARCHAR(20) NOT NULL,
    manipulation_types JSONB,
    analysis_details JSONB,
    model_versions JSONB,
    processing_time_ms INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_content_id (content_id),
    INDEX idx_risk_level (risk_level),
    INDEX idx_created_at (created_at)
);

-- Video frame analysis
CREATE TABLE deepfake_detection.frame_analysis (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    analysis_id UUID REFERENCES deepfake_detection.analysis_results(id),
    frame_number INTEGER NOT NULL,
    timestamp_ms INTEGER NOT NULL,
    is_deepfake BOOLEAN NOT NULL,
    confidence FLOAT NOT NULL,
    manipulation_type VARCHAR(50),
    bounding_boxes JSONB,
    
    INDEX idx_analysis_id (analysis_id),
    INDEX idx_frame_number (frame_number)
);
```

## 5. Performance Optimization

### 5.1 Model Optimization
- **ONNX Runtime**: Optimized inference engine
- **TensorRT**: GPU acceleration for NVIDIA cards
- **Model Quantization**: INT8 quantization for faster inference
- **Batch Processing**: Process multiple frames simultaneously

### 5.2 Caching Strategy
- **Redis**: Cache frequent analysis results
- **Model Caching**: Keep models in memory
- **Frame Caching**: Cache extracted frames for re-analysis

### 5.3 Async Processing
- **Celery Workers**: Background video processing
- **WebSocket**: Real-time progress updates
- **Queue Management**: Priority-based task scheduling

## 6. Monitoring และ Analytics

### 6.1 Performance Metrics
- **Detection Accuracy**: False positive/negative rates
- **Processing Time**: Average analysis time per media type
- **Throughput**: Files processed per hour
- **Model Performance**: Individual model accuracy

### 6.2 Security Monitoring
- **Suspicious Patterns**: Unusual detection patterns
- **Evasion Attempts**: Potential bypass attempts
- **Content Trends**: Emerging manipulation techniques

---

Deepfake Detection Service เป็นบริการสำคัญที่ช่วยปกป้องระบบจากสื่อปลอมและการปลอมแปลง โดยใช้เทคโนโลยี AI ที่ทันสมัยและการวิเคราะห์หลายมิติเพื่อให้ผลลัพธ์ที่แม่นยำและเชื่อถือได้
