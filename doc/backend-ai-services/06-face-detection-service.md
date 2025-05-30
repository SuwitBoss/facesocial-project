# Face Detection Service - การตรวจจับและวิเคราะห์ใบหน้า

## 1. Service Overview

### หน้าที่หลัก
- **Multi-Face Detection**: ตรวจจับหลายใบหน้าในภาพเดียว
- **Facial Landmarks**: หาจุดสำคัญบนใบหน้า (5-point, 68-point)
- **Face Quality Assessment**: ประเมินคุณภาพใบหน้า
- **Face Alignment**: ปรับแนวใบหน้าให้ได้มาตรฐาน
- **Face Attributes**: วิเคราะห์คุณลักษณะพื้นฐาน (age, gender, emotion)

### Integration Flow ใน Frontend
```
Create Post → Auto Face Detection → Face Recognition → Face Tagging
Profile Setup → Face Detection → Quality Check → Face Registration
Video Call → Real-time Detection → Live Face Tracking
Content Analysis → Batch Detection → Demographics Analysis
```

### AI Models ที่ใช้
- **SCRFD**: การตรวจจับใบหน้าหลักพร้อม landmarks
- **RetinaFace**: การตรวจจับใบหน้าความแม่นยำสูง
- **MTCNN**: การตรวจจับและ alignment แบบ multi-stage
- **3DDFA**: การวิเคราะห์ 3D face structure

---

## 2. API Endpoints

### 2.1 Multi-Face Detection

```http
POST /api/v1/face-detection/detect
Content-Type: application/json
Authorization: Bearer {token}

{
  "image": "base64_encoded_image",
  "detection_options": {
    "min_face_size": 40, // pixels
    "max_faces": 100,
    "detection_confidence": 0.7,
    "return_landmarks": true,
    "return_attributes": false,
    "quality_assessment": true
  },
  "processing_options": {
    "face_alignment": true,
    "crop_faces": true,
    "enhance_quality": false
  },
  "metadata": {
    "source": "post_upload",
    "content_id": "content_uuid",
    "request_priority": "normal"
  }
}
```

**Response (200 OK):**
```json
{
  "success": true,
  "data": {
    "total_faces": 3,
    "image_info": {
      "width": 1920,
      "height": 1080,
      "format": "JPEG"
    },
    "faces": [
      {
        "face_id": "face_001",
        "face_index": 0,
        "bounding_box": {
          "x": 120,
          "y": 80,
          "width": 150,
          "height": 180,
          "confidence": 0.9876
        },
        "landmarks": {
          "type": "5_point", // "5_point", "68_point", "106_point"
          "points": [
            {"name": "left_eye", "x": 145.2, "y": 125.8},
            {"name": "right_eye", "x": 195.6, "y": 123.4},
            {"name": "nose", "x": 168.9, "y": 145.7},
            {"name": "left_mouth", "x": 152.3, "y": 175.2},
            {"name": "right_mouth", "x": 186.1, "y": 173.8}
          ]
        },
        "quality_assessment": {
          "overall_score": 0.8965,
          "sharpness": 0.92,
          "brightness": 0.78,
          "contrast": 0.85,
          "face_angle": {
            "yaw": 5.2,
            "pitch": -2.1,
            "roll": 1.3
          },
          "occlusion": {
            "left_eye": 0.0,
            "right_eye": 0.0,
            "nose": 0.0,
            "mouth": 0.0
          }
        },
        "face_attributes": {
          "estimated_age": 28.5,
          "estimated_gender": "female",
          "gender_confidence": 0.94,
          "dominant_emotion": "happy",
          "emotion_scores": {
            "happy": 0.78,
            "neutral": 0.15,
            "sad": 0.04,
            "angry": 0.02,
            "surprised": 0.01
          }
        },
        "aligned_face": "base64_aligned_face_image", // if requested
        "cropped_face": "base64_cropped_face_image"   // if requested
      }
    ],
    "processing_stats": {
      "detection_time_ms": 245,
      "landmarks_time_ms": 89,
      "quality_assessment_time_ms": 56,
      "total_time_ms": 390
    }
  },
  "metadata": {
    "request_id": "req_uuid",
    "processing_time": "390ms",
    "model_versions": {
      "detector": "scrfd_v2.1.0",
      "landmarks": "3ddfa_v1.3.0"
    },
    "timestamp": "2024-01-20T10:30:00Z"
  }
}
```

### 2.2 Real-time Face Tracking

```http
POST /api/v1/face-detection/track-stream
Content-Type: application/json
Authorization: Bearer {token}

{
  "stream_id": "stream_uuid",
  "tracking_config": {
    "frame_rate": 15, // frames per second to process
    "detection_interval": 5, // re-detect every N frames
    "tracking_confidence": 0.8,
    "face_id_consistency": true
  },
  "detection_options": {
    "min_face_size": 80,
    "max_faces": 10,
    "return_landmarks": true,
    "quality_threshold": 0.6
  },
  "callback_config": {
    "webhook_url": "https://app.com/webhook/face-tracking",
    "real_time_updates": true
  }
}
```

**Response (202 Accepted):**
```json
{
  "success": true,
  "data": {
    "tracking_session_id": "track_uuid",
    "stream_id": "stream_uuid",
    "status": "active",
    "config": {
      "frame_rate": 15,
      "detection_interval": 5,
      "tracking_confidence": 0.8
    },
    "webhook_configured": true
  },
  "metadata": {
    "request_id": "req_uuid",
    "processing_time": "12ms",
    "timestamp": "2024-01-20T10:30:00Z"
  }
}
```

### 2.3 Batch Face Detection

```http
POST /api/v1/face-detection/batch-detect
Content-Type: application/json
Authorization: Bearer {token}

{
  "batch_id": "batch_uuid",
  "images": [
    {
      "image_id": "img_001",
      "image": "base64_encoded_image_1",
      "metadata": {"source": "album_1"}
    },
    {
      "image_id": "img_002", 
      "image": "base64_encoded_image_2",
      "metadata": {"source": "album_2"}
    }
  ],
  "detection_options": {
    "min_face_size": 40,
    "detection_confidence": 0.7,
    "return_landmarks": true,
    "return_attributes": true
  },
  "processing_options": {
    "parallel_processing": true,
    "priority": "normal" // "low", "normal", "high"
  }
}
```

**Response (202 Accepted):**
```json
{
  "success": true,
  "data": {
    "batch_id": "batch_uuid",
    "status": "processing",
    "total_images": 2,
    "estimated_completion": "2024-01-20T10:32:00Z",
    "progress": {
      "completed": 0,
      "failed": 0,
      "pending": 2
    }
  },
  "metadata": {
    "request_id": "req_uuid",
    "processing_time": "23ms",
    "timestamp": "2024-01-20T10:30:00Z"
  }
}
```

### 2.4 Get Detection Results

```http
GET /api/v1/face-detection/results/{request_id}
Authorization: Bearer {token}
```

**Response (200 OK):**
```json
{
  "success": true,
  "data": {
    "request_id": "req_uuid",
    "status": "completed", // "processing", "completed", "failed"
    "request_type": "single", // "single", "batch", "stream"
    "results": {
      // Same structure as detection response
    },
    "completion_time": "2024-01-20T10:30:45Z"
  },
  "metadata": {
    "request_id": "req_uuid",
    "processing_time": "8ms",
    "timestamp": "2024-01-20T10:35:00Z"
  }
}
```

---

## 3. Service Implementation

### 3.1 Project Structure

```
face-detection-service/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application
│   ├── api/
│   │   ├── __init__.py
│   │   ├── v1/
│   │   │   ├── __init__.py
│   │   │   ├── routes/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── detection.py    # Face detection endpoints
│   │   │   │   ├── tracking.py     # Real-time tracking
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
│   │   ├── detection.py
│   │   ├── tracking.py
│   │   └── batch.py
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── requests.py
│   │   ├── responses.py
│   │   └── models.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── face_detection/
│   │   │   ├── __init__.py
│   │   │   ├── detector.py          # Main face detector
│   │   │   ├── landmarks.py         # Landmark detection
│   │   │   ├── quality_assessor.py  # Face quality assessment
│   │   │   ├── aligner.py           # Face alignment
│   │   │   └── tracker.py           # Face tracking for videos
│   │   ├── database/
│   │   │   ├── __init__.py
│   │   │   ├── postgres.py
│   │   │   └── redis.py
│   │   └── external/
│   │       ├── __init__.py
│   │       ├── storage.py           # Cloud storage client
│   │       └── notification.py      # Webhook notifications
│   └── utils/
│       ├── __init__.py
│       ├── image.py
│       ├── validation.py
│       └── metrics.py
├── models/                          # AI model files
│   ├── detection/
│   │   ├── scrfd_10g.onnx
│   │   ├── retinaface_r50.onnx
│   │   └── mtcnn.onnx
│   ├── landmarks/
│   │   ├── 3ddfa_mb1.onnx
│   │   └── face_alignment.onnx
│   ├── quality/
│   │   └── face_quality.onnx
│   └── config/
│       ├── model_config.yaml
│       └── detection_config.yaml
├── tests/
│   ├── __init__.py
│   ├── test_api/
│   ├── test_services/
│   └── test_utils/
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── requirements.txt
├── deployment/
│   ├── k8s/
│   │   ├── deployment.yaml
│   │   ├── service.yaml
│   │   └── ingress.yaml
│   └── scripts/
│       ├── deploy.sh
│       └── migrate.sh
└── README.md
```

### 3.2 Core Implementation

#### Face Detection Main Service

```python
# app/services/face_detection/detector.py
import numpy as np
import cv2
import onnxruntime as ort
from typing import List, Dict, Any, Optional, Tuple
import asyncio
from dataclasses import dataclass

@dataclass
class DetectedFace:
    face_id: str
    face_index: int
    bbox: Dict[str, float]  # x, y, width, height, confidence
    landmarks: Optional[Dict[str, Any]]
    quality_score: float
    attributes: Optional[Dict[str, Any]]
    aligned_face: Optional[np.ndarray]
    cropped_face: Optional[np.ndarray]

class FaceDetectionService:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Load detection models
        self.scrfd_model = ort.InferenceSession(config["scrfd_model_path"])
        self.retinaface_model = ort.InferenceSession(config["retinaface_model_path"])
        
        # Load auxiliary models
        self.landmarks_model = ort.InferenceSession(config["landmarks_model_path"])
        self.quality_model = ort.InferenceSession(config["quality_model_path"])
        
        # Detection parameters
        self.min_face_size = config.get("min_face_size", 40)
        self.detection_threshold = config.get("detection_threshold", 0.7)
        self.nms_threshold = config.get("nms_threshold", 0.4)
        
    async def detect_faces(
        self, 
        image: np.ndarray,
        options: Dict[str, Any]
    ) -> List[DetectedFace]:
        """หลักการตรวจจับใบหน้าหลายใบหน้า"""
        
        # 1. Preprocess image
        processed_image = self._preprocess_image(image)
        
        # 2. Primary detection with SCRFD
        primary_detections = await self._detect_with_scrfd(processed_image, options)
        
        # 3. Secondary validation with RetinaFace (for high confidence)
        if options.get("dual_validation", False):
            validated_detections = await self._validate_with_retinaface(
                processed_image, primary_detections
            )
        else:
            validated_detections = primary_detections
            
        # 4. Post-process detections
        faces = []
        for i, detection in enumerate(validated_detections):
            face = await self._process_detected_face(
                image, detection, i, options
            )
            if face:
                faces.append(face)
                
        # 5. Sort by confidence or face size
        faces.sort(key=lambda x: x.bbox["confidence"], reverse=True)
        
        return faces[:options.get("max_faces", 100)]
        
    async def _detect_with_scrfd(
        self, 
        image: np.ndarray, 
        options: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """SCRFD model detection"""
        try:
            # Prepare input
            input_data = self._prepare_scrfd_input(image)
            
            # Run inference
            input_name = self.scrfd_model.get_inputs()[0].name
            output_names = [o.name for o in self.scrfd_model.get_outputs()]
            
            results = self.scrfd_model.run(output_names, {input_name: input_data})
            
            # Parse SCRFD outputs
            detections = self._parse_scrfd_output(results, image.shape)
            
            # Filter by confidence and size
            filtered_detections = []
            for detection in detections:
                if (detection["confidence"] >= options.get("detection_confidence", 0.7) and
                    detection["width"] >= options.get("min_face_size", 40)):
                    filtered_detections.append(detection)
                    
            # Apply NMS
            final_detections = self._apply_nms(filtered_detections)
            
            return final_detections
            
        except Exception as e:
            raise RuntimeError(f"SCRFD detection failed: {str(e)}")
            
    async def _process_detected_face(
        self,
        image: np.ndarray,
        detection: Dict[str, Any],
        face_index: int,
        options: Dict[str, Any]
    ) -> Optional[DetectedFace]:
        """ประมวลผลใบหน้าที่ตรวจพบ"""
        try:
            face_id = f"face_{face_index:03d}"
            
            # Extract face region
            x, y, w, h = detection["bbox"]
            face_crop = image[int(y):int(y+h), int(x):int(x+w)]
            
            # Get landmarks if requested
            landmarks = None
            if options.get("return_landmarks", False):
                landmarks = await self._detect_landmarks(face_crop)
                
            # Quality assessment
            quality_score = await self._assess_face_quality(face_crop)
            
            # Face alignment if requested
            aligned_face = None
            if options.get("face_alignment", False) and landmarks:
                aligned_face = await self._align_face(face_crop, landmarks)
                
            # Basic attributes if requested
            attributes = None
            if options.get("return_attributes", False):
                attributes = await self._extract_basic_attributes(face_crop)
                
            return DetectedFace(
                face_id=face_id,
                face_index=face_index,
                bbox={
                    "x": x,
                    "y": y, 
                    "width": w,
                    "height": h,
                    "confidence": detection["confidence"]
                },
                landmarks=landmarks,
                quality_score=quality_score,
                attributes=attributes,
                aligned_face=aligned_face,
                cropped_face=face_crop if options.get("crop_faces", False) else None
            )
            
        except Exception as e:
            # Log error but continue with other faces
            print(f"Error processing face {face_index}: {str(e)}")
            return None
            
    async def _detect_landmarks(self, face_image: np.ndarray) -> Dict[str, Any]:
        """ตรวจจับ facial landmarks"""
        try:
            # Prepare input for 3DDFA model
            input_data = self._prepare_landmarks_input(face_image)
            
            input_name = self.landmarks_model.get_inputs()[0].name
            output_name = self.landmarks_model.get_outputs()[0].name
            
            result = self.landmarks_model.run([output_name], {input_name: input_data})
            landmarks_raw = result[0][0]
            
            # Parse landmarks (assuming 68-point model)
            landmarks = self._parse_landmarks(landmarks_raw, face_image.shape)
            
            # Convert to standardized format
            return {
                "type": "68_point",
                "points": landmarks,
                "confidence": self._calculate_landmarks_confidence(landmarks)
            }
            
        except Exception as e:
            return {
                "type": "failed",
                "points": [],
                "confidence": 0.0,
                "error": str(e)
            }
            
    async def _assess_face_quality(self, face_image: np.ndarray) -> float:
        """ประเมินคุณภาพใบหน้า"""
        try:
            # Prepare input
            input_data = self._prepare_quality_input(face_image)
            
            input_name = self.quality_model.get_inputs()[0].name
            output_name = self.quality_model.get_outputs()[0].name
            
            result = self.quality_model.run([output_name], {input_name: input_data})
            quality_score = float(result[0][0])
            
            # Additional quality checks
            sharpness = self._calculate_sharpness(face_image)
            brightness = self._calculate_brightness(face_image)
            contrast = self._calculate_contrast(face_image)
            
            # Combined quality score
            combined_score = (
                quality_score * 0.5 +
                sharpness * 0.2 + 
                brightness * 0.15 +
                contrast * 0.15
            )
            
            return min(max(combined_score, 0.0), 1.0)
            
        except Exception:
            # Fallback to basic quality assessment
            return self._basic_quality_assessment(face_image)
            
    # Helper methods for image processing
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for detection"""
        # Resize if too large
        height, width = image.shape[:2]
        if max(height, width) > 1280:
            scale = 1280 / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))
            
        return image
        
    def _prepare_scrfd_input(self, image: np.ndarray) -> np.ndarray:
        """Prepare input for SCRFD model"""
        # Resize to model input size (e.g., 640x640)
        resized = cv2.resize(image, (640, 640))
        
        # Normalize
        normalized = resized.astype(np.float32) / 255.0
        
        # RGB format and add batch dimension
        rgb_image = cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB)
        input_data = np.transpose(rgb_image, (2, 0, 1))
        input_data = np.expand_dims(input_data, axis=0)
        
        return input_data
        
    def _parse_scrfd_output(
        self, 
        outputs: List[np.ndarray], 
        image_shape: Tuple[int, int, int]
    ) -> List[Dict[str, Any]]:
        """Parse SCRFD model output"""
        detections = []
        
        # SCRFD typically outputs: bbox, scores, landmarks
        bboxes = outputs[0]  # Shape: [N, 4]
        scores = outputs[1]  # Shape: [N, 1] 
        landmarks = outputs[2] if len(outputs) > 2 else None  # Shape: [N, 10]
        
        height, width = image_shape[:2]
        
        for i in range(len(bboxes)):
            if scores[i] < self.detection_threshold:
                continue
                
            # Scale bbox to original image size
            x1, y1, x2, y2 = bboxes[i]
            x1 = max(0, x1 * width / 640)
            y1 = max(0, y1 * height / 640)
            x2 = min(width, x2 * width / 640)
            y2 = min(height, y2 * height / 640)
            
            detection = {
                "bbox": [x1, y1, x2 - x1, y2 - y1],  # x, y, width, height
                "confidence": float(scores[i]),
                "landmarks": landmarks[i] if landmarks is not None else None
            }
            
            detections.append(detection)
            
        return detections
        
    def _apply_nms(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply Non-Maximum Suppression"""
        if not detections:
            return []
            
        # Convert to format for cv2.dnn.NMSBoxes
        boxes = []
        confidences = []
        
        for det in detections:
            x, y, w, h = det["bbox"]
            boxes.append([x, y, w, h])
            confidences.append(det["confidence"])
            
        # Apply NMS
        indices = cv2.dnn.NMSBoxes(
            boxes, 
            confidences, 
            self.detection_threshold, 
            self.nms_threshold
        )
        
        if len(indices) > 0:
            indices = indices.flatten()
            return [detections[i] for i in indices]
        else:
            return []
```

#### Face Quality Assessment

```python
# app/services/face_detection/quality_assessor.py
import numpy as np
import cv2
from typing import Dict, Any

class FaceQualityAssessment:
    def __init__(self):
        self.min_face_size = 80
        self.optimal_face_size = 200
        
    def assess_comprehensive_quality(self, face_image: np.ndarray) -> Dict[str, Any]:
        """ประเมินคุณภาพใบหน้าแบบครอบคลุม"""
        
        # Basic quality metrics
        sharpness = self._calculate_sharpness(face_image)
        brightness = self._calculate_brightness(face_image)
        contrast = self._calculate_contrast(face_image)
        
        # Face-specific metrics
        face_size_score = self._assess_face_size(face_image)
        angle_score = self._assess_face_angle(face_image)
        occlusion_score = self._assess_occlusion(face_image)
        
        # Lighting assessment
        lighting_score = self._assess_lighting(face_image)
        
        # Overall quality calculation
        overall_score = (
            sharpness * 0.25 +
            brightness * 0.15 +
            contrast * 0.15 +
            face_size_score * 0.15 +
            angle_score * 0.15 +
            occlusion_score * 0.10 +
            lighting_score * 0.05
        )
        
        return {
            "overall_score": min(max(overall_score, 0.0), 1.0),
            "metrics": {
                "sharpness": sharpness,
                "brightness": brightness,
                "contrast": contrast,
                "face_size_score": face_size_score,
                "angle_score": angle_score,
                "occlusion_score": occlusion_score,
                "lighting_score": lighting_score
            },
            "recommendations": self._generate_quality_recommendations(
                sharpness, brightness, contrast, face_size_score, angle_score
            )
        }
        
    def _calculate_sharpness(self, image: np.ndarray) -> float:
        """คำนวณความคมชัดของภาพ"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Normalize to 0-1 range
        return min(laplacian_var / 1000.0, 1.0)
        
    def _calculate_brightness(self, image: np.ndarray) -> float:
        """คำนวณความสว่างของภาพ"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray) / 255.0
        
        # Optimal brightness is around 0.4-0.7
        if 0.4 <= mean_brightness <= 0.7:
            return 1.0
        elif mean_brightness < 0.4:
            return mean_brightness / 0.4
        else:
            return (1.0 - mean_brightness) / 0.3
            
    def _calculate_contrast(self, image: np.ndarray) -> float:
        """คำนวณความคมชัดของภาพ"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        contrast = gray.std() / 255.0
        
        # Normalize to 0-1 range (good contrast is around 0.3-0.7)
        return min(contrast / 0.5, 1.0)
        
    def _assess_face_size(self, face_image: np.ndarray) -> float:
        """ประเมินขนาดใบหน้า"""
        height, width = face_image.shape[:2]
        face_area = height * width
        
        if face_area < self.min_face_size ** 2:
            return face_area / (self.min_face_size ** 2)
        elif face_area > self.optimal_face_size ** 2:
            return 1.0
        else:
            return face_area / (self.optimal_face_size ** 2)
            
    def _generate_quality_recommendations(
        self, 
        sharpness: float,
        brightness: float, 
        contrast: float,
        face_size_score: float,
        angle_score: float
    ) -> List[str]:
        """สร้างคำแนะนำสำหรับปรับปรุงคุณภาพ"""
        recommendations = []
        
        if sharpness < 0.5:
            recommendations.append("Image is blurry - use better focus or stabilization")
        if brightness < 0.4:
            recommendations.append("Image is too dark - increase lighting")
        elif brightness > 0.8:
            recommendations.append("Image is too bright - reduce lighting")
        if contrast < 0.3:
            recommendations.append("Low contrast - adjust lighting conditions")
        if face_size_score < 0.6:
            recommendations.append("Face too small - move closer to camera")
        if angle_score < 0.7:
            recommendations.append("Face angle not optimal - face camera directly")
            
        return recommendations
```

#### Real-time Face Tracking

```python
# app/services/face_detection/tracker.py
import numpy as np
import cv2
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import asyncio
from collections import defaultdict

@dataclass
class TrackedFace:
    track_id: str
    face_id: str
    bbox: Dict[str, float]
    confidence: float
    landmarks: Optional[Dict[str, Any]]
    first_seen: float
    last_seen: float
    track_confidence: float

class FaceTracker:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.active_tracks = {}
        self.next_track_id = 1
        self.max_tracks = config.get("max_tracks", 20)
        self.track_timeout = config.get("track_timeout", 5.0)  # seconds
        
        # Tracking parameters
        self.iou_threshold = config.get("iou_threshold", 0.5)
        self.confidence_threshold = config.get("confidence_threshold", 0.8)
        
    async def update_tracks(
        self, 
        detections: List[Dict[str, Any]],
        timestamp: float
    ) -> List[TrackedFace]:
        """อัพเดท face tracking"""
        
        # 1. Remove expired tracks
        self._remove_expired_tracks(timestamp)
        
        # 2. Match detections to existing tracks
        matches, unmatched_detections, unmatched_tracks = self._match_detections_to_tracks(
            detections, timestamp
        )
        
        # 3. Update matched tracks
        for detection_idx, track_id in matches:
            self._update_track(track_id, detections[detection_idx], timestamp)
            
        # 4. Create new tracks for unmatched detections
        for detection_idx in unmatched_detections:
            if len(self.active_tracks) < self.max_tracks:
                self._create_new_track(detections[detection_idx], timestamp)
                
        # 5. Mark unmatched tracks as lost
        for track_id in unmatched_tracks:
            if track_id in self.active_tracks:
                self.active_tracks[track_id].track_confidence *= 0.8
                
        return list(self.active_tracks.values())
        
    def _match_detections_to_tracks(
        self, 
        detections: List[Dict[str, Any]], 
        timestamp: float
    ) -> tuple:
        """จับคู่ detection กับ existing tracks"""
        
        if not self.active_tracks or not detections:
            return [], list(range(len(detections))), list(self.active_tracks.keys())
            
        # Calculate IoU matrix
        iou_matrix = np.zeros((len(detections), len(self.active_tracks)))
        track_ids = list(self.active_tracks.keys())
        
        for det_idx, detection in enumerate(detections):
            det_bbox = detection["bbox"]
            
            for track_idx, track_id in enumerate(track_ids):
                track_bbox = self.active_tracks[track_id].bbox
                iou = self._calculate_iou(det_bbox, track_bbox)
                iou_matrix[det_idx, track_idx] = iou
                
        # Hungarian algorithm or simple greedy matching
        matches = []
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = track_ids.copy()
        
        # Simple greedy matching (can be improved with Hungarian algorithm)
        while len(unmatched_detections) > 0 and len(unmatched_tracks) > 0:
            # Find best match
            best_iou = 0
            best_det_idx = -1
            best_track_idx = -1
            
            for det_idx in unmatched_detections:
                for track_idx, track_id in enumerate(unmatched_tracks):
                    if track_id in track_ids:
                        actual_track_idx = track_ids.index(track_id)
                        iou = iou_matrix[det_idx, actual_track_idx]
                        
                        if iou > best_iou and iou > self.iou_threshold:
                            best_iou = iou
                            best_det_idx = det_idx
                            best_track_idx = track_idx
                            
            if best_det_idx >= 0:
                matches.append((best_det_idx, unmatched_tracks[best_track_idx]))
                unmatched_detections.remove(best_det_idx)
                unmatched_tracks.pop(best_track_idx)
            else:
                break
                
        return matches, unmatched_detections, unmatched_tracks
        
    def _calculate_iou(self, bbox1: List[float], bbox2: Dict[str, float]) -> float:
        """คำนวณ Intersection over Union"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2["x"], bbox2["y"], bbox2["width"], bbox2["height"]
        
        # Calculate intersection
        ix1 = max(x1, x2)
        iy1 = max(y1, y2)
        ix2 = min(x1 + w1, x2 + w2)
        iy2 = min(y1 + h1, y2 + h2)
        
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
            
        intersection = (ix2 - ix1) * (iy2 - iy1)
        union = w1 * h1 + w2 * h2 - intersection
        
        return intersection / union if union > 0 else 0.0
        
    def _create_new_track(self, detection: Dict[str, Any], timestamp: float):
        """สร้าง track ใหม่"""
        track_id = f"track_{self.next_track_id:04d}"
        self.next_track_id += 1
        
        face_id = f"face_{track_id}"
        
        tracked_face = TrackedFace(
            track_id=track_id,
            face_id=face_id,
            bbox=detection["bbox"],
            confidence=detection["confidence"],
            landmarks=detection.get("landmarks"),
            first_seen=timestamp,
            last_seen=timestamp,
            track_confidence=1.0
        )
        
        self.active_tracks[track_id] = tracked_face
        
    def _update_track(self, track_id: str, detection: Dict[str, Any], timestamp: float):
        """อัพเดท existing track"""
        if track_id in self.active_tracks:
            track = self.active_tracks[track_id]
            
            # Update with detection data
            track.bbox = detection["bbox"]
            track.confidence = detection["confidence"]
            track.landmarks = detection.get("landmarks")
            track.last_seen = timestamp
            
            # Increase track confidence
            track.track_confidence = min(track.track_confidence + 0.1, 1.0)
            
    def _remove_expired_tracks(self, current_timestamp: float):
        """ลบ track ที่หมดอายุ"""
        expired_tracks = []
        
        for track_id, track in self.active_tracks.items():
            if current_timestamp - track.last_seen > self.track_timeout:
                expired_tracks.append(track_id)
                
        for track_id in expired_tracks:
            del self.active_tracks[track_id]
```

---

## 4. Database Integration

### 4.1 PostgreSQL Operations

```python
# app/services/database/postgres.py
from sqlalchemy import select, insert, update, delete
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime

from app.models.detection import DetectionRequest, DetectedFace
from app.core.database import get_db

class FaceDetectionRepository:
    
    async def create_detection_request(
        self,
        db: Session,
        request_data: Dict[str, Any]
    ) -> str:
        """สร้าง detection request ใหม่"""
        
        request_id = str(uuid.uuid4())
        
        detection_request = DetectionRequest(
            id=uuid.uuid4(),
            request_id=request_id,
            user_id=request_data.get("user_id"),
            image_hash=request_data["image_hash"],
            image_url=request_data.get("image_url"),
            image_metadata=request_data.get("image_metadata"),
            min_face_size=request_data.get("min_face_size", 40),
            max_faces=request_data.get("max_faces", 100),
            detection_confidence=request_data.get("detection_confidence", 0.7),
            return_landmarks=request_data.get("return_landmarks", True),
            return_attributes=request_data.get("return_attributes", False),
            request_source=request_data.get("request_source"),
            ip_address=request_data.get("ip_address"),
            status="processing"
        )
        
        db.add(detection_request)
        await db.commit()
        
        return request_id
        
    async def save_detection_results(
        self,
        db: Session,
        request_id: str,
        faces: List[Dict[str, Any]],
        processing_time_ms: int
    ):
        """บันทึกผลการตรวจจับ"""
        
        # Update request status
        stmt = update(DetectionRequest).where(
            DetectionRequest.request_id == request_id
        ).values(
            faces_detected=len(faces),
            processing_time_ms=processing_time_ms,
            status="completed",
            completed_at=datetime.utcnow()
        )
        await db.execute(stmt)
        
        # Save detected faces
        for face_data in faces:
            detected_face = DetectedFace(
                detection_request_id=request_id,
                face_index=face_data["face_index"],
                bbox_x=int(face_data["bbox"]["x"]),
                bbox_y=int(face_data["bbox"]["y"]),
                bbox_width=int(face_data["bbox"]["width"]),
                bbox_height=int(face_data["bbox"]["height"]),
                detection_confidence=face_data["bbox"]["confidence"],
                face_quality_score=face_data.get("quality_score"),
                landmarks=face_data.get("landmarks"),
                estimated_age=face_data.get("attributes", {}).get("estimated_age"),
                estimated_gender=face_data.get("attributes", {}).get("estimated_gender"),
                gender_confidence=face_data.get("attributes", {}).get("gender_confidence"),
                emotions=face_data.get("attributes", {}).get("emotion_scores")
            )
            
            db.add(detected_face)
            
        await db.commit()
        
    async def get_detection_results(
        self,
        db: Session,
        request_id: str
    ) -> Optional[Dict[str, Any]]:
        """ดึงผลการตรวจจับ"""
        
        # Get request info
        request_stmt = select(DetectionRequest).where(
            DetectionRequest.request_id == request_id
        )
        request_result = await db.execute(request_stmt)
        request = request_result.scalar_one_or_none()
        
        if not request:
            return None
            
        # Get detected faces
        faces_stmt = select(DetectedFace).where(
            DetectedFace.detection_request_id == request_id
        ).order_by(DetectedFace.face_index)
        
        faces_result = await db.execute(faces_stmt)
        faces = faces_result.scalars().all()
        
        return {
            "request_id": request_id,
            "status": request.status,
            "faces_detected": request.faces_detected,
            "processing_time_ms": request.processing_time_ms,
            "faces": [self._face_to_dict(face) for face in faces],
            "created_at": request.created_at,
            "completed_at": request.completed_at
        }
        
    def _face_to_dict(self, face: DetectedFace) -> Dict[str, Any]:
        """แปลง DetectedFace เป็น dictionary"""
        return {
            "face_index": face.face_index,
            "bbox": {
                "x": face.bbox_x,
                "y": face.bbox_y,
                "width": face.bbox_width,
                "height": face.bbox_height,
                "confidence": float(face.detection_confidence)
            },
            "quality_score": float(face.face_quality_score) if face.face_quality_score else None,
            "landmarks": face.landmarks,
            "attributes": {
                "estimated_age": float(face.estimated_age) if face.estimated_age else None,
                "estimated_gender": face.estimated_gender,
                "gender_confidence": float(face.gender_confidence) if face.gender_confidence else None,
                "emotions": face.emotions
            }
        }
```

---

## 5. Performance Optimization

### 5.1 Model Optimization

```python
# app/services/face_detection/model_optimizer.py
import onnxruntime as ort
from typing import Dict, Any
import numpy as np

class ModelOptimizer:
    
    @staticmethod
    def optimize_onnx_session(model_path: str, config: Dict[str, Any]) -> ort.InferenceSession:
        """Optimize ONNX model for inference"""
        
        # Session options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        # Enable parallel execution
        sess_options.intra_op_num_threads = config.get("intra_op_threads", 4)
        sess_options.inter_op_num_threads = config.get("inter_op_threads", 2)
        
        # Execution providers
        providers = []
        if config.get("use_gpu", False):
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        
        return ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=providers
        )
        
    @staticmethod
    def batch_inference(
        model: ort.InferenceSession,
        inputs: List[np.ndarray],
        batch_size: int = 8
    ) -> List[np.ndarray]:
        """Batch inference for better performance"""
        
        results = []
        input_name = model.get_inputs()[0].name
        output_names = [o.name for o in model.get_outputs()]
        
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i + batch_size]
            batch_array = np.stack(batch, axis=0)
            
            batch_results = model.run(output_names, {input_name: batch_array})
            results.extend(batch_results)
            
        return results
```

### 5.2 Caching Strategy

```python
# app/services/cache/face_detection_cache.py
import redis
import json
import hashlib
from typing import Optional, Dict, Any, List

class FaceDetectionCache:
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.cache_ttl = 3600  # 1 hour
        
    def _generate_cache_key(self, image_hash: str, options: Dict[str, Any]) -> str:
        """Generate cache key for detection results"""
        options_str = json.dumps(options, sort_keys=True)
        combined = f"{image_hash}:{options_str}"
        return f"face_detection:{hashlib.md5(combined.encode()).hexdigest()}"
        
    async def get_cached_results(
        self, 
        image_hash: str, 
        options: Dict[str, Any]
    ) -> Optional[List[Dict[str, Any]]]:
        """Get cached detection results"""
        cache_key = self._generate_cache_key(image_hash, options)
        
        try:
            cached_data = await self.redis.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
        except Exception:
            pass
            
        return None
        
    async def cache_results(
        self,
        image_hash: str,
        options: Dict[str, Any],
        results: List[Dict[str, Any]]
    ):
        """Cache detection results"""
        cache_key = self._generate_cache_key(image_hash, options)
        
        try:
            await self.redis.setex(
                cache_key,
                self.cache_ttl,
                json.dumps(results)
            )
        except Exception:
            pass  # Don't fail if caching fails
```

---

## 6. Monitoring และ Analytics

### 6.1 Performance Metrics

- **Detection Accuracy**: Face detection precision and recall
- **Processing Time**: Average time per image and per face
- **Throughput**: Images processed per second
- **Model Performance**: Individual model accuracy and speed

### 6.2 Quality Metrics

- **Face Quality Distribution**: Distribution of quality scores
- **Detection Confidence**: Distribution of detection confidence
- **Landmark Accuracy**: Landmark detection precision
- **False Positive Rate**: Rate of incorrect detections

---

Face Detection Service เป็นบริการพื้นฐานที่สำคัญที่สุดในระบบ AI เนื่องจากเป็นจุดเริ่มต้นของกระบวนการวิเคราะห์ใบหน้าทั้งหมด โดยต้องมีความแม่นยำสูงและประสิทธิภาพในการประมวลผลที่รวดเร็ว เพื่อรองรับการใช้งานในหลากหลายสถานการณ์
