# Face Antispoofing Service - Liveness Detection และ Spoof Prevention

## 1. Service Overview

### หน้าที่หลัก
- **Passive Antispoofing**: ตรวจสอบ spoof จากรูปภาพเดียว
- **Active Liveness Detection**: ตรวจสอบ liveness ด้วยการทำตามคำสั่ง
- **Real-time Video Analysis**: วิเคราะห์ video stream แบบเรียลไทม์
- **Multi-modal Detection**: รวมหลายเทคนิคการตรวจสอบ

### Integration Flow ใน Frontend
```
Login Page → Face Recognition → Antispoofing Check → Authentication Success
Video Call → Real-time Liveness → Continuous Monitoring
Payment → High Security Liveness → Transaction Approval
Profile Setup → Basic Antispoofing → Account Verification
```

## 2. API Endpoints

### 2.1 Passive Antispoofing (Single Image)
```http
POST /api/v1/antispoofing/check-passive
Content-Type: application/json
Authorization: Bearer {token}

{
  "image": "base64_encoded_image",
  "analysis_level": "standard", // "basic", "standard", "advanced"
  "metadata": {
    "source": "login_attempt",
    "session_id": "session_uuid",
    "device_info": {...}
  }
}
```

**Response (200 OK):**
```json
{
  "success": true,
  "data": {
    "is_live": true,
    "confidence_score": 0.9456,
    "spoof_type": null, // null if live, else "photo", "video", "mask", "screen"
    "risk_level": "low", // "low", "medium", "high"
    "analysis_details": {
      "texture_analysis": {
        "score": 0.92,
        "indicators": ["natural_skin_texture", "micro_movements"]
      },
      "depth_analysis": {
        "score": 0.88,
        "indicators": ["3d_structure", "lighting_consistency"]
      },
      "quality_metrics": {
        "resolution": "1920x1080",
        "brightness": 0.65,
        "contrast": 0.78,
        "sharpness": 0.89
      }
    },
    "recommendations": []
  },
  "metadata": {
    "request_id": "req_uuid",
    "processing_time": "156ms",
    "model_version": "antispoofing_v2.1.0",
    "timestamp": "2024-01-20T10:30:00Z"
  }
}
```

### 2.2 Active Liveness Detection (Interactive)
```http
POST /api/v1/antispoofing/start-liveness-session
Content-Type: application/json
Authorization: Bearer {token}

{
  "user_id": "uuid",
  "session_type": "standard", // "basic", "standard", "strict"
  "challenges": ["turn_left", "turn_right", "blink", "nod"], // optional: custom challenges
  "timeout_seconds": 30,
  "max_attempts": 3,
  "metadata": {
    "source": "payment_verification",
    "transaction_id": "tx_uuid"
  }
}
```

**Response (201 Created):**
```json
{
  "success": true,
  "data": {
    "session_token": "liveness_session_token_abc123",
    "session_id": "session_uuid",
    "challenges_required": ["turn_left", "turn_right", "blink"],
    "current_challenge": "turn_left",
    "instructions": {
      "turn_left": {
        "text": "โปรดหันหน้าไปทางซ้าย",
        "duration_seconds": 3,
        "detection_criteria": {
          "min_angle": 15,
          "max_angle": 45
        }
      }
    },
    "session_config": {
      "timeout_seconds": 30,
      "max_attempts": 3,
      "expires_at": "2024-01-20T10:31:00Z"
    }
  },
  "metadata": {
    "request_id": "req_uuid",
    "processing_time": "23ms",
    "timestamp": "2024-01-20T10:30:00Z"
  }
}
```

### 2.3 Submit Liveness Response
```http
POST /api/v1/antispoofing/liveness-response
Content-Type: application/json
Authorization: Bearer {token}

{
  "session_token": "liveness_session_token_abc123",
  "challenge": "turn_left",
  "response_data": {
    "video_frames": ["base64_frame1", "base64_frame2", "base64_frame3"],
    "timestamps": [0, 1000, 2000], // milliseconds
    "challenge_metadata": {
      "start_time": "2024-01-20T10:30:10Z",
      "end_time": "2024-01-20T10:30:13Z"
    }
  }
}
```

**Response (200 OK):**
```json
{
  "success": true,
  "data": {
    "challenge_result": "success", // "success", "failed", "retry"
    "challenge": "turn_left",
    "verification_details": {
      "movement_detected": true,
      "angle_range": [18, 42],
      "timing_score": 0.95,
      "smoothness_score": 0.88
    },
    "session_status": {
      "current_challenge": "turn_right", // next challenge
      "challenges_completed": ["turn_left"],
      "challenges_remaining": ["turn_right", "blink"],
      "attempts_used": 1,
      "session_progress": 33.3
    },
    "next_instruction": {
      "challenge": "turn_right",
      "text": "โปรดหันหน้าไปทางขวา",
      "duration_seconds": 3
    }
  },
  "metadata": {
    "request_id": "req_uuid",
    "processing_time": "234ms",
    "model_version": "liveness_v2.1.0",
    "timestamp": "2024-01-20T10:30:15Z"
  }
}
```

### 2.4 Get Liveness Session Result
```http
GET /api/v1/antispoofing/liveness-session/{session_token}
Authorization: Bearer {token}
```

**Response (200 OK):**
```json
{
  "success": true,
  "data": {
    "session_id": "session_uuid",
    "session_status": "completed", // "active", "completed", "failed", "expired"
    "overall_result": true,
    "final_confidence": 0.9234,
    "challenge_results": [
      {
        "challenge": "turn_left",
        "result": "success",
        "confidence": 0.95,
        "attempt_number": 1
      },
      {
        "challenge": "turn_right", 
        "result": "success",
        "confidence": 0.89,
        "attempt_number": 1
      },
      {
        "challenge": "blink",
        "result": "success", 
        "confidence": 0.93,
        "attempt_number": 1
      }
    ],
    "session_summary": {
      "total_challenges": 3,
      "successful_challenges": 3,
      "total_attempts": 3,
      "session_duration": "18.5s",
      "risk_indicators": []
    },
    "completed_at": "2024-01-20T10:30:28Z"
  },
  "metadata": {
    "request_id": "req_uuid",
    "processing_time": "12ms",
    "timestamp": "2024-01-20T10:30:30Z"
  }
}
```

### 2.5 Real-time Video Stream Analysis
```http
POST /api/v1/antispoofing/analyze-stream
Content-Type: application/json
Authorization: Bearer {token}

{
  "stream_id": "stream_uuid",
  "analysis_mode": "continuous", // "continuous", "periodic", "on_demand"
  "frame_rate": 5, // frames per second to analyze
  "detection_sensitivity": "medium", // "low", "medium", "high"
  "callback_url": "https://app.com/webhook/antispoofing", // optional
  "metadata": {
    "source": "video_call",
    "call_id": "call_uuid"
  }
}
```

### 2.6 Batch Image Analysis
```http
POST /api/v1/antispoofing/batch-analyze
Content-Type: application/json
Authorization: Bearer {token}

{
  "images": [
    {
      "image_id": "img_001",
      "image": "base64_encoded_image_1"
    },
    {
      "image_id": "img_002",
      "image": "base64_encoded_image_2"
    }
  ],
  "analysis_level": "standard",
  "processing_options": {
    "async": true,
    "priority": "normal" // "low", "normal", "high"
  }
}
```

## 3. Service Implementation

### 3.1 Project Structure
```
antispoofing-service/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application
│   ├── api/
│   │   ├── __init__.py
│   │   ├── v1/
│   │   │   ├── __init__.py
│   │   │   ├── routes/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── passive.py      # Passive antispoofing
│   │   │   │   ├── liveness.py     # Active liveness detection
│   │   │   │   ├── stream.py       # Real-time analysis
│   │   │   │   └── batch.py        # Batch processing
│   │   │   └── deps.py
│   │   └── middleware/
│   │       ├── __init__.py
│   │       ├── auth.py
│   │       ├── rate_limit.py
│   │       └── session.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   └── database.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── antispoofing.py
│   │   └── liveness.py
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── requests.py
│   │   ├── responses.py
│   │   └── sessions.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── antispoofing/
│   │   │   ├── __init__.py
│   │   │   ├── passive_detector.py     # Passive spoof detection
│   │   │   ├── liveness_detector.py    # Active liveness detection
│   │   │   ├── texture_analyzer.py     # Texture analysis
│   │   │   ├── depth_analyzer.py       # Depth/3D analysis
│   │   │   └── challenge_manager.py    # Liveness challenge management
│   │   ├── video/
│   │   │   ├── __init__.py
│   │   │   ├── stream_processor.py     # Video stream processing
│   │   │   ├── frame_analyzer.py       # Frame-by-frame analysis
│   │   │   └── motion_detector.py      # Motion analysis
│   │   ├── database/
│   │   │   ├── __init__.py
│   │   │   ├── postgres.py
│   │   │   └── redis.py               # Session management
│   │   └── external/
│   │       ├── __init__.py
│   │       └── face_detection.py      # Face detection client
│   └── utils/
│       ├── __init__.py
│       ├── image.py
│       ├── video.py
│       └── validation.py
├── models/                     # AI model files
│   ├── antispoofing/
│   │   ├── passive_model.onnx
│   │   ├── texture_model.onnx
│   │   └── depth_model.onnx
│   ├── liveness/
│   │   ├── head_pose.onnx
│   │   ├── eye_blink.onnx
│   │   └── motion_analysis.onnx
│   └── quality/
│       ├── image_quality.onnx
│       └── config.yaml
├── tests/
│   ├── __init__.py
│   ├── test_api/
│   ├── test_services/
│   └── test_utils/
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── requirements.txt
└── README.md
```

### 3.2 Core Implementation

#### Passive Antispoofing Service
```python
# app/services/antispoofing/passive_detector.py
import numpy as np
import cv2
import onnxruntime as ort
from typing import Dict, Any, Optional, Tuple
from enum import Enum

class SpoofType(Enum):
    LIVE = "live"
    PHOTO = "photo"
    VIDEO = "video"
    MASK = "mask"
    SCREEN = "screen"

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"

class PassiveAntispoofingDetector:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Load models
        self.main_model = ort.InferenceSession(config["main_model_path"])
        self.texture_model = ort.InferenceSession(config["texture_model_path"])
        self.depth_model = ort.InferenceSession(config["depth_model_path"])
        
        # Thresholds
        self.live_threshold = config.get("live_threshold", 0.7)
        self.high_risk_threshold = config.get("high_risk_threshold", 0.3)
        self.medium_risk_threshold = config.get("medium_risk_threshold", 0.6)
        
    async def analyze_image(self, image: np.ndarray) -> Dict[str, Any]:
        """วิเคราะห์รูปภาพเพื่อตรวจจับ spoof"""
        
        # 1. Preprocess image
        processed_image = self._preprocess_image(image)
        if processed_image is None:
            raise ValueError("Failed to preprocess image")
            
        # 2. Run main antispoofing model
        main_score = await self._run_main_model(processed_image)
        
        # 3. Run texture analysis
        texture_result = await self._analyze_texture(processed_image)
        
        # 4. Run depth analysis  
        depth_result = await self._analyze_depth(processed_image)
        
        # 5. Combine results
        final_result = self._combine_results(
            main_score, texture_result, depth_result
        )
        
        return final_result
        
    async def _run_main_model(self, image: np.ndarray) -> float:
        """Run main antispoofing model"""
        try:
            input_name = self.main_model.get_inputs()[0].name
            output_name = self.main_model.get_outputs()[0].name
            
            # Prepare input
            input_data = self._prepare_model_input(image, (224, 224))
            
            # Run inference
            result = self.main_model.run([output_name], {input_name: input_data})
            
            # Get live probability
            live_prob = float(result[0][0][1])  # Assuming [fake, live] output
            return live_prob
            
        except Exception as e:
            raise RuntimeError(f"Main model inference failed: {e}")
            
    async def _analyze_texture(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze texture patterns for spoof detection"""
        try:
            # Convert to grayscale for texture analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Extract texture features
            lbp_features = self._extract_lbp_features(gray)
            gabor_features = self._extract_gabor_features(gray)
            
            # Run texture model
            input_data = np.concatenate([lbp_features, gabor_features])
            input_data = input_data.reshape(1, -1).astype(np.float32)
            
            input_name = self.texture_model.get_inputs()[0].name
            output_name = self.texture_model.get_outputs()[0].name
            
            result = self.texture_model.run([output_name], {input_name: input_data})
            texture_score = float(result[0][0])
            
            return {
                "score": texture_score,
                "indicators": self._get_texture_indicators(texture_score),
                "features": {
                    "lbp_variance": float(np.var(lbp_features)),
                    "gabor_energy": float(np.mean(gabor_features))
                }
            }
            
        except Exception as e:
            return {
                "score": 0.5,
                "indicators": ["analysis_failed"],
                "error": str(e)
            }
            
    async def _analyze_depth(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze depth cues for 3D face detection"""
        try:
            # Prepare input for depth model
            input_data = self._prepare_model_input(image, (160, 160))
            
            input_name = self.depth_model.get_inputs()[0].name
            output_name = self.depth_model.get_outputs()[0].name
            
            result = self.depth_model.run([output_name], {input_name: input_data})
            depth_score = float(result[0][0])
            
            # Analyze lighting consistency
            lighting_score = self._analyze_lighting(image)
            
            # Analyze shadow patterns
            shadow_score = self._analyze_shadows(image)
            
            combined_score = (depth_score + lighting_score + shadow_score) / 3
            
            return {
                "score": combined_score,
                "indicators": self._get_depth_indicators(combined_score),
                "details": {
                    "depth_consistency": depth_score,
                    "lighting_consistency": lighting_score,
                    "shadow_patterns": shadow_score
                }
            }
            
        except Exception as e:
            return {
                "score": 0.5,
                "indicators": ["analysis_failed"],
                "error": str(e)
            }
            
    def _combine_results(
        self, 
        main_score: float, 
        texture_result: Dict[str, Any], 
        depth_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Combine all analysis results"""
        
        # Weighted combination
        weights = {
            "main": 0.5,
            "texture": 0.3,
            "depth": 0.2
        }
        
        final_score = (
            main_score * weights["main"] +
            texture_result["score"] * weights["texture"] +
            depth_result["score"] * weights["depth"]
        )
        
        # Determine if live
        is_live = final_score >= self.live_threshold
        
        # Determine spoof type if not live
        spoof_type = None if is_live else self._determine_spoof_type(
            main_score, texture_result, depth_result
        )
        
        # Determine risk level
        risk_level = self._determine_risk_level(final_score)
        
        return {
            "is_live": is_live,
            "confidence_score": final_score,
            "spoof_type": spoof_type,
            "risk_level": risk_level.value,
            "analysis_details": {
                "main_model_score": main_score,
                "texture_analysis": texture_result,
                "depth_analysis": depth_result,
                "combination_weights": weights
            }
        }
        
    def _determine_spoof_type(
        self, 
        main_score: float, 
        texture_result: Dict[str, Any], 
        depth_result: Dict[str, Any]
    ) -> str:
        """Determine the type of spoof attack"""
        
        texture_score = texture_result["score"]
        depth_score = depth_result["score"]
        
        # Photo attack: low texture variation, poor depth
        if texture_score < 0.3 and depth_score < 0.4:
            return SpoofType.PHOTO.value
            
        # Video attack: good texture, poor depth, temporal inconsistency
        elif texture_score > 0.6 and depth_score < 0.4:
            return SpoofType.VIDEO.value
            
        # Mask attack: unusual texture patterns
        elif "unusual_texture" in texture_result.get("indicators", []):
            return SpoofType.MASK.value
            
        # Screen attack: lighting inconsistencies
        elif "lighting_inconsistency" in depth_result.get("indicators", []):
            return SpoofType.SCREEN.value
            
        # Default to photo if unclear
        else:
            return SpoofType.PHOTO.value
            
    def _determine_risk_level(self, score: float) -> RiskLevel:
        """Determine risk level based on confidence score"""
        if score >= self.medium_risk_threshold:
            return RiskLevel.LOW
        elif score >= self.high_risk_threshold:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.HIGH
            
    # Helper methods for feature extraction
    def _extract_lbp_features(self, gray_image: np.ndarray) -> np.ndarray:
        """Extract Local Binary Pattern features"""
        from skimage.feature import local_binary_pattern
        
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(gray_image, n_points, radius, method='uniform')
        
        # Calculate histogram
        hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, 
                             range=(0, n_points + 2), density=True)
        return hist.astype(np.float32)
        
    def _extract_gabor_features(self, gray_image: np.ndarray) -> np.ndarray:
        """Extract Gabor filter features"""
        from skimage.filters import gabor
        
        features = []
        
        # Multiple orientations and frequencies
        for theta in [0, 45, 90, 135]:
            for frequency in [0.1, 0.3, 0.5]:
                real, _ = gabor(gray_image, frequency=frequency, 
                              theta=np.deg2rad(theta))
                features.extend([
                    np.mean(real),
                    np.std(real),
                    np.var(real)
                ])
                
        return np.array(features, dtype=np.float32)
        
    def _analyze_lighting(self, image: np.ndarray) -> float:
        """Analyze lighting consistency"""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        
        # Calculate lighting uniformity
        lighting_std = np.std(l_channel)
        lighting_mean = np.mean(l_channel)
        
        # Normalize score (lower std indicates more uniform lighting)
        uniformity_score = 1.0 / (1.0 + lighting_std / lighting_mean)
        
        return min(uniformity_score, 1.0)
        
    def _analyze_shadows(self, image: np.ndarray) -> float:
        """Analyze shadow patterns"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Edge detection for shadow boundaries
        edges = cv2.Canny(gray, 50, 150)
        
        # Calculate edge density (natural faces have moderate edge density)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Score based on expected edge density range
        if 0.05 <= edge_density <= 0.15:
            return 0.9
        elif 0.03 <= edge_density <= 0.20:
            return 0.7
        else:
            return 0.3
```

#### Active Liveness Detection Service
```python
# app/services/antispoofing/liveness_detector.py
import numpy as np
import cv2
from typing import Dict, Any, List, Optional
from enum import Enum
import asyncio

class ChallengeType(Enum):
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    TURN_UP = "turn_up"
    TURN_DOWN = "turn_down"
    BLINK = "blink"
    NOD = "nod"
    SMILE = "smile"

class ChallengeResult(Enum):
    SUCCESS = "success"
    FAILED = "failed"
    RETRY = "retry"

class LivenessDetector:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Load models
        self.pose_model = ort.InferenceSession(config["pose_model_path"])
        self.blink_model = ort.InferenceSession(config["blink_model_path"])
        self.expression_model = ort.InferenceSession(config["expression_model_path"])
        
        # Challenge configurations
        self.challenge_configs = {
            ChallengeType.TURN_LEFT: {
                "min_angle": 15,
                "max_angle": 45,
                "duration_seconds": 3,
                "tolerance": 5
            },
            ChallengeType.TURN_RIGHT: {
                "min_angle": 15,
                "max_angle": 45,
                "duration_seconds": 3,
                "tolerance": 5
            },
            ChallengeType.BLINK: {
                "min_blinks": 2,
                "max_blinks": 5,
                "duration_seconds": 4,
                "eye_close_threshold": 0.3
            },
            ChallengeType.NOD: {
                "min_angle": 10,
                "max_angle": 30,
                "duration_seconds": 3,
                "cycles": 2
            }
        }
        
    async def verify_challenge_response(
        self,
        challenge_type: ChallengeType,
        video_frames: List[np.ndarray],
        timestamps: List[int]
    ) -> Dict[str, Any]:
        """Verify challenge response from video frames"""
        
        if challenge_type == ChallengeType.TURN_LEFT:
            return await self._verify_head_turn(video_frames, "left")
        elif challenge_type == ChallengeType.TURN_RIGHT:
            return await self._verify_head_turn(video_frames, "right")
        elif challenge_type == ChallengeType.BLINK:
            return await self._verify_blink(video_frames, timestamps)
        elif challenge_type == ChallengeType.NOD:
            return await self._verify_nod(video_frames, timestamps)
        else:
            raise ValueError(f"Unsupported challenge type: {challenge_type}")
            
    async def _verify_head_turn(
        self, 
        frames: List[np.ndarray], 
        direction: str
    ) -> Dict[str, Any]:
        """Verify head turn challenge"""
        
        try:
            pose_angles = []
            
            # Analyze each frame
            for frame in frames:
                pose = await self._detect_head_pose(frame)
                if pose:
                    pose_angles.append(pose["yaw"])
                    
            if not pose_angles:
                return {
                    "result": ChallengeResult.FAILED.value,
                    "reason": "no_pose_detected",
                    "confidence": 0.0
                }
                
            # Calculate angle range
            min_angle = min(pose_angles)
            max_angle = max(pose_angles)
            angle_range = max_angle - min_angle
            
            # Get challenge configuration
            config = self.challenge_configs[
                ChallengeType.TURN_LEFT if direction == "left" 
                else ChallengeType.TURN_RIGHT
            ]
            
            # Check if movement meets requirements
            required_min = config["min_angle"]
            required_max = config["max_angle"]
            
            # Determine direction factor
            direction_factor = -1 if direction == "left" else 1
            peak_angle = max_angle if direction == "right" else abs(min_angle)
            
            # Verify movement
            movement_detected = angle_range >= required_min
            correct_direction = (
                (direction == "right" and max_angle > required_min) or
                (direction == "left" and abs(min_angle) > required_min)
            )
            
            # Calculate smoothness score
            smoothness_score = self._calculate_movement_smoothness(pose_angles)
            
            # Calculate timing score
            timing_score = self._calculate_timing_score(
                pose_angles, config["duration_seconds"]
            )
            
            # Overall confidence
            confidence = (
                (0.4 if movement_detected else 0.0) +
                (0.4 if correct_direction else 0.0) +
                (0.1 * smoothness_score) +
                (0.1 * timing_score)
            )
            
            # Determine result
            if confidence >= 0.7:
                result = ChallengeResult.SUCCESS.value
            elif confidence >= 0.4:
                result = ChallengeResult.RETRY.value
            else:
                result = ChallengeResult.FAILED.value
                
            return {
                "result": result,
                "confidence": confidence,
                "verification_details": {
                    "movement_detected": movement_detected,
                    "correct_direction": correct_direction,
                    "angle_range": [min_angle, max_angle],
                    "peak_angle": peak_angle,
                    "smoothness_score": smoothness_score,
                    "timing_score": timing_score
                }
            }
            
        except Exception as e:
            return {
                "result": ChallengeResult.FAILED.value,
                "reason": "analysis_error",
                "error": str(e),
                "confidence": 0.0
            }
            
    async def _verify_blink(
        self, 
        frames: List[np.ndarray], 
        timestamps: List[int]
    ) -> Dict[str, Any]:
        """Verify blink challenge"""
        
        try:
            eye_states = []
            
            # Analyze each frame for eye state
            for frame in frames:
                eye_state = await self._detect_eye_state(frame)
                eye_states.append(eye_state)
                
            # Count blinks
            blink_count = self._count_blinks(eye_states)
            
            # Get challenge configuration
            config = self.challenge_configs[ChallengeType.BLINK]
            
            # Verify blink count
            valid_blink_count = (
                config["min_blinks"] <= blink_count <= config["max_blinks"]
            )
            
            # Calculate naturalness score
            naturalness_score = self._calculate_blink_naturalness(
                eye_states, timestamps
            )
            
            # Overall confidence
            confidence = (
                (0.6 if valid_blink_count else 0.0) +
                (0.4 * naturalness_score)
            )
            
            # Determine result
            if confidence >= 0.7:
                result = ChallengeResult.SUCCESS.value
            elif confidence >= 0.4:
                result = ChallengeResult.RETRY.value
            else:
                result = ChallengeResult.FAILED.value
                
            return {
                "result": result,
                "confidence": confidence,
                "verification_details": {
                    "blink_count": blink_count,
                    "required_range": [config["min_blinks"], config["max_blinks"]],
                    "valid_count": valid_blink_count,
                    "naturalness_score": naturalness_score,
                    "eye_states": eye_states
                }
            }
            
        except Exception as e:
            return {
                "result": ChallengeResult.FAILED.value,
                "reason": "analysis_error",
                "error": str(e),
                "confidence": 0.0
            }
            
    async def _detect_head_pose(self, frame: np.ndarray) -> Optional[Dict[str, float]]:
        """Detect head pose from frame"""
        try:
            # Preprocess frame for pose model
            input_data = self._prepare_model_input(frame, (224, 224))
            
            input_name = self.pose_model.get_inputs()[0].name
            output_names = [o.name for o in self.pose_model.get_outputs()]
            
            # Run inference
            result = self.pose_model.run(output_names, {input_name: input_data})
            
            # Extract Euler angles (yaw, pitch, roll)
            yaw = float(result[0][0])      # Left-right rotation
            pitch = float(result[1][0])    # Up-down rotation
            roll = float(result[2][0])     # Tilt rotation
            
            return {
                "yaw": yaw,
                "pitch": pitch,
                "roll": roll
            }
            
        except Exception as e:
            return None
            
    async def _detect_eye_state(self, frame: np.ndarray) -> Dict[str, float]:
        """Detect eye open/close state"""
        try:
            # Preprocess frame
            input_data = self._prepare_model_input(frame, (128, 128))
            
            input_name = self.blink_model.get_inputs()[0].name
            output_name = self.blink_model.get_outputs()[0].name
            
            # Run inference
            result = self.blink_model.run([output_name], {input_name: input_data})
            
            # Get eye openness probability
            eye_openness = float(result[0][0])
            
            return {
                "openness": eye_openness,
                "is_open": eye_openness > 0.5,
                "is_closed": eye_openness <= 0.3
            }
            
        except Exception as e:
            return {
                "openness": 0.5,
                "is_open": True,
                "is_closed": False
            }
            
    def _count_blinks(self, eye_states: List[Dict[str, Any]]) -> int:
        """Count number of blinks from eye states"""
        blink_count = 0
        in_blink = False
        
        for state in eye_states:
            if state["is_closed"] and not in_blink:
                # Start of blink
                in_blink = True
            elif state["is_open"] and in_blink:
                # End of blink
                blink_count += 1
                in_blink = False
                
        return blink_count
        
    def _calculate_movement_smoothness(self, angles: List[float]) -> float:
        """Calculate smoothness score for movement"""
        if len(angles) < 3:
            return 0.0
            
        # Calculate derivatives (velocity and acceleration)
        velocities = np.diff(angles)
        accelerations = np.diff(velocities)
        
        # Smooth movements have low acceleration variance
        acc_variance = np.var(accelerations)
        
        # Normalize score (lower variance = smoother movement)
        smoothness = 1.0 / (1.0 + acc_variance)
        
        return min(smoothness, 1.0)
        
    def _calculate_timing_score(self, angles: List[float], expected_duration: float) -> float:
        """Calculate timing score for movement"""
        # This is a simplified version - in practice you'd use actual timestamps
        actual_frames = len(angles)
        expected_frames = int(expected_duration * 30)  # Assuming 30 FPS
        
        timing_ratio = min(actual_frames, expected_frames) / max(actual_frames, expected_frames)
        
        return timing_ratio
        
    def _calculate_blink_naturalness(
        self, 
        eye_states: List[Dict[str, Any]], 
        timestamps: List[int]
    ) -> float:
        """Calculate naturalness score for blinks"""
        if len(timestamps) < 2:
            return 0.5
            
        # Check blink durations (natural blinks are 100-400ms)
        blink_durations = []
        blink_start = None
        
        for i, state in enumerate(eye_states):
            if state["is_closed"] and blink_start is None:
                blink_start = timestamps[i]
            elif state["is_open"] and blink_start is not None:
                duration = timestamps[i] - blink_start
                blink_durations.append(duration)
                blink_start = None
                
        if not blink_durations:
            return 0.0
            
        # Check if durations are in natural range (100-400ms)
        natural_count = sum(1 for d in blink_durations if 100 <= d <= 400)
        naturalness = natural_count / len(blink_durations)
        
        return naturalness
```

### 3.3 Session Management
```python
# app/services/antispoofing/session_manager.py
import uuid
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from app.services.database.redis import RedisClient

class LivenessSessionManager:
    def __init__(self, redis_client: RedisClient):
        self.redis = redis_client
        self.session_prefix = "liveness_session:"
        self.default_timeout = 300  # 5 minutes
        
    async def create_session(
        self,
        user_id: str,
        challenges: List[str],
        timeout_seconds: int = None,
        max_attempts: int = 3,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Create new liveness session"""
        
        session_token = f"liveness_{uuid.uuid4().hex}"
        session_id = str(uuid.uuid4())
        
        timeout = timeout_seconds or self.default_timeout
        expires_at = datetime.utcnow() + timedelta(seconds=timeout)
        
        session_data = {
            "session_id": session_id,
            "session_token": session_token,
            "user_id": user_id,
            "challenges_required": challenges,
            "current_challenge": challenges[0] if challenges else None,
            "challenges_completed": [],
            "attempts_count": 0,
            "max_attempts": max_attempts,
            "status": "active",
            "overall_result": None,
            "final_confidence": None,
            "challenge_results": [],
            "created_at": datetime.utcnow().isoformat(),
            "expires_at": expires_at.isoformat(),
            "metadata": metadata or {}
        }
        
        # Store in Redis
        key = f"{self.session_prefix}{session_token}"
        await self.redis.setex(
            key, 
            timeout, 
            json.dumps(session_data, default=str)
        )
        
        return session_data
        
    async def get_session(self, session_token: str) -> Optional[Dict[str, Any]]:
        """Get session data"""
        key = f"{self.session_prefix}{session_token}"
        data = await self.redis.get(key)
        
        if not data:
            return None
            
        session_data = json.loads(data)
        
        # Check if expired
        expires_at = datetime.fromisoformat(session_data["expires_at"])
        if datetime.utcnow() > expires_at:
            await self.delete_session(session_token)
            return None
            
        return session_data
        
    async def update_session(
        self, 
        session_token: str, 
        updates: Dict[str, Any]
    ) -> bool:
        """Update session data"""
        session_data = await self.get_session(session_token)
        if not session_data:
            return False
            
        # Apply updates
        session_data.update(updates)
        session_data["updated_at"] = datetime.utcnow().isoformat()
        
        # Calculate remaining TTL
        expires_at = datetime.fromisoformat(session_data["expires_at"])
        remaining_seconds = int((expires_at - datetime.utcnow()).total_seconds())
        
        if remaining_seconds <= 0:
            await self.delete_session(session_token)
            return False
            
        # Save back to Redis
        key = f"{self.session_prefix}{session_token}"
        await self.redis.setex(
            key, 
            remaining_seconds, 
            json.dumps(session_data, default=str)
        )
        
        return True
        
    async def process_challenge_response(
        self,
        session_token: str,
        challenge: str,
        result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process challenge response and update session"""
        
        session_data = await self.get_session(session_token)
        if not session_data:
            raise ValueError("Session not found or expired")
            
        if session_data["status"] != "active":
            raise ValueError("Session is not active")
            
        # Update attempts count
        session_data["attempts_count"] += 1
        
        # Store challenge result
        challenge_result = {
            "challenge": challenge,
            "result": result["result"],
            "confidence": result["confidence"],
            "attempt_number": session_data["attempts_count"],
            "timestamp": datetime.utcnow().isoformat(),
            "details": result.get("verification_details", {})
        }
        
        session_data["challenge_results"].append(challenge_result)
        
        # Handle successful challenge
        if result["result"] == "success":
            if challenge not in session_data["challenges_completed"]:
                session_data["challenges_completed"].append(challenge)
                
            # Move to next challenge
            remaining_challenges = [
                c for c in session_data["challenges_required"]
                if c not in session_data["challenges_completed"]
            ]
            
            if remaining_challenges:
                session_data["current_challenge"] = remaining_challenges[0]
            else:
                # All challenges completed
                await self._complete_session(session_data, True)
                
        # Handle failed challenge
        elif result["result"] == "failed":
            if session_data["attempts_count"] >= session_data["max_attempts"]:
                # Max attempts reached
                await self._complete_session(session_data, False)
            else:
                # Allow retry
                pass
                
        # Update session
        await self.update_session(session_token, session_data)
        
        return {
            "challenge_result": result["result"],
            "challenge": challenge,
            "verification_details": result.get("verification_details", {}),
            "session_status": {
                "current_challenge": session_data.get("current_challenge"),
                "challenges_completed": session_data["challenges_completed"],
                "challenges_remaining": [
                    c for c in session_data["challenges_required"]
                    if c not in session_data["challenges_completed"]
                ],
                "attempts_used": session_data["attempts_count"],
                "session_progress": len(session_data["challenges_completed"]) / len(session_data["challenges_required"]) * 100
            }
        }
        
    async def _complete_session(self, session_data: Dict[str, Any], success: bool):
        """Complete the session"""
        session_data["status"] = "completed" if success else "failed"
        session_data["overall_result"] = success
        session_data["completed_at"] = datetime.utcnow().isoformat()
        
        # Calculate final confidence
        if success:
            confidences = [r["confidence"] for r in session_data["challenge_results"] if r["result"] == "success"]
            session_data["final_confidence"] = sum(confidences) / len(confidences) if confidences else 0.0
        else:
            session_data["final_confidence"] = 0.0
            
    async def delete_session(self, session_token: str) -> bool:
        """Delete session"""
        key = f"{self.session_prefix}{session_token}"
        result = await self.redis.delete(key)
        return result > 0
```

Face Antispoofing Service เป็นบริการสำคัญที่ป้องกันการโจมตีด้วยภาพหรือวิดีโอปลอม โดยมีทั้งการตรวจสอบแบบ passive และ active liveness detection เพื่อให้ความปลอดภัยสูงสุดในการยืนยันตัวตนผู้ใช้
