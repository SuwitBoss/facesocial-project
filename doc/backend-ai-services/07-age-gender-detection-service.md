# Age & Gender Detection Service - การประมาณอายุและเพศ

## 1. Service Overview

### หน้าที่หลัก
- **Age Estimation**: ประมาณช่วงอายุจากภาพใบหน้า (±3 ปี)
- **Gender Classification**: จำแนกเพศ (ชาย/หญิง/ไม่ระบุ)
- **Demographic Analysis**: วิเคราะห์ข้อมูลประชากรศาสตร์
- **Batch Demographics**: วิเคราะห์หลายใบหน้าพร้อมกัน

### Integration Flow ใน Frontend
```
Auto Face Tagging → Age & Gender Detection → Demographics Display
Content Analysis → Batch Demographics → Analytics Dashboard
Profile Completion → Age Verification → Account Setup
Video Analysis → Real-time Demographics → Live Stream Info
```

### AI Models ที่ใช้
- **DEX (Deep EXpectation)**: การประมาณอายุแบบ deep learning
- **AgeGenderNet**: การจำแนกอายุและเพศพร้อมกัน
- **SSR-Net**: State-space regression สำหรับอายุ
- **FairFace**: การวิเคราะห์ demographics ที่ยุติธรรม

---

## 2. API Endpoints

### 2.1 Age & Gender Detection
```http
POST /api/v1/demographics/detect
Content-Type: multipart/form-data

Request Body:
- image: File (รูปภาพ)
- options: JSON {
    "return_confidence": true,
    "age_range_only": false,
    "include_attributes": true
  }
```

**Response:**
```json
{
  "success": true,
  "data": {
    "demographics": {
      "age": {
        "estimated_age": 25,
        "age_range": "20-30",
        "confidence": 0.87,
        "distribution": {
          "0-10": 0.02,
          "10-20": 0.15,
          "20-30": 0.65,
          "30-40": 0.15,
          "40-50": 0.03
        }
      },
      "gender": {
        "predicted_gender": "female",
        "confidence": 0.92,
        "probabilities": {
          "male": 0.08,
          "female": 0.92
        }
      },
      "additional_attributes": {
        "ethnicity_confidence": 0.84,
        "face_clarity": 0.91,
        "face_size": "medium"
      }
    },
    "face_region": {
      "bbox": {"x": 120, "y": 80, "width": 150, "height": 180},
      "landmarks": [
        {"name": "left_eye", "x": 145.2, "y": 125.8},
        {"name": "right_eye", "x": 195.6, "y": 123.4}
      ]
    }
  },
  "metadata": {
    "request_id": "demo_uuid_001",
    "processing_time": "156ms",
    "model_versions": {
      "age_model": "dex_v2.1.0",
      "gender_model": "agegender_v1.3.2"
    },
    "timestamp": "2024-01-20T10:30:00Z"
  }
}
```

### 2.2 Batch Demographics Analysis
```http
POST /api/v1/demographics/batch
Content-Type: multipart/form-data

Request Body:
- images: File[] (หลายรูปภาพ)
- options: JSON {
    "max_faces_per_image": 10,
    "return_summary": true,
    "include_statistics": true
  }
```

**Response:**
```json
{
  "success": true,
  "data": {
    "batch_results": [
      {
        "image_index": 0,
        "filename": "group_photo_1.jpg",
        "faces_detected": 3,
        "demographics": [
          {
            "face_id": 1,
            "age": {"estimated_age": 25, "range": "20-30"},
            "gender": {"predicted": "female", "confidence": 0.92}
          },
          {
            "face_id": 2,
            "age": {"estimated_age": 35, "range": "30-40"},
            "gender": {"predicted": "male", "confidence": 0.88}
          }
        ]
      }
    ],
    "summary_statistics": {
      "total_faces": 8,
      "age_distribution": {
        "0-20": 1,
        "20-30": 3,
        "30-40": 2,
        "40-50": 1,
        "50+": 1
      },
      "gender_distribution": {
        "male": 4,
        "female": 4
      },
      "average_age": 32.5,
      "age_std_deviation": 12.3
    }
  }
}
```

### 2.3 Real-time Demographics Stream
```http
POST /api/v1/demographics/stream
Content-Type: application/json

Request Body:
{
  "stream_id": "live_stream_001",
  "webhook_url": "https://app.com/webhooks/demographics",
  "analysis_interval": 5000,
  "options": {
    "track_individuals": true,
    "aggregate_statistics": true
  }
}
```

### 2.4 Demographics Analytics
```http
GET /api/v1/demographics/analytics/{user_id}
Authorization: Bearer <token>

Query Parameters:
- date_from: 2024-01-01
- date_to: 2024-01-31
- group_by: day|week|month
```

**Response:**
```json
{
  "success": true,
  "data": {
    "user_analytics": {
      "total_detections": 245,
      "date_range": {
        "from": "2024-01-01",
        "to": "2024-01-31"
      },
      "trends": {
        "age_groups_detected": {
          "children": 15,
          "teens": 32,
          "young_adults": 89,
          "adults": 76,
          "seniors": 33
        },
        "gender_distribution": {
          "male": 128,
          "female": 117
        }
      },
      "accuracy_metrics": {
        "avg_age_confidence": 0.84,
        "avg_gender_confidence": 0.91,
        "high_confidence_rate": 0.78
      }
    }
  }
}
```

---

## 3. Service Implementation

### 3.1 Project Structure

```
age-gender-service/
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI application
│   ├── api/
│   │   ├── __init__.py
│   │   ├── v1/
│   │   │   ├── __init__.py
│   │   │   ├── routes/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── demographics.py    # Age & gender endpoints
│   │   │   │   ├── analytics.py       # Analytics endpoints
│   │   │   │   ├── batch.py            # Batch processing
│   │   │   │   └── health.py           # Health check endpoints
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
│   │   ├── demographics.py
│   │   ├── analytics.py
│   │   └── batch.py
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── requests.py
│   │   ├── responses.py
│   │   └── models.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── demographics/
│   │   │   ├── __init__.py
│   │   │   ├── age_estimator.py       # Age estimation core
│   │   │   ├── gender_classifier.py   # Gender classification
│   │   │   ├── demographics_analyzer.py # Combined analysis
│   │   │   └── face_preprocessor.py    # Face preprocessing
│   │   ├── analytics/
│   │   │   ├── __init__.py
│   │   │   ├── statistics.py          # Statistical analysis
│   │   │   ├── trends.py               # Trend analysis
│   │   │   └── aggregator.py           # Data aggregation
│   │   ├── database/
│   │   │   ├── __init__.py
│   │   │   ├── postgres.py
│   │   │   └── redis.py
│   │   └── external/
│   │       ├── __init__.py
│   │       ├── face_detection.py      # Face detection client
│   │       └── notification.py        # Webhook notifications
│   └── utils/
│       ├── __init__.py
│       ├── image.py
│       ├── validation.py
│       └── metrics.py
├── models/                           # AI model files
│   ├── age/
│   │   ├── dex_imdb_wiki.onnx
│   │   ├── ssr_net_age.onnx
│   │   └── age_config.yaml
│   ├── gender/
│   │   ├── agegender_net.onnx
│   │   ├── fairface_gender.onnx
│   │   └── gender_config.yaml
│   ├── combined/
│   │   ├── age_gender_combined.onnx
│   │   └── combined_config.yaml
│   └── config/
│       ├── model_config.yaml
│       └── demographics_config.yaml
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

#### Age Estimation Service
```python
# app/services/demographics/age_estimator.py
import numpy as np
import cv2
from typing import Dict, Any, List, Optional
import onnxruntime as ort
from ..database.postgres import DatabaseManager
from ..database.redis import CacheManager

class AgeEstimationService:
    def __init__(self):
        self.dex_model = None
        self.ssr_model = None
        self.db_manager = DatabaseManager()
        self.cache_manager = CacheManager()
        self.age_ranges = [
            (0, 2), (4, 6), (8, 12), (15, 20), (25, 32),
            (38, 43), (48, 53), (60, 100)
        ]
        
    async def initialize_models(self):
        """Initialize age estimation models"""
        try:
            # Load DEX model for age estimation
            self.dex_model = ort.InferenceSession(
                "models/age/dex_imdb_wiki.onnx",
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            
            # Load SSR-Net model for precise age regression
            self.ssr_model = ort.InferenceSession(
                "models/age/ssr_net_age.onnx",
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to load age models: {e}")
            
    async def estimate_age(
        self, 
        face_image: np.ndarray,
        options: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Estimate age from face image"""
        if options is None:
            options = {}
            
        try:
            # 1. Preprocess face image
            processed_face = self._preprocess_face(face_image)
            
            # 2. DEX age estimation (age distribution)
            age_distribution = await self._estimate_age_dex(processed_face)
            
            # 3. SSR-Net age regression (precise age)
            precise_age = await self._estimate_age_ssr(processed_face)
            
            # 4. Combine results
            final_age = self._combine_age_estimates(
                age_distribution, precise_age
            )
            
            # 5. Calculate confidence and range
            age_range = self._get_age_range(final_age)
            confidence = self._calculate_age_confidence(
                age_distribution, final_age
            )
            
            result = {
                "estimated_age": int(final_age),
                "age_range": age_range,
                "confidence": confidence,
                "distribution": age_distribution if options.get("return_confidence") else None,
                "precise_age": precise_age
            }
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Age estimation failed: {e}")
            
    async def _estimate_age_dex(
        self, face_image: np.ndarray
    ) -> Dict[str, float]:
        """DEX model age distribution estimation"""
        try:
            # Prepare input
            input_data = self._prepare_dex_input(face_image)
            
            # Run inference
            input_name = self.dex_model.get_inputs()[0].name
            output_names = [o.name for o in self.dex_model.get_outputs()]
            
            results = self.dex_model.run(output_names, {input_name: input_data})
            age_probs = results[0][0]  # Age probabilities for each age group
            
            # Convert to age distribution
            age_distribution = {}
            for i, prob in enumerate(age_probs):
                age_range = self.age_ranges[i]
                range_key = f"{age_range[0]}-{age_range[1]}"
                age_distribution[range_key] = float(prob)
                
            return age_distribution
            
        except Exception as e:
            raise RuntimeError(f"DEX age estimation failed: {e}")
            
    async def _estimate_age_ssr(
        self, face_image: np.ndarray
    ) -> float:
        """SSR-Net precise age regression"""
        try:
            # Prepare input
            input_data = self._prepare_ssr_input(face_image)
            
            # Run inference
            input_name = self.ssr_model.get_inputs()[0].name
            output_names = [o.name for o in self.ssr_model.get_outputs()]
            
            results = self.ssr_model.run(output_names, {input_name: input_data})
            predicted_age = float(results[0][0])
            
            # Clamp age to reasonable range
            predicted_age = max(0, min(100, predicted_age))
            
            return predicted_age
            
        except Exception as e:
            raise RuntimeError(f"SSR age estimation failed: {e}")
            
    def _preprocess_face(self, face_image: np.ndarray) -> np.ndarray:
        """Preprocess face image for age estimation"""
        # Resize to model input size
        face_resized = cv2.resize(face_image, (224, 224))
        
        # Normalize
        face_normalized = face_resized.astype(np.float32) / 255.0
        
        # Convert BGR to RGB if needed
        if len(face_normalized.shape) == 3 and face_normalized.shape[2] == 3:
            face_normalized = cv2.cvtColor(face_normalized, cv2.COLOR_BGR2RGB)
            
        return face_normalized
        
    def _combine_age_estimates(
        self, age_distribution: Dict[str, float], precise_age: float
    ) -> float:
        """Combine DEX and SSR-Net estimates"""
        # Weight: 70% SSR-Net precision, 30% DEX distribution
        distribution_age = self._calculate_expected_age(age_distribution)
        combined_age = 0.7 * precise_age + 0.3 * distribution_age
        return combined_age
        
    def _calculate_expected_age(
        self, age_distribution: Dict[str, float]
    ) -> float:
        """Calculate expected age from distribution"""
        expected_age = 0.0
        for age_range_str, prob in age_distribution.items():
            age_min, age_max = map(int, age_range_str.split('-'))
            age_mid = (age_min + age_max) / 2
            expected_age += age_mid * prob
        return expected_age
        
    def _get_age_range(self, age: float) -> str:
        """Get age range string"""
        for age_min, age_max in self.age_ranges:
            if age_min <= age <= age_max:
                return f"{age_min}-{age_max}"
        return "unknown"
        
    def _calculate_age_confidence(
        self, age_distribution: Dict[str, float], estimated_age: float
    ) -> float:
        """Calculate confidence based on distribution"""
        # Find the range containing estimated age
        age_range = self._get_age_range(estimated_age)
        if age_range in age_distribution:
            # Confidence is the probability of the predicted range
            return age_distribution[age_range]
        return 0.0
```

#### Gender Classification Service
```python
# app/services/demographics/gender_classifier.py
import numpy as np
import cv2
from typing import Dict, Any, List
import onnxruntime as ort

class GenderClassificationService:
    def __init__(self):
        self.agegender_model = None
        self.fairface_model = None
        self.gender_labels = ["male", "female"]
        
    async def initialize_models(self):
        """Initialize gender classification models"""
        try:
            # Load AgeGenderNet model
            self.agegender_model = ort.InferenceSession(
                "models/gender/agegender_net.onnx",
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            
            # Load FairFace model for better diversity
            self.fairface_model = ort.InferenceSession(
                "models/gender/fairface_gender.onnx",
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to load gender models: {e}")
            
    async def classify_gender(
        self, 
        face_image: np.ndarray,
        options: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Classify gender from face image"""
        if options is None:
            options = {}
            
        try:
            # 1. Preprocess face image
            processed_face = self._preprocess_face(face_image)
            
            # 2. AgeGenderNet prediction
            agegender_probs = await self._classify_agegender(processed_face)
            
            # 3. FairFace prediction
            fairface_probs = await self._classify_fairface(processed_face)
            
            # 4. Ensemble prediction
            final_probs = self._ensemble_predictions(
                agegender_probs, fairface_probs
            )
            
            # 5. Get final prediction
            predicted_gender = self.gender_labels[np.argmax(final_probs)]
            confidence = float(np.max(final_probs))
            
            result = {
                "predicted_gender": predicted_gender,
                "confidence": confidence,
                "probabilities": {
                    "male": float(final_probs[0]),
                    "female": float(final_probs[1])
                }
            }
            
            if options.get("return_individual_predictions"):
                result["individual_predictions"] = {
                    "agegender_net": {
                        "male": float(agegender_probs[0]),
                        "female": float(agegender_probs[1])
                    },
                    "fairface": {
                        "male": float(fairface_probs[0]),
                        "female": float(fairface_probs[1])
                    }
                }
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Gender classification failed: {e}")
            
    async def _classify_agegender(
        self, face_image: np.ndarray
    ) -> np.ndarray:
        """AgeGenderNet gender classification"""
        try:
            # Prepare input for AgeGenderNet (224x224)
            input_data = self._prepare_agegender_input(face_image)
            
            # Run inference
            input_name = self.agegender_model.get_inputs()[0].name
            output_names = [o.name for o in self.agegender_model.get_outputs()]
            
            results = self.agegender_model.run(output_names, {input_name: input_data})
            
            # Get gender probabilities (usually second output)
            gender_probs = results[1][0]  # [male_prob, female_prob]
            
            return gender_probs
            
        except Exception as e:
            raise RuntimeError(f"AgeGenderNet classification failed: {e}")
            
    async def _classify_fairface(
        self, face_image: np.ndarray
    ) -> np.ndarray:
        """FairFace gender classification"""
        try:
            # Prepare input for FairFace (224x224)
            input_data = self._prepare_fairface_input(face_image)
            
            # Run inference
            input_name = self.fairface_model.get_inputs()[0].name
            output_names = [o.name for o in self.fairface_model.get_outputs()]
            
            results = self.fairface_model.run(output_names, {input_name: input_data})
            gender_probs = results[0][0]  # Gender probabilities
            
            return gender_probs
            
        except Exception as e:
            raise RuntimeError(f"FairFace classification failed: {e}")
            
    def _ensemble_predictions(
        self, agegender_probs: np.ndarray, fairface_probs: np.ndarray
    ) -> np.ndarray:
        """Ensemble multiple model predictions"""
        # Weighted average: 60% AgeGenderNet, 40% FairFace
        ensemble_probs = 0.6 * agegender_probs + 0.4 * fairface_probs
        return ensemble_probs
        
    def _preprocess_face(self, face_image: np.ndarray) -> np.ndarray:
        """Preprocess face image for gender classification"""
        # Resize to 224x224
        face_resized = cv2.resize(face_image, (224, 224))
        
        # Normalize to [0, 1]
        face_normalized = face_resized.astype(np.float32) / 255.0
        
        # Convert BGR to RGB if needed
        if len(face_normalized.shape) == 3 and face_normalized.shape[2] == 3:
            face_normalized = cv2.cvtColor(face_normalized, cv2.COLOR_BGR2RGB)
            
        return face_normalized
```

#### Demographics Analytics Service
```python
# app/services/analytics/statistics.py
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import numpy as np
from sqlalchemy import select, func, and_
from ..database.postgres import DatabaseManager

class DemographicsAnalyticsService:
    def __init__(self):
        self.db_manager = DatabaseManager()
        
    async def get_user_analytics(
        self,
        user_id: str,
        date_from: datetime,
        date_to: datetime,
        group_by: str = "day"
    ) -> Dict[str, Any]:
        """Get demographics analytics for a user"""
        try:
            async with self.db_manager.get_session() as db:
                # Query demographics detections
                detections = await self._get_detections_in_range(
                    db, user_id, date_from, date_to
                )
                
                # Calculate analytics
                analytics = {
                    "total_detections": len(detections),
                    "date_range": {
                        "from": date_from.isoformat(),
                        "to": date_to.isoformat()
                    },
                    "trends": await self._calculate_trends(detections),
                    "accuracy_metrics": await self._calculate_accuracy_metrics(detections),
                    "temporal_analysis": await self._analyze_temporal_patterns(
                        detections, group_by
                    )
                }
                
                return analytics
                
        except Exception as e:
            raise RuntimeError(f"Analytics calculation failed: {e}")
            
    async def get_global_statistics(
        self,
        date_from: datetime,
        date_to: datetime
    ) -> Dict[str, Any]:
        """Get global demographics statistics"""
        try:
            async with self.db_manager.get_session() as db:
                stats = {
                    "overview": await self._get_global_overview(db, date_from, date_to),
                    "age_distribution": await self._get_age_distribution(db, date_from, date_to),
                    "gender_distribution": await self._get_gender_distribution(db, date_from, date_to),
                    "accuracy_trends": await self._get_accuracy_trends(db, date_from, date_to),
                    "usage_patterns": await self._get_usage_patterns(db, date_from, date_to)
                }
                
                return stats
                
        except Exception as e:
            raise RuntimeError(f"Global statistics calculation failed: {e}")
            
    async def _calculate_trends(
        self, detections: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate demographic trends"""
        age_groups = {
            "children": 0, "teens": 0, "young_adults": 0, 
            "adults": 0, "seniors": 0
        }
        gender_counts = {"male": 0, "female": 0, "unknown": 0}
        
        for detection in detections:
            # Age group classification
            age = detection.get("estimated_age", 0)
            if age < 13:
                age_groups["children"] += 1
            elif age < 20:
                age_groups["teens"] += 1
            elif age < 35:
                age_groups["young_adults"] += 1
            elif age < 60:
                age_groups["adults"] += 1
            else:
                age_groups["seniors"] += 1
                
            # Gender count
            gender = detection.get("predicted_gender", "unknown")
            if gender in gender_counts:
                gender_counts[gender] += 1
            else:
                gender_counts["unknown"] += 1
                
        return {
            "age_groups_detected": age_groups,
            "gender_distribution": gender_counts
        }
        
    async def _calculate_accuracy_metrics(
        self, detections: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate accuracy metrics"""
        if not detections:
            return {"avg_age_confidence": 0.0, "avg_gender_confidence": 0.0}
            
        age_confidences = []
        gender_confidences = []
        
        for detection in detections:
            if "age_confidence" in detection:
                age_confidences.append(detection["age_confidence"])
            if "gender_confidence" in detection:
                gender_confidences.append(detection["gender_confidence"])
                
        metrics = {}
        
        if age_confidences:
            metrics["avg_age_confidence"] = np.mean(age_confidences)
            metrics["age_confidence_std"] = np.std(age_confidences)
            
        if gender_confidences:
            metrics["avg_gender_confidence"] = np.mean(gender_confidences)
            metrics["gender_confidence_std"] = np.std(gender_confidences)
            
        # High confidence rate (>0.8)
        high_conf_count = sum(1 for d in detections 
                             if d.get("age_confidence", 0) > 0.8 and 
                                d.get("gender_confidence", 0) > 0.8)
        metrics["high_confidence_rate"] = high_conf_count / len(detections)
        
        return metrics
```

---

## 4. Database Integration

### 4.1 PostgreSQL Operations
```python
# app/services/database/postgres.py
from sqlalchemy import select, insert, update, delete, func, and_
from typing import Dict, Any, List, Optional
import uuid
from datetime import datetime

class DemographicsDatabase:
    def __init__(self, db_manager):
        self.db_manager = db_manager
        
    async def save_demographics_result(
        self,
        db,
        user_id: str,
        image_hash: str,
        age_data: Dict[str, Any],
        gender_data: Dict[str, Any],
        processing_time_ms: int,
        additional_data: Dict[str, Any] = None
    ) -> str:
        """Save demographics detection result"""
        detection_id = str(uuid.uuid4())
        
        detection_data = {
            "detection_id": detection_id,
            "user_id": user_id,
            "image_hash": image_hash,
            "estimated_age": age_data.get("estimated_age"),
            "age_range": age_data.get("age_range"),
            "age_confidence": age_data.get("confidence"),
            "predicted_gender": gender_data.get("predicted_gender"),
            "gender_confidence": gender_data.get("confidence"),
            "processing_time_ms": processing_time_ms,
            "model_versions": additional_data.get("model_versions") if additional_data else None,
            "created_at": datetime.utcnow(),
            "metadata": additional_data
        }
        
        stmt = insert(DemographicsDetection).values(detection_data)
        await db.execute(stmt)
        await db.commit()
        
        return detection_id
        
    async def get_user_demographics_history(
        self,
        db,
        user_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get user's demographics detection history"""
        stmt = select(DemographicsDetection).where(
            DemographicsDetection.user_id == user_id
        ).order_by(DemographicsDetection.created_at.desc()).limit(limit).offset(offset)
        
        result = await db.execute(stmt)
        detections = result.scalars().all()
        
        return [self._detection_to_dict(detection) for detection in detections]
        
    async def get_demographics_statistics(
        self,
        db,
        date_from: datetime,
        date_to: datetime,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get demographics statistics"""
        base_query = select(DemographicsDetection).where(
            and_(
                DemographicsDetection.created_at >= date_from,
                DemographicsDetection.created_at <= date_to
            )
        )
        
        if user_id:
            base_query = base_query.where(DemographicsDetection.user_id == user_id)
            
        result = await db.execute(base_query)
        detections = result.scalars().all()
        
        # Calculate statistics
        total_detections = len(detections)
        
        if total_detections == 0:
            return {"total_detections": 0}
            
        # Age statistics
        ages = [d.estimated_age for d in detections if d.estimated_age is not None]
        age_stats = {
            "avg_age": sum(ages) / len(ages) if ages else 0,
            "min_age": min(ages) if ages else 0,
            "max_age": max(ages) if ages else 0
        }
        
        # Gender distribution
        gender_counts = {}
        for detection in detections:
            gender = detection.predicted_gender or "unknown"
            gender_counts[gender] = gender_counts.get(gender, 0) + 1
            
        # Confidence statistics
        age_confidences = [d.age_confidence for d in detections if d.age_confidence is not None]
        gender_confidences = [d.gender_confidence for d in detections if d.gender_confidence is not None]
        
        confidence_stats = {
            "avg_age_confidence": sum(age_confidences) / len(age_confidences) if age_confidences else 0,
            "avg_gender_confidence": sum(gender_confidences) / len(gender_confidences) if gender_confidences else 0
        }
        
        return {
            "total_detections": total_detections,
            "age_statistics": age_stats,
            "gender_distribution": gender_counts,
            "confidence_statistics": confidence_stats,
            "date_range": {
                "from": date_from.isoformat(),
                "to": date_to.isoformat()
            }
        }
```

---

## 5. Performance Optimization

### 5.1 Model Optimization
- **ONNX Runtime**: GPU acceleration สำหรับ inference
- **Model Quantization**: INT8 quantization สำหรับ production
- **Batch Processing**: ประมวลผลหลายใบหน้าพร้อมกัน
- **TensorRT**: NVIDIA TensorRT optimization

### 5.2 Caching Strategy
- **Redis Cache**: Cache ผลลัพธ์ของภาพที่เคยประมวลผล
- **Model Cache**: Cache โมเดลที่โหลดแล้วในหน่วยความจำ
- **Statistical Cache**: Cache สถิติที่คำนวณเสร็จแล้ว

### 5.3 Async Processing
```python
# Async batch processing for multiple faces
async def process_batch_demographics(self, images: List[np.ndarray]):
    tasks = []
    for image in images:
        task = asyncio.create_task(self.analyze_demographics(image))
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    return results
```

---

## 6. Monitoring และ Analytics

### 6.1 Performance Metrics
- **Processing Time**: Average time per face analysis
- **Accuracy Metrics**: Age and gender prediction accuracy
- **Throughput**: Faces processed per second
- **Model Performance**: Individual model accuracy

### 6.2 Quality Metrics
- **Age Accuracy**: ±3 years accuracy rate
- **Gender Accuracy**: Binary classification accuracy
- **Confidence Distribution**: Distribution of prediction confidence
- **Error Analysis**: Common failure cases and improvements

Age & Gender Detection Service เป็นบริการที่เสริมข้อมูลประชากรศาสตร์ให้กับระบบ AI โดยสามารถใช้ร่วมกับ Face Detection และ Face Recognition เพื่อให้ข้อมูลที่ครบถ้วนสำหรับการวิเคราะห์และการใช้งานในหน้าต่างๆ ของ Frontend รวมถึงการสร้าง Analytics Dashboard ที่มีประโยชน์
