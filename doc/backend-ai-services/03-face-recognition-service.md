# Face Recognition Service - API และ Implementation

## 1. Service Overview

### หน้าที่หลัก
- **Register Face**: ลงทะเบียนใบหน้าใหม่ในระบบ
- **Verify Face**: ตรวจสอบใบหน้ากับข้อมูลที่มีอยู่
- **Identify Face**: ระบุตัวตนจากใบหน้า
- **Face Embedding Management**: จัดการ face embeddings

### Integration Flow ใน Frontend
```
Login Page → Face Recognition → Deepfake Check → Antispoofing → Authentication
Create Post → Auto Face Tagging → Face Detection → Face Recognition
Profile → Face Management → Register/Update Face Data
Video Call → Real-time Verification → Face Recognition
```

## 2. API Endpoints

### 2.1 Register New Face
```http
POST /api/v1/face-recognition/register
Content-Type: application/json
Authorization: Bearer {token}

{
  "user_id": "uuid",
  "image": "base64_encoded_image",
  "face_group": "primary", // optional: "primary", "alternate", "group_name"
  "metadata": {
    "source": "profile_setup",
    "device_info": {...},
    "quality_check": true
  }
}
```

**Response (201 Created):**
```json
{
  "success": true,
  "data": {
    "face_id": "face_abc123",
    "user_id": "user_uuid",
    "registration_status": "success",
    "face_quality_score": 0.9456,
    "embedding_version": "arcface_v1.2",
    "face_attributes": {
      "estimated_age": 28.5,
      "estimated_gender": "female",
      "face_landmarks": {...}
    }
  },
  "metadata": {
    "request_id": "req_uuid",
    "processing_time": "234ms",
    "model_version": "arcface_v1.2.3",
    "timestamp": "2024-01-20T10:30:00Z"
  }
}
```

### 2.2 Verify Face
```http
POST /api/v1/face-recognition/verify
Content-Type: application/json
Authorization: Bearer {token}

{
  "user_id": "uuid",
  "image": "base64_encoded_image",
  "face_id": "face_abc123", // optional: specific face to verify against
  "verification_level": "standard", // "basic", "standard", "strict"
  "metadata": {
    "source": "login_attempt",
    "session_id": "session_uuid"
  }
}
```

**Response (200 OK):**
```json
{
  "success": true,
  "data": {
    "verification_result": "success",
    "is_match": true,
    "confidence_score": 0.9823,
    "similarity_score": 0.9756,
    "matched_face_id": "face_abc123",
    "verification_level": "standard",
    "risk_assessment": {
      "risk_level": "low",
      "factors": []
    }
  },
  "metadata": {
    "request_id": "req_uuid",
    "processing_time": "187ms",
    "model_version": "arcface_v1.2.3",
    "timestamp": "2024-01-20T10:30:00Z"
  }
}
```

### 2.3 Identify Face
```http
POST /api/v1/face-recognition/identify
Content-Type: application/json
Authorization: Bearer {token}

{
  "image": "base64_encoded_image",
  "search_scope": {
    "user_groups": ["friends", "public"], // optional: limit search scope
    "max_candidates": 10,
    "min_confidence": 0.8
  },
  "metadata": {
    "source": "auto_tagging",
    "context": "post_upload"
  }
}
```

**Response (200 OK):**
```json
{
  "success": true,
  "data": {
    "identification_result": "found",
    "candidates": [
      {
        "user_id": "user_uuid1",
        "face_id": "face_abc123",
        "confidence_score": 0.9823,
        "similarity_score": 0.9756,
        "user_info": {
          "display_name": "John Doe",
          "profile_image": "url",
          "privacy_level": "friends"
        }
      },
      {
        "user_id": "user_uuid2", 
        "face_id": "face_def456",
        "confidence_score": 0.8967,
        "similarity_score": 0.8892,
        "user_info": {
          "display_name": "Jane Smith",
          "profile_image": "url",
          "privacy_level": "public"
        }
      }
    ],
    "total_candidates": 2,
    "search_stats": {
      "faces_searched": 15678,
      "search_time_ms": 45
    }
  },
  "metadata": {
    "request_id": "req_uuid",
    "processing_time": "298ms",
    "model_version": "arcface_v1.2.3",
    "timestamp": "2024-01-20T10:30:00Z"
  }
}
```

### 2.4 Batch Face Processing
```http
POST /api/v1/face-recognition/batch-identify
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
  "search_scope": {
    "max_candidates": 5,
    "min_confidence": 0.85
  },
  "processing_options": {
    "async": true,
    "callback_url": "https://app.com/webhook/face-recognition"
  }
}
```

### 2.5 Face Management APIs

#### Get User Faces
```http
GET /api/v1/face-recognition/users/{user_id}/faces
Authorization: Bearer {token}
```

#### Update Face Data
```http
PUT /api/v1/face-recognition/faces/{face_id}
Content-Type: application/json
Authorization: Bearer {token}

{
  "face_group": "alternate",
  "metadata": {...},
  "privacy_settings": {
    "searchable": true,
    "taggable": "friends_only"
  }
}
```

#### Delete Face
```http
DELETE /api/v1/face-recognition/faces/{face_id}
Authorization: Bearer {token}
```

## 3. Service Implementation

### 3.1 Project Structure
```
face-recognition-service/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application
│   ├── api/
│   │   ├── __init__.py
│   │   ├── v1/
│   │   │   ├── __init__.py
│   │   │   ├── routes/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── recognition.py
│   │   │   │   ├── management.py
│   │   │   │   └── batch.py
│   │   │   └── deps.py         # Dependencies
│   │   └── middleware/
│   │       ├── __init__.py
│   │       ├── auth.py
│   │       ├── rate_limit.py
│   │       └── logging.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── security.py
│   │   └── database.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── face.py
│   │   ├── user.py
│   │   └── session.py
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── requests.py
│   │   ├── responses.py
│   │   └── models.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── face_recognition/
│   │   │   ├── __init__.py
│   │   │   ├── extractor.py     # Face embedding extraction
│   │   │   ├── matcher.py       # Face matching logic
│   │   │   ├── aligner.py       # Face alignment
│   │   │   └── quality.py       # Face quality assessment
│   │   ├── database/
│   │   │   ├── __init__.py
│   │   │   ├── postgres.py      # PostgreSQL operations
│   │   │   └── milvus.py        # Vector operations
│   │   └── external/
│   │       ├── __init__.py
│   │       ├── deepfake.py      # Deepfake service client
│   │       └── antispoofing.py  # Antispoofing service client
│   └── utils/
│       ├── __init__.py
│       ├── image.py
│       ├── crypto.py
│       └── validation.py
├── models/                     # AI model files
│   ├── arcface/
│   │   ├── arcface_r100.onnx
│   │   └── config.yaml
│   ├── retinaface/
│   │   ├── retinaface_r50.onnx
│   │   └── config.yaml
│   └── quality/
│       ├── face_quality.onnx
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

#### Face Recognition Main Service
```python
# app/services/face_recognition/face_service.py
import uuid
import numpy as np
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from milvus import connections, Collection

from app.services.face_recognition.extractor import FaceEmbeddingExtractor
from app.services.face_recognition.matcher import FaceMatcher
from app.services.face_recognition.quality import FaceQualityAssessment
from app.services.database.postgres import FaceDataRepository
from app.services.database.milvus import VectorRepository
from app.core.config import settings
from app.schemas.requests import (
    RegisterFaceRequest, VerifyFaceRequest, IdentifyFaceRequest
)
from app.schemas.responses import (
    RegisterFaceResponse, VerifyFaceResponse, IdentifyFaceResponse
)

class FaceRecognitionService:
    def __init__(self):
        self.extractor = FaceEmbeddingExtractor(
            model_path=settings.ARCFACE_MODEL_PATH
        )
        self.matcher = FaceMatcher(
            distance_threshold=settings.FACE_MATCH_THRESHOLD
        )
        self.quality_assessor = FaceQualityAssessment(
            model_path=settings.QUALITY_MODEL_PATH
        )
        self.face_repo = FaceDataRepository()
        self.vector_repo = VectorRepository()
        
    async def register_face(
        self, 
        request: RegisterFaceRequest,
        db: Session
    ) -> RegisterFaceResponse:
        """ลงทะเบียนใบหน้าใหม่"""
        
        # 1. Validate and preprocess image
        image = self._decode_base64_image(request.image)
        if image is None:
            raise ValueError("Invalid image format")
            
        # 2. Detect and align face
        faces = await self._detect_and_align_faces(image)
        if not faces:
            raise ValueError("No face detected in image")
            
        if len(faces) > 1:
            raise ValueError("Multiple faces detected. Please use single face image")
            
        face_image = faces[0]
        
        # 3. Quality assessment
        quality_score = self.quality_assessor.assess(face_image)
        if quality_score < settings.MIN_FACE_QUALITY:
            raise ValueError(f"Face quality too low: {quality_score}")
            
        # 4. Extract embedding
        embedding = self.extractor.extract(face_image)
        
        # 5. Check for duplicates
        existing_faces = await self._search_similar_faces(
            embedding, request.user_id, threshold=0.95
        )
        if existing_faces:
            raise ValueError("Face already registered")
            
        # 6. Generate face ID
        face_id = f"face_{uuid.uuid4().hex[:12]}"
        
        # 7. Store in databases
        # PostgreSQL
        face_data = await self.face_repo.create_face_data(
            db=db,
            face_id=face_id,
            user_id=request.user_id,
            quality_score=quality_score,
            embedding_version=self.extractor.model_version,
            metadata=request.metadata
        )
        
        # Milvus
        await self.vector_repo.insert_embedding(
            collection_name=settings.FACE_EMBEDDINGS_COLLECTION,
            face_id=face_id,
            user_id=request.user_id,
            embedding=embedding.tolist()
        )
        
        return RegisterFaceResponse(
            face_id=face_id,
            user_id=request.user_id,
            registration_status="success",
            face_quality_score=quality_score,
            embedding_version=self.extractor.model_version
        )
        
    async def verify_face(
        self,
        request: VerifyFaceRequest,
        db: Session
    ) -> VerifyFaceResponse:
        """ตรวจสอบใบหน้า"""
        
        # 1. Preprocess input image
        image = self._decode_base64_image(request.image)
        faces = await self._detect_and_align_faces(image)
        
        if not faces:
            return VerifyFaceResponse(
                verification_result="no_face_detected",
                is_match=False,
                confidence_score=0.0
            )
            
        # 2. Extract embedding from input
        input_embedding = self.extractor.extract(faces[0])
        
        # 3. Get user's registered faces
        if request.face_id:
            # Verify against specific face
            registered_faces = [await self.face_repo.get_face_by_id(db, request.face_id)]
        else:
            # Verify against all user faces
            registered_faces = await self.face_repo.get_user_faces(db, request.user_id)
            
        if not registered_faces:
            return VerifyFaceResponse(
                verification_result="no_registered_faces",
                is_match=False,
                confidence_score=0.0
            )
            
        # 4. Find best match
        best_match = None
        best_score = 0.0
        
        for face_data in registered_faces:
            # Get embedding from Milvus
            stored_embedding = await self.vector_repo.get_embedding(
                collection_name=settings.FACE_EMBEDDINGS_COLLECTION,
                face_id=face_data.face_id
            )
            
            # Calculate similarity
            similarity = self.matcher.calculate_similarity(
                input_embedding, np.array(stored_embedding)
            )
            
            if similarity > best_score:
                best_score = similarity
                best_match = face_data
                
        # 5. Determine verification result
        is_match = best_score >= settings.VERIFICATION_THRESHOLD
        confidence_score = best_score
        
        # Update verification count
        if is_match and best_match:
            await self.face_repo.increment_verification_count(db, best_match.id)
            
        return VerifyFaceResponse(
            verification_result="success" if is_match else "no_match",
            is_match=is_match,
            confidence_score=confidence_score,
            similarity_score=best_score,
            matched_face_id=best_match.face_id if best_match else None
        )
        
    async def identify_face(
        self,
        request: IdentifyFaceRequest,
        db: Session
    ) -> IdentifyFaceResponse:
        """ระบุตัวตนจากใบหน้า"""
        
        # 1. Extract embedding from input
        image = self._decode_base64_image(request.image)
        faces = await self._detect_and_align_faces(image)
        
        if not faces:
            return IdentifyFaceResponse(
                identification_result="no_face_detected",
                candidates=[]
            )
            
        input_embedding = self.extractor.extract(faces[0])
        
        # 2. Search in vector database
        search_results = await self.vector_repo.search_similar(
            collection_name=settings.FACE_EMBEDDINGS_COLLECTION,
            query_embedding=input_embedding.tolist(),
            top_k=request.search_scope.max_candidates,
            score_threshold=request.search_scope.min_confidence
        )
        
        # 3. Get user information for candidates
        candidates = []
        for result in search_results:
            face_data = await self.face_repo.get_face_by_id(db, result.id)
            user_info = await self._get_user_info(face_data.user_id)
            
            # Check privacy settings
            if await self._check_search_permission(
                user_info, request.search_scope.user_groups
            ):
                candidates.append({
                    "user_id": face_data.user_id,
                    "face_id": face_data.face_id,
                    "confidence_score": result.score,
                    "similarity_score": result.distance,
                    "user_info": user_info
                })
                
        return IdentifyFaceResponse(
            identification_result="found" if candidates else "not_found",
            candidates=candidates,
            total_candidates=len(candidates)
        )
        
    # Helper methods
    async def _detect_and_align_faces(self, image: np.ndarray) -> List[np.ndarray]:
        """ตรวจจับและปรับแนวใบหน้า"""
        # Call face detection service
        detection_client = FaceDetectionClient()
        faces = await detection_client.detect_faces(image)
        return faces
        
    async def _search_similar_faces(
        self, 
        embedding: np.ndarray, 
        user_id: str, 
        threshold: float = 0.95
    ) -> List[Dict]:
        """ค้นหาใบหน้าที่คล้ายกัน"""
        results = await self.vector_repo.search_similar(
            collection_name=settings.FACE_EMBEDDINGS_COLLECTION,
            query_embedding=embedding.tolist(),
            top_k=10,
            score_threshold=threshold,
            filter_expr=f"user_id == '{user_id}'"
        )
        return results
        
    def _decode_base64_image(self, base64_str: str) -> Optional[np.ndarray]:
        """แปลง base64 เป็น image array"""
        try:
            import base64
            import cv2
            
            image_data = base64.b64decode(base64_str)
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return image
        except Exception as e:
            return None
```

#### Face Embedding Extractor
```python
# app/services/face_recognition/extractor.py
import numpy as np
import onnxruntime as ort
import cv2
from typing import Optional

class FaceEmbeddingExtractor:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_size = (112, 112)  # ArcFace standard input size
        self.model_version = "arcface_v1.2.3"
        
    def extract(self, face_image: np.ndarray) -> np.ndarray:
        """สกัด face embedding จากภาพใบหน้า"""
        
        # 1. Preprocess image
        processed_image = self._preprocess_image(face_image)
        
        # 2. Run inference
        embedding = self.session.run(
            [self.output_name], 
            {self.input_name: processed_image}
        )[0]
        
        # 3. Normalize embedding
        embedding = self._normalize_embedding(embedding)
        
        return embedding.flatten()
        
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """เตรียมภาพสำหรับโมเดล"""
        
        # Resize to model input size
        resized = cv2.resize(image, self.input_size)
        
        # Normalize pixel values
        normalized = resized.astype(np.float32) / 255.0
        
        # Standardize (ImageNet stats)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        standardized = (normalized - mean) / std
        
        # Convert to NCHW format
        transposed = standardized.transpose(2, 0, 1)
        
        # Add batch dimension
        batched = np.expand_dims(transposed, axis=0)
        
        return batched
        
    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """Normalize embedding ให้เป็น unit vector"""
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm
```

### 3.3 Configuration
```python
# app/core/config.py
from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    # Service settings
    SERVICE_NAME: str = "face-recognition-service"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # API settings
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Database settings
    POSTGRES_URL: str
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530
    REDIS_URL: str = "redis://localhost:6379"
    
    # Model paths
    ARCFACE_MODEL_PATH: str = "./models/arcface/arcface_r100.onnx"
    QUALITY_MODEL_PATH: str = "./models/quality/face_quality.onnx"
    
    # Face recognition settings
    FACE_MATCH_THRESHOLD: float = 0.8
    VERIFICATION_THRESHOLD: float = 0.85
    MIN_FACE_QUALITY: float = 0.7
    
    # Milvus collections
    FACE_EMBEDDINGS_COLLECTION: str = "face_embeddings_512d"
    
    # External services
    FACE_DETECTION_SERVICE_URL: str = "http://face-detection-service:8000"
    DEEPFAKE_DETECTION_SERVICE_URL: str = "http://deepfake-detection-service:8000"
    ANTISPOOFING_SERVICE_URL: str = "http://antispoofing-service:8000"
    
    # Rate limiting
    RATE_LIMIT_PER_MINUTE: int = 60
    RATE_LIMIT_PER_HOUR: int = 1000
    
    # Processing limits
    MAX_IMAGE_SIZE_MB: int = 10
    MAX_BATCH_SIZE: int = 100
    ASYNC_PROCESSING_TIMEOUT: int = 300  # 5 minutes
    
    class Config:
        env_file = ".env"

settings = Settings()
```

### 3.4 Docker Configuration
```dockerfile
# docker/Dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Download and setup AI models
RUN mkdir -p models && \
    python scripts/download_models.py

# Create non-root user
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 4. Testing Strategy

### 4.1 Unit Tests
```python
# tests/test_services/test_face_recognition.py
import pytest
import numpy as np
from app.services.face_recognition.face_service import FaceRecognitionService

class TestFaceRecognitionService:
    
    @pytest.fixture
    def service(self):
        return FaceRecognitionService()
        
    @pytest.fixture
    def sample_face_image(self):
        # Load test image
        return np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        
    def test_extract_embedding(self, service, sample_face_image):
        """Test face embedding extraction"""
        embedding = service.extractor.extract(sample_face_image)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (512,)  # ArcFace output dimension
        assert np.allclose(np.linalg.norm(embedding), 1.0, atol=1e-6)  # Unit vector
        
    def test_face_matching(self, service):
        """Test face matching logic"""
        embedding1 = np.random.random(512)
        embedding1 = embedding1 / np.linalg.norm(embedding1)
        
        embedding2 = embedding1 + np.random.random(512) * 0.1
        embedding2 = embedding2 / np.linalg.norm(embedding2)
        
        similarity = service.matcher.calculate_similarity(embedding1, embedding2)
        assert 0.0 <= similarity <= 1.0
        assert similarity > 0.8  # Should be similar
```

### 4.2 Integration Tests
```python
# tests/test_api/test_recognition_endpoints.py
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

class TestRecognitionAPI:
    
    def test_register_face(self):
        """Test face registration endpoint"""
        response = client.post(
            "/api/v1/face-recognition/register",
            json={
                "user_id": "test_user_123",
                "image": "base64_test_image_data",
                "metadata": {"source": "test"}
            },
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 201
        data = response.json()
        assert data["success"] is True
        assert "face_id" in data["data"]
        
    def test_verify_face(self):
        """Test face verification endpoint"""
        response = client.post(
            "/api/v1/face-recognition/verify", 
            json={
                "user_id": "test_user_123",
                "image": "base64_test_image_data"
            },
            headers={"Authorization": "Bearer test_token"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "verification_result" in data["data"]
```

## 5. Performance Optimization

### 5.1 Caching Strategy
```python
# app/services/cache/face_cache.py
import redis
import json
import pickle
import numpy as np
from typing import Optional, List

class FaceCache:
    def __init__(self, redis_url: str):
        self.redis = redis.from_url(redis_url)
        self.embedding_ttl = 3600  # 1 hour
        self.result_ttl = 300      # 5 minutes
        
    async def cache_user_embeddings(self, user_id: str, embeddings: List[dict]):
        """Cache user's face embeddings"""
        key = f"face_embeddings:{user_id}"
        
        # Serialize embeddings
        serialized = []
        for emb in embeddings:
            emb_copy = emb.copy()
            emb_copy['embedding'] = pickle.dumps(emb['embedding'])
            serialized.append(emb_copy)
            
        await self.redis.setex(
            key, 
            self.embedding_ttl, 
            json.dumps(serialized)
        )
        
    async def get_user_embeddings(self, user_id: str) -> Optional[List[dict]]:
        """Get cached user embeddings"""
        key = f"face_embeddings:{user_id}"
        cached = await self.redis.get(key)
        
        if not cached:
            return None
            
        # Deserialize embeddings
        serialized = json.loads(cached)
        embeddings = []
        for emb in serialized:
            emb['embedding'] = pickle.loads(emb['embedding'])
            embeddings.append(emb)
            
        return embeddings
```

### 5.2 Async Processing
```python
# app/services/async_processing/face_jobs.py
from celery import Celery
from app.services.face_recognition.face_service import FaceRecognitionService

celery_app = Celery("face-recognition-worker")

@celery_app.task
def process_batch_identification(images: List[dict], search_scope: dict):
    """Process batch face identification asynchronously"""
    service = FaceRecognitionService()
    results = []
    
    for image_data in images:
        try:
            result = service.identify_face_sync(image_data, search_scope)
            results.append({
                "image_id": image_data["image_id"],
                "status": "success",
                "result": result
            })
        except Exception as e:
            results.append({
                "image_id": image_data["image_id"], 
                "status": "error",
                "error": str(e)
            })
            
    return results
```

Face Recognition Service เป็นส่วนหลักของระบบ AI ที่จะถูกใช้งานมากที่สุด โดยต้องมีประสิทธิภาพสูงและความแม่นยำในการทำงาน รวมถึงการรองรับ real-time processing สำหรับการใช้งานหลากหลายรูปแบบใน Frontend
