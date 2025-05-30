#!/usr/bin/env python3
"""
Face Recognition Service - Production Implementation with Lazy Loading
ระบบจดจำใบหน้าโดยใช้โมเดล FaceNet (lazy loading เพื่อป้องกัน memory crash)
"""

import os
import uuid
import time
import json
import asyncio
import logging
from typing import List, Dict, Optional, Tuple, Any, Union
from datetime import datetime, timezone
from pathlib import Path
import base64
from io import BytesIO
import hashlib
import pickle

import cv2
import numpy as np
from PIL import Image
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, File, UploadFile, BackgroundTasks, Depends, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
from loguru import logger
import psutil
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import aiohttp
from enum import Enum

# Import dynamic detection manager
from dynamic_detection import FaceDetectionManager, DetectionMethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger.add("logs/face_recognition_{time}.log", rotation="1 day", retention="30 days")

class FaceRecognitionConfig:
    """Configuration for Face Recognition Service"""
    
    # Model paths
    MODELS_DIR = Path("/app/models")
    ADAFACE_MODEL = MODELS_DIR / "adaface_ir101.onnx"
    FACENET_MODEL = MODELS_DIR / "facenet_vggface2.onnx"
    ARCFACE_MODEL = MODELS_DIR / "arcface_r100.onnx"
    
    # Database for face embeddings (simplified - in production use real DB)
    EMBEDDINGS_DB = Path("/app/data/embeddings.pkl")
    
    # Face detection settings
    FACE_SIZE = (112, 112)  # Standard size for face recognition models
    MIN_FACE_SIZE = 50
    CONFIDENCE_THRESHOLD = 0.7
    SIMILARITY_THRESHOLD = 0.65
    
    # Model weights for ensemble (all models available)
    MODEL_WEIGHTS = {
        "adaface": 0.4,
        "facenet": 0.3,
        "arcface": 0.3
    }

class FaceData(BaseModel):
    """Face data model"""
    face_id: str
    user_id: str
    embeddings: Dict[str, List[float]]
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime

class RegisterRequest(BaseModel):
    """Face registration request"""
    user_id: str
    image: str  # base64 encoded
    face_group: Optional[str] = "primary"
    metadata: Optional[Dict[str, Any]] = {}

class VerifyRequest(BaseModel):
    """Face verification request"""
    user_id: str
    image: str  # base64 encoded
    face_id: Optional[str] = None
    verification_level: Optional[str] = "standard"
    metadata: Optional[Dict[str, Any]] = {}

class IdentifyRequest(BaseModel):
    """Face identification request"""
    image: str  # base64 encoded
    max_results: Optional[int] = 5
    min_confidence: Optional[float] = 0.7
    metadata: Optional[Dict[str, Any]] = {}

class FaceRecognitionEngine:
    """Core face recognition engine with lazy loading and dynamic detection"""
    
    def __init__(self, config: FaceRecognitionConfig):
        self.config = config
        self.models = {}
        self.model_files = {}
        self.face_cascade = None
        self.embeddings_db = {}
        
        # Initialize dynamic detection manager
        self.detection_manager = FaceDetectionManager()
        
    async def initialize(self):
        """Initialize models and database"""
        try:
            logger.info("🚀 Initializing Face Recognition Engine...")
            
            # Initialize face detection
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            # Setup model configuration (lazy loading)
            await self._setup_models()
            
            # Load embeddings database
            await self._load_embeddings_db()
            
            logger.info("✅ Face Recognition Engine initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Face Recognition Engine: {e}")
            raise
    
    async def _setup_models(self):
        """Setup model configuration for lazy loading"""
        try:
            logger.info("🔧 Setting up model configuration...")
              # Model files configuration (lazy loading)
            self.model_files = {
                "adaface": self.config.ADAFACE_MODEL,
                "facenet": self.config.FACENET_MODEL,
                "arcface": self.config.ARCFACE_MODEL
            }
            
            logger.info("📦 Model configuration ready - using lazy loading")
            logger.info(f"💡 Available models: {list(self.model_files.keys())}")
                
        except Exception as e:
            logger.error(f"❌ Failed to configure models: {e}")
            raise
    
    def _load_model_lazy(self, model_name: str):
        """Lazy load a specific model when first needed"""
        if model_name in self.models:
            return self.models[model_name]
            
        if model_name not in self.model_files:
            raise ValueError(f"Unknown model: {model_name}")
            
        model_path = self.model_files[model_name]
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        try:
            logger.info(f"📦 Lazy loading {model_name} model...")
            providers = ['CPUExecutionProvider']
            if ort.get_device() == 'GPU':
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                
            session = ort.InferenceSession(str(model_path), providers=providers)
            self.models[model_name] = session
            logger.info(f"✅ {model_name} model loaded successfully")
            return session
        except Exception as e:
            logger.error(f"❌ Failed to load {model_name} model: {e}")
            raise
    
    async def _load_embeddings_db(self):
        """Load embeddings database"""
        try:
            if self.config.EMBEDDINGS_DB.exists():
                with open(self.config.EMBEDDINGS_DB, 'rb') as f:
                    self.embeddings_db = pickle.load(f)
                logger.info(f"📊 Loaded {len(self.embeddings_db)} face records from database")
            else:
                self.embeddings_db = {}
                # Create directory if not exists
                self.config.EMBEDDINGS_DB.parent.mkdir(parents=True, exist_ok=True)
                logger.info("📊 Initialized empty embeddings database")
                
        except Exception as e:
            logger.error(f"❌ Failed to load embeddings database: {e}")
            self.embeddings_db = {}
    
    async def _save_embeddings_db(self):
        """Save embeddings database"""
        try:
            with open(self.config.EMBEDDINGS_DB, 'wb') as f:
                pickle.dump(self.embeddings_db, f)
            logger.info("💾 Embeddings database saved successfully")        except Exception as e:
            logger.error(f"❌ Failed to save embeddings database: {e}")
    
    def _preprocess_image(self, image: np.ndarray, target_size: Tuple[int, int] = (112, 112)) -> np.ndarray:
        """Preprocess image for face recognition models"""
        logger.info(f"🔍 Preprocessing image: input_shape={image.shape}, target_size={target_size}")
        
        # Add detailed validation
        logger.info(f"🔬 Detailed validation:")
        logger.info(f"   Image type: {type(image)}")
        logger.info(f"   Image dtype: {image.dtype}")
        logger.info(f"   Image shape: {image.shape}")
        logger.info(f"   Image size: {image.size}")
        logger.info(f"   Image flags: {image.flags}")
        logger.info(f"   Target size type: {type(target_size)}")
        logger.info(f"   Target size value: {target_size}")
        logger.info(f"   Target size[0]: {target_size[0]} (type: {type(target_size[0])})")
        logger.info(f"   Target size[1]: {target_size[1]} (type: {type(target_size[1])})")
        
        # Validate input
        if image.size == 0:
            raise ValueError("Input image is empty")
        if target_size[0] <= 0 or target_size[1] <= 0:
            raise ValueError(f"Invalid target size: {target_size}")
        
        # Resize to target size - this is where the error occurs
        logger.info(f"🎯 About to call cv2.resize with image.shape={image.shape}, target_size={target_size}")
        face = cv2.resize(image, target_size)
        
        # Normalize pixel values
        face = face.astype(np.float32) / 255.0
        
        # Convert BGR to RGB if needed
        if len(face.shape) == 3 and face.shape[2] == 3:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        
        # Transpose for ONNX (HWC to CHW)
        face = np.transpose(face, (2, 0, 1))  # HWC to CHW
        
        # Add batch dimension
        face = np.expand_dims(face, axis=0)   # Add batch dimension
        
        logger.info(f"✅ Preprocessed image shape: {face.shape}")
        return face

    def _detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in image using Haar Cascade (legacy method)"""
        try:
            # Convert to grayscale for detection
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(self.config.MIN_FACE_SIZE, self.config.MIN_FACE_SIZE)
            )
            
            return [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]
            
        except Exception as e:
            logger.error(f"❌ Face detection failed: {e}")
            return []
    
    async def _detect_faces_dynamic(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using dynamic detection manager with fallback"""
        try:
            # Use dynamic detection manager
            faces = await self.detection_manager.detect_faces_dynamic(image, engine=self)
            
            if not faces:
                logger.warning("Dynamic detection failed, falling back to Haar Cascade")
                faces = self._detect_faces(image)
            
            logger.info(f"Detected {len(faces)} faces using {self.detection_manager.get_current_method().value}")
            return faces
            
        except Exception as e:
            logger.error(f"❌ Dynamic face detection failed: {e}")
            # Fallback to legacy detection
            return self._detect_faces(image)
    
    def _extract_embeddings(self, face_image: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract face embeddings using available models with lazy loading"""
        embeddings = {}
        
        # Use lazy loading to load models only when needed
        for model_name in self.model_files.keys():
            try:
                # Lazy load the model if not already loaded
                session = self._load_model_lazy(model_name)
                
                # Get input name and shape
                input_name = session.get_inputs()[0].name
                
                # Determine target size based on model
                if 'facenet' in model_name.lower():
                    target_size = (160, 160)
                else:  # adaface, arcface
                    target_size = (112, 112)
                  # Preprocess image for this specific model
                logger.info(f"🔍 Pre-preprocessing validation for {model_name}:")
                logger.info(f"   Input face_image shape: {face_image.shape}")
                logger.info(f"   Input face_image size: {face_image.size}")
                logger.info(f"   Input face_image dtype: {face_image.dtype}")
                logger.info(f"   Target size: {target_size}")
                
                if face_image.size == 0:
                    logger.error(f"❌ Face image is empty before preprocessing for {model_name}")
                    continue
                
                processed_face = self._preprocess_image(face_image, target_size)
                
                # Run inference
                outputs = session.run(None, {input_name: processed_face})
                embedding = outputs[0][0]  # Get first output, first batch
                
                # Normalize embedding
                embedding = normalize(embedding.reshape(1, -1))[0]
                embeddings[model_name] = embedding
                
                logger.debug(f"✅ Extracted {model_name} embedding: shape {embedding.shape}")
                
            except Exception as e:
                logger.error(f"❌ Failed to extract {model_name} embedding: {e}")
                continue
        
        return embeddings
    
    def _calculate_similarity(self, embedding1: Dict[str, np.ndarray], embedding2: Dict[str, np.ndarray]) -> float:
        """Calculate similarity between two face embeddings using ensemble approach"""
        similarities = []
        weights = []
        
        for model_name in embedding1.keys():
            if model_name in embedding2:                # Calculate cosine similarity
                sim = cosine_similarity(
                    embedding1[model_name].reshape(1, -1),
                    embedding2[model_name].reshape(1, -1)
                )[0, 0]
                
                similarities.append(sim)
                weights.append(self.config.MODEL_WEIGHTS.get(model_name, 1.0))
        
        if not similarities:
            return 0.0
        
        # Weighted average
        total_weight = sum(weights)
        weighted_sim = sum(s * w for s, w in zip(similarities, weights)) / total_weight
        
        return float(weighted_sim)

    async def register_face(self, user_id: str, image_data: str, face_group: str = "primary", metadata: Dict = None) -> Dict:
        """Register a new face"""
        try:
            # Decode image
            image_bytes = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_bytes))
            image_np = np.array(image)
            
            # Detect faces using dynamic detection
            faces = await self._detect_faces_dynamic(image_np)
            if not faces:
                raise HTTPException(status_code=400, detail="No face detected in image")
            
            # Use largest face
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = largest_face
              # Add validation and bounds checking
            logger.info(f"🔍 Face coordinates: x={x}, y={y}, w={w}, h={h}")
            logger.info(f"🖼️ Image shape: {image_np.shape}")
            
            # Ensure coordinates are valid
            height, width = image_np.shape[:2]
            logger.info(f"📏 Original bounds: x={x}, y={y}, w={w}, h={h}")
            logger.info(f"📐 Image dimensions: width={width}, height={height}")
            
            # Apply bounds checking
            x_orig, y_orig, w_orig, h_orig = x, y, w, h
            x = max(0, min(x, width - 1))
            y = max(0, min(y, height - 1))
            w = max(1, min(w, width - x))
            h = max(1, min(h, height - y))
            
            logger.info(f"🔧 Corrected bounds: x={x}, y={y}, w={w}, h={h}")
            logger.info(f"🎯 Extraction region: [{y}:{y+h}, {x}:{x+w}]")
            
            # Extract face with safety checks
            if w <= 0 or h <= 0:
                raise HTTPException(status_code=400, detail=f"Invalid face dimensions: {w}x{h}")
            
            # Log extraction details
            logger.info(f"🔍 About to extract: image_np[{y}:{y+h}, {x}:{x+w}]")
            
            face_image = image_np[y:y+h, x:x+w]
              # Validate extracted face
            logger.info(f"✅ Extracted face shape: {face_image.shape}")
            logger.info(f"🔢 Face image size (total elements): {face_image.size}")
            if face_image.shape[0] == 0 or face_image.shape[1] == 0:
                raise HTTPException(status_code=400, detail=f"Extracted face has zero dimensions: {face_image.shape}")
            
            if face_image.size == 0:
                raise HTTPException(status_code=400, detail="Extracted face is empty")
            
            # Additional detailed validation before embedding extraction
            logger.info(f"🔬 Pre-embedding validation:")
            logger.info(f"   Face image dtype: {face_image.dtype}")
            logger.info(f"   Face image shape: {face_image.shape}")
            logger.info(f"   Face image size: {face_image.size}")
            logger.info(f"   Face image min/max: {face_image.min()}/{face_image.max()}")
            logger.info(f"   Face image is_contiguous: {face_image.flags.contiguous}")
            
            # Validate image data integrity
            if not isinstance(face_image, np.ndarray):
                raise HTTPException(status_code=400, detail=f"Face image is not numpy array: {type(face_image)}")
            
            if len(face_image.shape) != 3:
                raise HTTPException(status_code=400, detail=f"Face image must be 3D (H,W,C): got {face_image.shape}")
            
            if face_image.shape[2] != 3:
                raise HTTPException(status_code=400, detail=f"Face image must have 3 channels: got {face_image.shape[2]}")
            
            # Extract embeddings
            embeddings = self._extract_embeddings(face_image)
            if not embeddings:
                raise HTTPException(status_code=500, detail="Failed to extract face embeddings")
            
            # Create face record
            face_id = str(uuid.uuid4())
            face_data = FaceData(
                face_id=face_id,
                user_id=user_id,
                embeddings={k: v.tolist() for k, v in embeddings.items()},
                metadata=metadata or {},
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc)
            )
            
            # Save to database
            self.embeddings_db[face_id] = face_data.dict()
            await self._save_embeddings_db()
            
            logger.info(f"✅ Registered face {face_id} for user {user_id}")
            
            return {
                "face_id": face_id,
                "user_id": user_id,
                "registration_status": "success",
                "face_quality_score": min(len(embeddings) / len(self.model_files), 1.0),
                "embedding_version": "lazy_v1.0",
                "face_attributes": {
                    "face_size": [w, h],
                    "face_position": [x, y],
                    "models_used": list(embeddings.keys())            }
            }
            
        except Exception as e:
            logger.error(f"❌ Face registration failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def verify_face(self, user_id: str, image_data: str, face_id: str = None, verification_level: str = "standard") -> Dict:
        """Verify face against registered faces"""
        try:
            # Decode image
            image_bytes = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_bytes))
            image_np = np.array(image)
            
            # Detect faces using dynamic detection
            faces = await self._detect_faces_dynamic(image_np)
            if not faces:
                raise HTTPException(status_code=400, detail="No face detected in image")
            
            # Use largest face
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = largest_face
            face_image = image_np[y:y+h, x:x+w]
            
            # Extract embeddings
            query_embeddings = self._extract_embeddings(face_image)
            if not query_embeddings:
                raise HTTPException(status_code=500, detail="Failed to extract face embeddings")
            
            # Find faces to compare against
            target_faces = []
            if face_id:
                # Verify against specific face
                if face_id in self.embeddings_db:
                    target_faces = [self.embeddings_db[face_id]]
            else:
                # Verify against all faces of the user
                target_faces = [face for face in self.embeddings_db.values() 
                              if face['user_id'] == user_id]
            
            if not target_faces:
                return {
                    "verification_result": "no_match",
                    "is_match": False,
                    "confidence_score": 0.0,
                    "similarity_score": 0.0,
                    "matched_face_id": None,
                    "verification_level": verification_level,
                    "message": "No registered faces found for user"
                }
            
            # Calculate similarities
            best_match = None
            best_similarity = 0.0
            
            for face_record in target_faces:
                # Convert embeddings back to numpy arrays
                stored_embeddings = {
                    k: np.array(v) for k, v in face_record['embeddings'].items()
                }
                
                similarity = self._calculate_similarity(query_embeddings, stored_embeddings)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = face_record
            
            # Determine verification threshold based on level
            thresholds = {
                "basic": 0.5,
                "standard": self.config.SIMILARITY_THRESHOLD,
                "strict": 0.8
            }
            threshold = thresholds.get(verification_level, self.config.SIMILARITY_THRESHOLD)
            
            is_match = best_similarity >= threshold
            
            logger.info(f"🔍 Face verification: similarity={best_similarity:.4f}, threshold={threshold}, match={is_match}")
            
            return {
                "verification_result": "success" if is_match else "no_match",
                "is_match": is_match,
                "confidence_score": best_similarity,
                "similarity_score": best_similarity,
                "matched_face_id": best_match['face_id'] if best_match else None,
                "verification_level": verification_level,
                "risk_assessment": {
                    "risk_level": "low" if is_match else "medium",
                    "factors": []
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Face verification failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def identify_face(self, image_data: str, max_results: int = 5, min_confidence: float = 0.7) -> Dict:
        """Identify face from all registered faces"""
        try:
            # Decode image
            image_bytes = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_bytes))
            image_np = np.array(image)
            
            # Detect faces
            faces = self._detect_faces(image_np)
            if not faces:
                raise HTTPException(status_code=400, detail="No face detected in image")
            
            # Use largest face
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = largest_face
            face_image = image_np[y:y+h, x:x+w]
            
            # Extract embeddings
            query_embeddings = self._extract_embeddings(face_image)
            if not query_embeddings:
                raise HTTPException(status_code=500, detail="Failed to extract face embeddings")
            
            # Calculate similarities with all faces
            matches = []
            
            for face_record in self.embeddings_db.values():
                # Convert embeddings back to numpy arrays
                stored_embeddings = {
                    k: np.array(v) for k, v in face_record['embeddings'].items()
                }
                
                similarity = self._calculate_similarity(query_embeddings, stored_embeddings)
                
                if similarity >= min_confidence:
                    matches.append({
                        "face_id": face_record['face_id'],
                        "user_id": face_record['user_id'],
                        "similarity_score": similarity,
                        "confidence_score": similarity,
                        "metadata": face_record.get('metadata', {})
                    })
            
            # Sort by similarity and limit results
            matches.sort(key=lambda x: x['similarity_score'], reverse=True)
            matches = matches[:max_results]
            
            logger.info(f"🎯 Face identification: found {len(matches)} matches")
            
            return {
                "identification_result": "success" if matches else "no_match",
                "matches_found": len(matches),
                "matches": matches,
                "face_attributes": {
                    "face_size": [w, h],
                    "face_position": [x, y],
                    "models_used": list(query_embeddings.keys())
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Face identification failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

# Initialize FastAPI app
app = FastAPI(
    title="Face Recognition Service",
    description="Production Face Recognition API with Lazy Loading",
    version="1.0.1"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize config and engine
config = FaceRecognitionConfig()
engine = FaceRecognitionEngine(config)

@app.on_event("startup")
async def startup_event():
    """Initialize the face recognition engine"""
    await engine.initialize()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "face-recognition",
        "models_loaded": len(engine.models),
        "available_models": list(engine.model_files.keys()),
        "faces_registered": len(engine.embeddings_db),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.post("/register")
async def register_face(request: RegisterRequest):
    """Register a new face"""
    result = await engine.register_face(
        user_id=request.user_id,
        image_data=request.image,
        face_group=request.face_group,
        metadata=request.metadata
    )
    
    return {
        "success": True,
        "data": result,
        "metadata": {
            "request_id": str(uuid.uuid4()),
            "processing_time": "0ms",
            "model_version": "lazy_v1.0",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    }

@app.post("/verify")
async def verify_face(request: VerifyRequest):
    """Verify a face against registered faces"""
    result = await engine.verify_face(
        user_id=request.user_id,
        image_data=request.image,
        face_id=request.face_id,
        verification_level=request.verification_level
    )
    
    return {
        "success": True,
        "data": result,
        "metadata": {
            "request_id": str(uuid.uuid4()),
            "processing_time": "0ms",
            "model_version": "lazy_v1.0",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    }

@app.post("/identify")
async def identify_face(request: IdentifyRequest):
    """Identify a face from all registered faces"""
    result = await engine.identify_face(
        image_data=request.image,
        max_results=request.max_results,
        min_confidence=request.min_confidence
    )
    
    return {
        "success": True,
        "data": result,
        "metadata": {
            "request_id": str(uuid.uuid4()),
            "processing_time": "0ms",
            "model_version": "lazy_v1.0",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    }

@app.get("/stats")
async def get_stats():
    """Get service statistics"""
    return {
        "total_faces_registered": len(engine.embeddings_db),
        "models_loaded": list(engine.models.keys()),
        "models_available": list(engine.model_files.keys()),
        "model_weights": config.MODEL_WEIGHTS,
        "similarity_threshold": config.SIMILARITY_THRESHOLD,
        "system_info": {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent
        }
    }

# Dynamic Detection Management Endpoints
@app.get("/detection/methods")
async def get_detection_methods():
    """Get available face detection methods and their status"""
    try:
        health_status = await engine.detection_manager.health_check_all_methods()
        performance_stats = engine.detection_manager.get_performance_stats()
        current_method = engine.detection_manager.get_current_method()
        
        return {
            "success": True,
            "current_method": current_method.value,
            "available_methods": {
                method.value: {
                    "healthy": health_status.get(method.value, False),
                    "stats": performance_stats.get(method.value, {}),
                    "config": {
                        "enabled": engine.detection_manager.detection_configs[method].enabled,
                        "confidence_threshold": engine.detection_manager.detection_configs[method].confidence_threshold,
                        "priority": engine.detection_manager.detection_configs[method].priority
                    }
                }
                for method in DetectionMethod
            }
        }
    except Exception as e:
        logger.error(f"Failed to get detection methods: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detection/switch")
async def switch_detection_method(request: dict):
    """Switch to a specific detection method"""
    try:
        method_name = request.get("method")
        if not method_name:
            raise HTTPException(status_code=400, detail="Method name is required")
        
        # Convert string to enum
        try:
            method = DetectionMethod(method_name)
        except ValueError:
            available_methods = [m.value for m in DetectionMethod]
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid method. Available: {available_methods}"
            )
        
        # Switch method
        engine.detection_manager.set_detection_method(method)
        
        return {
            "success": True,
            "message": f"Switched to {method.value}",
            "current_method": method.value
        }
        
    except Exception as e:
        logger.error(f"Failed to switch detection method: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detection/configure")
async def configure_detection_method(request: dict):
    """Configure a specific detection method"""
    try:
        method_name = request.get("method")
        config_updates = request.get("config", {})
        
        if not method_name:
            raise HTTPException(status_code=400, detail="Method name is required")
        
        try:
            method = DetectionMethod(method_name)
        except ValueError:
            available_methods = [m.value for m in DetectionMethod]
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid method. Available: {available_methods}"
            )
        
        # Update configuration
        engine.detection_manager.configure_method(method, **config_updates)
        
        return {
            "success": True,
            "message": f"Updated configuration for {method.value}",
            "updated_config": config_updates
        }
        
    except Exception as e:
        logger.error(f"Failed to configure detection method: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/detection/health")
async def detection_health_check():
    """Check health of all detection services"""
    try:
        health_status = await engine.detection_manager.check_all_methods_health()
        
        return {
            "success": True,
            "health_status": health_status,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to check detection health: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detection/test")
async def test_detection_method(request: dict):
    """Test face detection with a specific method using uploaded image"""
    try:
        method_name = request.get("method")
        image_data = request.get("image")  # base64 encoded
        
        if not method_name or not image_data:
            raise HTTPException(status_code=400, detail="Method and image are required")
        
        try:
            method = DetectionMethod(method_name)
        except ValueError:
            available_methods = [m.value for m in DetectionMethod]
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid method. Available: {available_methods}"
            )
        
        # Decode image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        image_np = np.array(image)
        
        # Test detection with specific method
        start_time = time.time()
        faces = await engine.detection_manager._detect_with_method(method, image_np, engine)
        detection_time = (time.time() - start_time) * 1000
        
        return {
            "success": True,
            "method": method.value,
            "faces_detected": len(faces),
            "faces": [
                {
                    "bbox": {"x": x, "y": y, "width": w, "height": h},
                    "area": w * h
                }
                for x, y, w, h in faces
            ],
            "detection_time_ms": round(detection_time, 2),
            "image_size": {"width": image_np.shape[1], "height": image_np.shape[0]}
        }
        
    except Exception as e:
        logger.error(f"Failed to test detection method: {e}")
        raise HTTPException(status_code=500, detail=str(e))
