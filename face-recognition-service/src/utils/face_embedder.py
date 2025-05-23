import numpy as np
import onnxruntime as ort
import cv2
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class FaceEmbedder:
    def __init__(self, model_configs: Dict[str, Dict]):
        """
        Initialize multiple face recognition models
        model_configs = {
            "adaface": {"path": "models/adaface_ir101.onnx", "input_size": (112, 112)},
            "arcface": {"path": "models/arcface_r100.onnx", "input_size": (112, 112)},
            "facenet": {"path": "models/facenet_vggface2.onnx", "input_size": (160, 160)}
        }
        """
        self.models = {}
        self.configs = model_configs
        
        # Setup providers
        providers = ['CPUExecutionProvider']
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        for name, config in model_configs.items():
            try:
                # เพิ่ม providers parameter
                session = ort.InferenceSession(
                    config["path"], 
                    providers=providers  # แก้ไขตรงนี้
                )
                self.models[name] = {
                    "session": session,
                    "input_name": session.get_inputs()[0].name,
                    "input_size": config["input_size"]
                }
                logger.info(f"Loaded {name} model successfully")
            except Exception as e:
                logger.error(f"Failed to load {name} model: {e}")
    
    def preprocess_face(self, face_image: np.ndarray, input_size: tuple) -> np.ndarray:
        """Preprocess face for embedding extraction"""
        # Resize
        face_resized = cv2.resize(face_image, input_size)
        
        # Normalize to [-1, 1] or [0, 1] based on model
        face_normalized = face_resized.astype(np.float32) / 255.0
        
        # Add batch dimension and transpose if needed
        face_input = np.expand_dims(face_normalized, axis=0)
        face_input = face_input.transpose(0, 3, 1, 2)  # NHWC to NCHW
        
        return face_input
    
    def extract_embedding(self, face_image: np.ndarray, model_name: str = "adaface") -> Optional[np.ndarray]:
        """Extract face embedding using specified model"""
        if model_name not in self.models:
            logger.error(f"Model {model_name} not loaded")
            return None
            
        try:
            model = self.models[model_name]
            
            # Preprocess
            face_input = self.preprocess_face(face_image, model["input_size"])
            
            # Extract embedding
            outputs = model["session"].run(None, {model["input_name"]: face_input})
            embedding = outputs[0][0]  # Remove batch dimension
            
            # Normalize embedding
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to extract embedding: {e}")
            return None
    
    def extract_ensemble_embedding(self, face_image: np.ndarray, weights: Dict[str, float] = None) -> Optional[np.ndarray]:
        """Extract ensemble embedding from multiple models"""
        if weights is None:
            weights = {"adaface": 0.4, "arcface": 0.3, "facenet": 0.3}
        
        embeddings = []
        used_weights = []
        
        for model_name, weight in weights.items():
            if model_name in self.models:
                embedding = self.extract_embedding(face_image, model_name)
                if embedding is not None:
                    embeddings.append(embedding * weight)
                    used_weights.append(weight)
        
        if not embeddings:
            return None
            
        # Combine and normalize
        ensemble_embedding = np.sum(embeddings, axis=0)
        ensemble_embedding = ensemble_embedding / np.sum(used_weights)
        ensemble_embedding = ensemble_embedding / np.linalg.norm(ensemble_embedding)
        
        return ensemble_embedding
