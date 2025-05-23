import numpy as np
import onnxruntime as ort
import cv2
import logging
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)

class AntiSpoofingDetector:
    def __init__(self, model_path: str = "/app/models/AntiSpoofing_bin_1.5_128.onnx"):
        """Initialize Anti-Spoofing Detector"""
        try:
            providers = ['CPUExecutionProvider']
            if 'CUDAExecutionProvider' in ort.get_available_providers():
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.session = ort.InferenceSession(model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [output.name for output in self.session.get_outputs()]
            input_shape = self.session.get_inputs()[0].shape
            self.input_size = (128, 128)
            self.scale = 1.5
            logger.info(f"Anti-Spoofing Detector loaded successfully")
            logger.info(f"Input shape: {input_shape}")
            logger.info(f"Providers: {self.session.get_providers()}")
        except Exception as e:
            logger.error(f"Failed to load Anti-Spoofing Detector: {e}")
            raise
    def preprocess_face(self, face_image: np.ndarray) -> np.ndarray:
        face_resized = cv2.resize(face_image, self.input_size)
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        face_normalized = face_rgb.astype(np.float32) / 255.0
        face_transposed = face_normalized.transpose(2, 0, 1)
        face_batch = np.expand_dims(face_transposed, axis=0)
        return face_batch
    def detect_spoofing(self, face_image: np.ndarray) -> Dict:
        try:
            input_data = self.preprocess_face(face_image)
            outputs = self.session.run(self.output_names, {self.input_name: input_data})
            if len(outputs) > 0:
                output = outputs[0]
                if output.shape[-1] == 2:
                    real_score = output[0][0]
                    fake_score = output[0][1]
                    is_real = real_score > fake_score
                    confidence = float(max(real_score, fake_score))
                    spoofing_score = float(fake_score)
                else:
                    spoofing_score = float(output[0])
                    is_real = spoofing_score < 0.5
                    confidence = 1 - spoofing_score if is_real else spoofing_score
                return {
                    "is_real": is_real,
                    "confidence": confidence,
                    "spoofing_score": spoofing_score,
                    "label": "REAL" if is_real else "FAKE"
                }
            else:
                raise ValueError("No output from model")
        except Exception as e:
            logger.error(f"Anti-spoofing detection failed: {e}")
            return {
                "is_real": None,
                "confidence": 0.0,
                "spoofing_score": 0.0,
                "label": "ERROR",
                "error": str(e)
            }

class LivenessDetector:
    def __init__(self):
        self.min_face_size = (50, 50)
        self.blink_threshold = 0.2
        self.mouth_threshold = 0.3
    def detect_blink(self, face_landmarks) -> bool:
        return False  # Placeholder
    def detect_mouth_movement(self, face_landmarks) -> bool:
        return False  # Placeholder
    def check_face_size(self, face_bbox: list, image_shape: tuple) -> bool:
        face_width = face_bbox[2] - face_bbox[0]
        face_height = face_bbox[3] - face_bbox[1]
        image_height, image_width = image_shape[:2]
        face_area = face_width * face_height
        image_area = image_width * image_height
        return (face_area / image_area) > 0.1
    def analyze_texture(self, face_image: np.ndarray) -> Dict:
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        is_sharp = variance > 100
        return {
            "variance": float(variance),
            "is_sharp": is_sharp
        }
