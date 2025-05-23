import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

class GenderAgeDetector:
    def __init__(self, model_path: str = "/app/models/genderage.onnx"):
        """
        Initialize Gender & Age Detection model
        """
        try:
            providers = ['CPUExecutionProvider']
            if 'CUDAExecutionProvider' in ort.get_available_providers():
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.session = ort.InferenceSession(model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [output.name for output in self.session.get_outputs()]
            input_shape = self.session.get_inputs()[0].shape
            self.input_height = input_shape[2] if len(input_shape) > 2 else 224
            self.input_width = input_shape[3] if len(input_shape) > 3 else 224
            logger.info(f"Gender & Age Detector loaded successfully")
            logger.info(f"Input shape: {self.input_width}x{self.input_height}")
            logger.info(f"Providers: {self.session.get_providers()}")
            logger.info(f"Output names: {self.output_names}")
        except Exception as e:
            logger.error(f"Failed to load Gender & Age Detector: {e}")
            raise

    def preprocess_face(self, face_image: np.ndarray):
        img_resized = cv2.resize(face_image, (self.input_width, self.input_height))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_normalized = img_rgb.astype(np.float32) / 255.0
        img_transposed = img_normalized.transpose(2, 0, 1)
        img_batch = np.expand_dims(img_transposed, axis=0)
        return img_batch

    def postprocess(self, predictions: np.ndarray) -> Tuple[int, int]:
        # Gender: argmax of first 2 elements
        gender = int(np.argmax(predictions[:2]))
        # Age: third element * 100 (denormalize)
        age = int(np.round(predictions[2] * 100))
        age = max(0, min(100, age))
        return gender, age

    def detect_gender_age(self, face_image: np.ndarray):
        try:
            input_data = self.preprocess_face(face_image)
            outputs = self.session.run(self.output_names, {self.input_name: input_data})
            predictions = outputs[0][0]  # Remove batch dimension
            logger.info(f"Raw predictions: {predictions}")
            logger.info(f"Predictions shape: {predictions.shape}")
            gender_pred, age_pred = self.postprocess(predictions)
            gender_logits = predictions[:2]
            gender_probs = self._softmax(gender_logits)
            gender_confidence = float(np.max(gender_probs))
            gender_text = "Male" if gender_pred == 1 else "Female"
            logger.info(f"Parsed - Gender: {gender_text} ({gender_confidence:.3f}), Age: {age_pred}")
            return {
                "gender": {
                    "prediction": gender_text,
                    "value": int(gender_pred),
                    "confidence": gender_confidence,
                    "probabilities": {
                        "Female": float(gender_probs[0]),
                        "Male": float(gender_probs[1])
                    }
                },
                "age": {
                    "prediction": int(age_pred),
                    "value": float(age_pred),
                    "range": {
                        "min": max(0, int(age_pred - 5)),
                        "max": min(100, int(age_pred + 5))
                    },
                    "raw_output": float(predictions[2])
                }
            }
        except Exception as e:
            logger.error(f"Gender & Age detection failed: {e}")
            return {
                "gender": {
                    "prediction": "Unknown",
                    "value": -1,
                    "confidence": 0.0
                },
                "age": {
                    "prediction": 0,
                    "value": 0.0,
                    "range": {"min": 0, "max": 0}
                },
                "error": str(e)
            }
    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def analyze_multiple_faces(self, face_images: list):
        results = []
        for i, face_image in enumerate(face_images):
            result = self.detect_gender_age(face_image)
            result["face_id"] = i
            results.append(result)
        return results

    def get_demographics_summary(self, results: list):
        if not results:
            return {
                "total_faces": 0,
                "gender_distribution": {"Male": 0, "Female": 0, "Unknown": 0},
                "age_statistics": {"mean": 0, "min": 0, "max": 0}
            }
        gender_count = {"Male": 0, "Female": 0, "Unknown": 0}
        ages = []
        for result in results:
            gender = result.get("gender", {}).get("prediction", "Unknown")
            gender_count[gender] = gender_count.get(gender, 0) + 1
            age = result.get("age", {}).get("value", 0)
            if age > 0:
                ages.append(age)
        age_stats = {
            "mean": round(np.mean(ages)) if ages else 0,
            "min": round(min(ages)) if ages else 0,
            "max": round(max(ages)) if ages else 0,
            "median": round(np.median(ages)) if ages else 0
        }
        return {
            "total_faces": len(results),
            "gender_distribution": gender_count,
            "age_statistics": age_stats,
            "age_groups": self._categorize_ages(ages)
        }
    def _categorize_ages(self, ages: list):
        if not ages:
            return {}
        categories = {
            "Child (0-12)": 0,
            "Teen (13-19)": 0,
            "Young Adult (20-29)": 0,
            "Adult (30-49)": 0,
            "Middle-aged (50-64)": 0,
            "Senior (65+)": 0
        }
        for age in ages:
            if age <= 12:
                categories["Child (0-12)"] += 1
            elif age <= 19:
                categories["Teen (13-19)"] += 1
            elif age <= 29:
                categories["Young Adult (20-29)"] += 1
            elif age <= 49:
                categories["Adult (30-49)"] += 1
            elif age <= 64:
                categories["Middle-aged (50-64)"] += 1
            else:
                categories["Senior (65+)"] += 1
        return categories
