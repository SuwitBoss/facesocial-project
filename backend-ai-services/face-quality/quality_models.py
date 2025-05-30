'''
Enhanced Face Quality Models with ONNX Support
'''

import numpy as np
import onnxruntime as ort
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import cv2
import logging
import os # Make sure os is imported for getenv in AdvancedQualityAssessor

logger = logging.getLogger(__name__)

@dataclass
class QualityMetrics:
    """Comprehensive quality metrics"""
    overall_score: float
    sharpness: float
    brightness: float
    contrast: float
    face_size_score: float
    pose_quality: float
    symmetry_score: float
    skin_quality: float
    expression_quality: float
    occlusion_score: float
    blur_score: float
    noise_level: float
    illumination_uniformity: float
    
@dataclass
class FacePose:
    """Face pose angles"""
    yaw: float  # Left-right rotation
    pitch: float  # Up-down rotation  
    roll: float  # Tilt rotation
    frontal_score: float  # How frontal the face is (0-1)

class FaceQualityONNX:
    """ONNX-based face quality assessment model"""
    
    def __init__(self, model_path: str, device: str = "gpu"):
        self.model_path = model_path
        self.device = device
        self.session = None
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize ONNX model with optimized settings"""
        try:
            # Session options for optimization
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.enable_mem_pattern = True
            sess_options.enable_cpu_mem_arena = True
            
            # Providers based on device
            providers = []
            if self.device == "gpu" and 'CUDAExecutionProvider' in ort.get_available_providers():
                providers.append(('CUDAExecutionProvider', {
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': 200 * 1024 * 1024,  # 200MB
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                }))
            providers.append('CPUExecutionProvider')
            
            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=sess_options,
                providers=providers
            )
            
            # Get input/output info
            self.input_name = self.session.get_inputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            self.output_names = [o.name for o in self.session.get_outputs()]
            
            logger.info(f"Face quality model loaded: {self.model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load quality model: {e}")
            # Fallback to rule-based assessment
            self.session = None
            
    def assess_quality(self, face_image: np.ndarray) -> QualityMetrics:
        """Assess face quality using ONNX model + rule-based metrics"""
        
        if self.session is not None:
            # Use ONNX model
            return self._assess_with_model(face_image)
        # else:
            # Fallback to pure rule-based
            # return self._assess_rule_based(face_image)
        # Fallback to pure rule-based - corrected based on user prompt structure
        return self._calculate_all_metrics(face_image) # Assuming this is the intended fallback
            
    def _assess_with_model(self, face_image: np.ndarray) -> QualityMetrics:
        """Assessment using ONNX model"""
        try:
            # Preprocess for model
            input_data = self._preprocess_for_model(face_image)
            
            # Run inference
            outputs = self.session.run(self.output_names, {self.input_name: input_data})
            
            # Parse model outputs
            quality_vector = outputs[0][0]  # Assuming [1, N] output
            
            # Combine with rule-based metrics
            # rule_metrics = self._calculate_rule_based_metrics(face_image)
            # Corrected based on user prompt structure - AdvancedQualityAssessor handles this logic
            adv_assessor = AdvancedQualityAssessor(enable_ml_model=False) # Create a temporary rule-based assessor
            rule_metrics_obj = adv_assessor._calculate_all_metrics(face_image)

            return QualityMetrics(
                overall_score=float(quality_vector[0]),
                sharpness=rule_metrics_obj.sharpness,
                brightness=rule_metrics_obj.brightness,
                contrast=rule_metrics_obj.contrast,
                face_size_score=rule_metrics_obj.face_size_score,
                pose_quality=float(quality_vector[1]) if len(quality_vector) > 1 else rule_metrics_obj.pose_quality,
                symmetry_score=rule_metrics_obj.symmetry_score,
                skin_quality=float(quality_vector[2]) if len(quality_vector) > 2 else rule_metrics_obj.skin_quality,
                expression_quality=float(quality_vector[3]) if len(quality_vector) > 3 else rule_metrics_obj.expression_quality,
                occlusion_score=rule_metrics_obj.occlusion_score,
                blur_score=rule_metrics_obj.blur_score,
                noise_level=rule_metrics_obj.noise_level,
                illumination_uniformity=rule_metrics_obj.illumination_uniformity
            )
            
        except Exception as e:
            logger.error(f"Model inference failed: {e}")
            # return self._assess_rule_based(face_image)
            # Corrected based on user prompt structure
            adv_assessor = AdvancedQualityAssessor(enable_ml_model=False)
            return adv_assessor._calculate_all_metrics(face_image)
            
    def _preprocess_for_model(self, face_image: np.ndarray) -> np.ndarray:
        """Preprocess image for ONNX model"""
        # Resize to model input size (e.g., 112x112)
        target_size = (self.input_shape[2], self.input_shape[3])
        resized = cv2.resize(face_image, target_size)
        
        # Normalize
        normalized = resized.astype(np.float32) / 255.0
        
        # Transpose to NCHW format
        transposed = np.transpose(normalized, (2, 0, 1))
        
        # Add batch dimension
        batch = np.expand_dims(transposed, axis=0)
        
        return batch
    
    # Added based on user prompt structure for FaceQualityONNX to have a fallback
    def _calculate_all_metrics(self, face_image: np.ndarray) -> QualityMetrics:
        """Calculate all quality metrics using computer vision (used as fallback)"""
        adv_assessor = AdvancedQualityAssessor(enable_ml_model=False) # Avoid recursion
        return adv_assessor._calculate_all_metrics(face_image)

class AdvancedQualityAssessor:
    """Advanced quality assessment with multiple techniques"""
    
    def __init__(self, enable_ml_model: bool = True):
        self.enable_ml_model = enable_ml_model
        self.ml_model = None
        
        if enable_ml_model:
            try:
                # Ensure model path is correct, assuming it's relative to app.py or an absolute path is set via ENV
                model_path = os.getenv('MODEL_PATH', '/app/models/face_quality_v2.onnx')
                self.ml_model = FaceQualityONNX(model_path)
            except Exception as e:
                logger.warning(f"ML model not available: {e}")
                self.ml_model = None # Ensure it's None if init fails
                
    def comprehensive_assessment(self, face_image: np.ndarray) -> Dict[str, Any]:
        """Perform comprehensive quality assessment"""
        
        # Basic image checks
        if face_image is None or face_image.size == 0:
            return self._empty_result()
            
        # Get quality metrics
        if self.ml_model and self.ml_model.session: # Check if session is initialized
            metrics = self.ml_model.assess_quality(face_image)
        else:
            metrics = self._calculate_all_metrics(face_image)
            
        # Analyze pose
        pose = self._estimate_face_pose(face_image)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(metrics, pose)
        
        # Calculate category
        category = self._determine_category(metrics.overall_score)
        
        return {
            "overall_score": metrics.overall_score,
            "category": category,
            "metrics": {
                "sharpness": metrics.sharpness,
                "brightness": metrics.brightness,
                "contrast": metrics.contrast,
                "face_size": metrics.face_size_score,
                "pose_quality": metrics.pose_quality,
                "symmetry": metrics.symmetry_score,
                "skin_quality": metrics.skin_quality,
                "expression": metrics.expression_quality,
                "occlusion": metrics.occlusion_score,
                "blur": metrics.blur_score,
                "noise": metrics.noise_level,
                "illumination": metrics.illumination_uniformity
            },
            "pose": {
                "yaw": pose.yaw,
                "pitch": pose.pitch,
                "roll": pose.roll,
                "frontal_score": pose.frontal_score
            },
            "recommendations": recommendations,
            "usable_for": self._determine_use_cases(metrics)
        }
        
    def _calculate_all_metrics(self, face_image: np.ndarray) -> QualityMetrics:
        """Calculate all quality metrics using computer vision"""
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        
        # Basic metrics
        sharpness = self._calculate_sharpness_advanced(gray)
        brightness = self._calculate_brightness_advanced(gray)
        contrast = self._calculate_contrast_advanced(gray)
        
        # Face-specific metrics
        face_size_score = self._calculate_face_size_score(face_image)
        pose_quality = self._estimate_pose_quality(face_image) # Uses _estimate_face_pose
        symmetry_score = self._calculate_symmetry(gray)
        
        # Quality metrics
        skin_quality = self._assess_skin_quality(face_image)
        expression_quality = self._assess_expression_quality(gray)
        occlusion_score = self._detect_occlusions_advanced(face_image)
        
        # Image quality metrics
        blur_score = self._calculate_motion_blur(gray) # Uses _calculate_sharpness_advanced
        noise_level = self._calculate_noise_level_advanced(gray)
        illumination_uniformity = self._calculate_illumination_uniformity(gray)
        
        # Calculate overall score
        overall = self._calculate_weighted_score({
            'sharpness': (sharpness, 0.20),
            'brightness': (brightness, 0.15),
            'contrast': (contrast, 0.10),
            'face_size': (face_size_score, 0.10),
            'pose_quality': (pose_quality, 0.15),
            'symmetry': (symmetry_score, 0.05),
            'skin_quality': (skin_quality, 0.05),
            'occlusion': (1.0 - occlusion_score, 0.10), # Inverted: higher occlusion = lower score contribution
            'blur': (1.0 - blur_score, 0.05), # Inverted: higher blur = lower score contribution
            'illumination': (illumination_uniformity, 0.05)
        })
        
        return QualityMetrics(
            overall_score=overall,
            sharpness=sharpness,
            brightness=brightness,
            contrast=contrast,
            face_size_score=face_size_score,
            pose_quality=pose_quality,
            symmetry_score=symmetry_score,
            skin_quality=skin_quality,
            expression_quality=expression_quality,
            occlusion_score=occlusion_score,
            blur_score=blur_score,
            noise_level=noise_level,
            illumination_uniformity=illumination_uniformity
        )
        
    def _calculate_sharpness_advanced(self, gray_image: np.ndarray) -> float:
        """Advanced sharpness calculation using multiple methods"""
        if gray_image is None or gray_image.size == 0: return 0.0
        
        # Method 1: Laplacian variance
        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
        laplacian_var = laplacian.var()
        laplacian_score = min(laplacian_var / 500.0, 1.0)
        
        # Method 2: Gradient magnitude
        grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
        gradient_score = min(gradient_mag.mean() / 50.0, 1.0)
        
        # Method 3: High-frequency content
        fft = np.fft.fft2(gray_image)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)
        
        # Calculate high-frequency energy
        h, w = gray_image.shape
        center_h, center_w = h // 2, w // 2
        radius = min(h, w) // 4
        
        y, x = np.ogrid[:h, :w]
        mask = (x - center_w)**2 + (y - center_h)**2 > radius**2
        high_freq_energy = np.sum(magnitude[mask])
        total_energy = np.sum(magnitude)
        
        freq_score = high_freq_energy / total_energy if total_energy > 0 else 0.0
        
        # Combine scores
        sharpness = (laplacian_score * 0.4 + gradient_score * 0.4 + freq_score * 0.2)
        
        return float(min(max(sharpness, 0.0), 1.0))
        
    def _calculate_symmetry(self, gray_image: np.ndarray) -> float:
        """Calculate facial symmetry"""
        if gray_image is None or gray_image.size == 0: return 0.0
        h, w = gray_image.shape
        
        # Split image into left and right halves
        mid = w // 2
        if mid == 0: return 0.0 # Image too narrow
        left_half = gray_image[:, :mid]
        right_half = gray_image[:, mid:]
        
        # Flip right half
        right_flipped = cv2.flip(right_half, 1)
        
        # Ensure same size
        min_width = min(left_half.shape[1], right_flipped.shape[1])
        if min_width == 0: return 0.0
        left_half = left_half[:, :min_width]
        right_flipped = right_flipped[:, :min_width]
        
        # Calculate similarity
        if left_half.shape == right_flipped.shape and left_half.size > 0:
            # Method 1: Structural Similarity
            diff = cv2.absdiff(left_half, right_flipped)
            similarity = 1.0 - (diff.mean() / 255.0)
            
            # Method 2: Correlation
            try:
                correlation = cv2.matchTemplate(left_half, right_flipped, cv2.TM_CCOEFF_NORMED)
                corr_score = correlation[0][0] if correlation.size > 0 else 0.0
            except cv2.error: # Happens if one image is too small for template matching
                corr_score = 0.0
            
            symmetry = (similarity * 0.6 + corr_score * 0.4)
        else:
            symmetry = 0.5
            
        return float(min(max(symmetry, 0.0), 1.0))
        
    def _detect_occlusions_advanced(self, face_image: np.ndarray) -> float:
        """Advanced occlusion detection"""
        if face_image is None or face_image.size == 0: return 1.0 # Max occlusion if no image

        # Initialize cascade classifiers if not already done (can be slow, consider class members)
        # For simplicity here, loading them each time. In production, load once.
        eye_cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_eye.xml')
        mouth_cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_smile.xml')

        if not os.path.exists(eye_cascade_path) or not os.path.exists(mouth_cascade_path):
            logger.warning("Haarcascade files not found, skipping occlusion detection.")
            return 0.3 # Default moderate occlusion if cascades are missing
            
        eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
        mouth_cascade = cv2.CascadeClassifier(mouth_cascade_path)
        
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        if h == 0 or w == 0: return 1.0
        
        # Define regions of interest
        upper_region = gray[:h//2, :]  # Eyes region
        lower_region = gray[h//2:, :]  # Mouth region
        
        eye_occlusion = 0.5 # Default if no eyes found
        if upper_region.size > 0:
            eyes = eye_cascade.detectMultiScale(upper_region, 1.1, 5, minSize=(max(1,w//10), max(1,h//10)))
            eye_occlusion = 0.0 if len(eyes) >= 2 else (0.25 if len(eyes) == 1 else 0.5)
        
        mouth_occlusion = 0.3 # Default if no mouth found
        if lower_region.size > 0:
            mouths = mouth_cascade.detectMultiScale(lower_region, 1.1, 5, minSize=(max(1,w//8), max(1,h//8)))
            mouth_occlusion = 0.0 if len(mouths) > 0 else 0.3
        
        # Check for sunglasses/masks using color analysis
        if upper_region.size > 0:
            eye_region_brightness = upper_region.mean() / 255.0
            if eye_region_brightness < 0.3: # Dark regions in eye area might indicate sunglasses
                eye_occlusion = max(eye_occlusion, 0.7)
            
        # Combine occlusion scores
        total_occlusion = (eye_occlusion * 0.6 + mouth_occlusion * 0.4)
        
        return float(min(max(total_occlusion, 0.0), 1.0))
        
    def _estimate_face_pose(self, face_image: np.ndarray) -> FacePose:
        """Estimate face pose angles"""
        if face_image is None or face_image.size == 0: 
            return FacePose(yaw=0.0, pitch=0.0, roll=0.0, frontal_score=0.0)

        h, w = face_image.shape[:2]
        if h == 0 or w == 0: 
            return FacePose(yaw=0.0, pitch=0.0, roll=0.0, frontal_score=0.0)
            
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        
        # Estimate yaw (left-right)
        mid_w = w // 2
        if mid_w == 0: yaw = 0.0
        else:
            left_half = gray[:, :mid_w]
            right_half = gray[:, mid_w:]
            left_energy = left_half.mean() if left_half.size > 0 else 0
            right_energy = right_half.mean() if right_half.size > 0 else 0
            energy_diff = (left_energy - right_energy) / max(left_energy, right_energy, 1)
            yaw = energy_diff * 30  # Convert to approximate degrees
        
        # Estimate pitch (up-down)
        mid_h = h // 2
        if mid_h == 0: pitch = 0.0
        else:
            upper_half = gray[:mid_h, :]
            lower_half = gray[mid_h:, :]
            upper_energy = upper_half.mean() if upper_half.size > 0 else 0
            lower_energy = lower_half.mean() if lower_half.size > 0 else 0
            vertical_diff = (upper_energy - lower_energy) / max(upper_energy, lower_energy, 1)
            pitch = vertical_diff * 20
        
        # Roll is harder to estimate without landmarks
        roll = 0.0
        
        # Calculate frontal score
        frontal_score = 1.0 - (abs(yaw) + abs(pitch)) / 60.0 # Max deviation of 60 degrees sum
        frontal_score = max(0.0, min(1.0, frontal_score))
        
        return FacePose(
            yaw=float(yaw),
            pitch=float(pitch),
            roll=float(roll),
            frontal_score=float(frontal_score)
        )
        
    def _calculate_weighted_score(self, scores: Dict[str, Tuple[float, float]]) -> float:
        """Calculate weighted overall score"""
        total_score = 0.0
        total_weight = 0.0
        
        for name, (score, weight) in scores.items():
            total_score += score * weight
            total_weight += weight
            
        return total_score / total_weight if total_weight > 0 else 0.0
        
    def _determine_use_cases(self, metrics: QualityMetrics) -> List[str]:
        """Determine suitable use cases based on quality"""
        use_cases = []
        
        if metrics.overall_score >= 0.8:
            use_cases.extend([
                "face_recognition",
                "identity_verification", 
                "profile_photo",
                "document_photo"
            ])
        elif metrics.overall_score >= 0.6:
            use_cases.extend([
                "face_detection",
                "demographic_analysis",
                "social_media_photo"
            ])
        else:
            use_cases.append("basic_detection_only")
            
        # Specific quality checks
        if metrics.sharpness >= 0.7 and metrics.pose_quality >= 0.8:
            if "facial_landmark_detection" not in use_cases: use_cases.append("facial_landmark_detection")
            
        if metrics.illumination_uniformity >= 0.7:
            if "skin_analysis" not in use_cases: use_cases.append("skin_analysis")
            
        return list(set(use_cases)) # Ensure unique entries
    
    def _empty_result(self) -> Dict[str, Any]:
        # Ensure all keys from QualityMetrics and FacePose are present with default float values
        default_metric_values = {key: 0.0 for key in QualityMetrics.__annotations__ if key != 'overall_score'}
        default_pose_values = {key: 0.0 for key in FacePose.__annotations__}
        return {
            "overall_score": 0.0,
            "category": "poor",
            "metrics": default_metric_values,
            "pose": default_pose_values,
            "recommendations": ["Invalid or empty image provided."],
            "usable_for": ["none"]
        }

    def _generate_recommendations(self, metrics: QualityMetrics, pose: FacePose) -> List[str]:
        recommendations = []
        if metrics.sharpness < 0.5: recommendations.append("Improve sharpness. Ensure the image is in focus.")
        if metrics.brightness < 0.3 or metrics.brightness > 0.7: recommendations.append("Adjust brightness. Avoid over or under exposure.")
        if metrics.contrast < 0.4: recommendations.append("Increase contrast for better feature definition.")
        if metrics.face_size_score < 0.5: recommendations.append("Ensure face occupies a larger portion of the image.")
        if pose.frontal_score < 0.6: recommendations.append(f"Face should be more frontal (current frontal score: {pose.frontal_score:.2f}, yaw: {pose.yaw:.1f}°, pitch: {pose.pitch:.1f}°).")
        if metrics.occlusion_score > 0.3: recommendations.append(f"Reduce occlusions (score: {metrics.occlusion_score:.2f}). Ensure eyes, nose, and mouth are visible.")
        if metrics.blur_score > 0.5 : recommendations.append(f"Image is too blurry (blur score: {metrics.blur_score:.2f}).")
        if metrics.noise_level < 0.5 : recommendations.append(f"Image is too noisy (noise score: {metrics.noise_level:.2f}).")
        if metrics.illumination_uniformity < 0.5 : recommendations.append(f"Improve illumination uniformity (score: {metrics.illumination_uniformity:.2f}).")
        if not recommendations: recommendations.append("Image quality is generally good.")
        return recommendations

    def _determine_category(self, score: float) -> str:
        if score >= 0.8: return "excellent"
        if score >= 0.65: return "good"
        if score >= 0.5: return "fair"
        if score >= 0.35: return "poor"
        return "very_poor"

    def _calculate_brightness_advanced(self, gray_image: np.ndarray) -> float:
        if gray_image is None or gray_image.size == 0: return 0.0
        brightness = gray_image.mean() / 255.0
        # Score is 1 at 0.5 brightness, 0 at 0 or 1
        return max(0.0, 1.0 - abs(brightness - 0.5) * 2)

    def _calculate_contrast_advanced(self, gray_image: np.ndarray) -> float:
        if gray_image is None or gray_image.size == 0: return 0.0
        contrast = gray_image.std() / 128.0 # Normalize std dev (max std dev for uint8 is ~127.5)
        return min(contrast, 1.0)

    def _calculate_face_size_score(self, face_image: np.ndarray) -> float:
        if face_image is None or face_image.size == 0: return 0.0
        h, w = face_image.shape[:2]
        area = h * w
        # Ideal area could be e.g. 100x100 pixels for a good quality crop
        ideal_area = 100*100 
        return min(area / ideal_area, 1.0)

    def _estimate_pose_quality(self, face_image: np.ndarray) -> float:
        pose = self._estimate_face_pose(face_image)
        return pose.frontal_score

    def _assess_skin_quality(self, face_image: np.ndarray) -> float:
        if face_image is None or face_image.size == 0: return 0.0
        # Convert to YCrCb and check Cr Cb variance for skin tone consistency
        try:
            ycrcb = cv2.cvtColor(face_image, cv2.COLOR_BGRYCrCb)
            _, cr, cb = cv2.split(ycrcb)
            # Skin pixels typically fall in a certain range in CrCb space
            # This is a very simplified check for variance in those channels
            cr_std = cr.std()
            cb_std = cb.std()
            # Lower std dev might mean more uniform skin tone
            # Normalize score: higher is better (less variance)
            skin_variance_score = 1.0 - min((cr_std + cb_std) / 100.0, 1.0) # Arbitrary scaling
            return skin_variance_score
        except cv2.error:
            return 0.5 # Fallback if color conversion fails

    def _assess_expression_quality(self, gray_image: np.ndarray) -> float:
        # Placeholder: True expression analysis is complex (e.g. neutral, smile)
        # For now, assume a generic good quality is 0.8
        # Could be enhanced by checking for open/closed eyes, mouth shape if landmarks were available
        return 0.8

    def _calculate_motion_blur(self, gray_image: np.ndarray) -> float:
        # Inverse of sharpness can be a proxy for blur
        sharpness = self._calculate_sharpness_advanced(gray_image)
        return 1.0 - sharpness

    def _calculate_noise_level_advanced(self, gray_image: np.ndarray) -> float:
        if gray_image is None or gray_image.size == 0: return 0.0 # Low score for no image
        # Estimate noise using difference between image and blurred version
        try:
            blurred = cv2.GaussianBlur(gray_image, (5,5), 0)
            diff = cv2.absdiff(gray_image, blurred)
            noise_estimate = diff.mean()
            # Higher noise_estimate means more noise, so score should be lower
            noise_score = 1.0 - min(noise_estimate / 20.0, 1.0) # Arbitrary scaling
            return noise_score
        except cv2.error:
            return 0.3 # Fallback if blurring fails

    def _calculate_illumination_uniformity(self, gray_image: np.ndarray) -> float:
        if gray_image is None or gray_image.size == 0: return 0.0
        h, w = gray_image.shape
        if h < 2 or w < 2: return 0.0 # Not enough pixels to divide

        q_h, q_w = h // 2, w // 2
        quadrants = []
        if q_h > 0 and q_w > 0:
            quadrants.append(gray_image[0:q_h, 0:q_w])
            quadrants.append(gray_image[0:q_h, q_w:w])
            quadrants.append(gray_image[q_h:h, 0:q_w])
            quadrants.append(gray_image[q_h:h, q_w:w])
        else: # If image is too small, use the whole image as one quadrant
            quadrants.append(gray_image)
            
        mean_brightness = [q.mean() for q in quadrants if q.size > 0]
        if not mean_brightness: return 0.0
        
        std_dev_brightness = np.std(mean_brightness)
        # Lower std_dev means more uniform illumination
        uniformity = 1.0 - min(std_dev_brightness / 50.0, 1.0) # Arbitrary scaling
        return uniformity
