"""
Face Processing Pipeline
Unified face detection and analysis pipeline for FaceSocial AI Services
"""

import asyncio
import aiohttp
import cv2
import numpy as np
import base64
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from io import BytesIO
from PIL import Image
import json
import uuid

from .face_models import (
    DetectedFace, BoundingBox, FaceLandmarks, FaceLandmark, QualityAssessment,
    FaceAttributes, ImageInfo, ProcessingStats, FaceDetectionMethod,
    DetectionOptions, ProcessingOptions, EmotionScores, FaceAngle, OcclusionScores
)

logger = logging.getLogger(__name__)

class FaceProcessingPipeline:
    """Unified face processing pipeline that orchestrates multiple AI services"""
    
    def __init__(self, http_session: aiohttp.ClientSession):
        self.session = http_session
        self.services = {
            'mediapipe': 'http://mediapipe-predetection:5000',
            'yolo': 'http://yolo10-main-detection:5000', 
            'mtcnn': 'http://mtcnn-precision:5000',
            'face_quality': 'http://face-quality:5000',
            'gender_age': 'http://gender-age-service:5000'
        }
        
        # Processing configuration
        self.default_detection_method = FaceDetectionMethod.MEDIAPIPE
        self.quality_threshold = 0.5
        self.max_parallel_requests = 5
        
    async def process_image(
        self,
        image_data: bytes,
        detection_options: DetectionOptions,
        processing_options: ProcessingOptions,
        detection_method: FaceDetectionMethod = None
    ) -> Tuple[List[DetectedFace], ImageInfo, ProcessingStats]:
        """Main processing pipeline"""
        
        start_time = time.time()
        method = detection_method or self.default_detection_method
        
        # Get image info
        image_info = self._get_image_info(image_data)
        
        # Stage 1: Face Detection
        detection_start = time.time()
        raw_faces = await self._detect_faces(image_data, detection_options, method)
        detection_time = (time.time() - detection_start) * 1000
        
        if not raw_faces:
            # No faces detected
            return [], image_info, ProcessingStats(
                detection_time_ms=detection_time,
                total_time_ms=(time.time() - start_time) * 1000
            )
        
        # Stage 2: Parallel processing of additional analyses
        landmarks_time = 0.0
        quality_time = 0.0
        attributes_time = 0.0
        alignment_time = 0.0
        
        detected_faces = []
        
        # Process each detected face
        for i, raw_face in enumerate(raw_faces):
            face_id = f"face_{i:03d}"
            
            # Convert raw detection to standardized format
            bbox = self._standardize_bounding_box(raw_face, method)
            
            detected_face = DetectedFace(
                face_id=face_id,
                face_index=i,
                bounding_box=bbox,
                detection_method=method
            )
            
            # Prepare tasks for parallel processing
            tasks = []
            
            # Task 1: Landmarks (if requested and supported)
            if detection_options.return_landmarks and method in [FaceDetectionMethod.MTCNN]:
                tasks.append(self._process_landmarks(raw_face, method))
            
            # Task 2: Quality Assessment (if requested)
            if detection_options.quality_assessment:
                face_crop = self._extract_face_crop(image_data, bbox)
                tasks.append(self._assess_face_quality(face_crop, bbox))
            
            # Task 3: Face Attributes (if requested)
            if detection_options.return_attributes:
                face_crop = self._extract_face_crop(image_data, bbox)
                tasks.append(self._analyze_face_attributes(face_crop, bbox))
            
            # Task 4: Face Alignment and Cropping (if requested)
            if processing_options.face_alignment or processing_options.crop_faces:
                face_crop = self._extract_face_crop(image_data, bbox)
                tasks.append(self._process_face_alignment_and_cropping(
                    face_crop, processing_options
                ))
            
            # Execute parallel tasks
            if tasks:
                task_start = time.time()
                results = await asyncio.gather(*tasks, return_exceptions=True)
                task_time = (time.time() - task_start) * 1000
                
                # Process results
                result_idx = 0
                
                if detection_options.return_landmarks and method in [FaceDetectionMethod.MTCNN]:
                    if not isinstance(results[result_idx], Exception):
                        detected_face.landmarks = results[result_idx]
                        landmarks_time = max(landmarks_time, task_time)
                    result_idx += 1
                
                if detection_options.quality_assessment:
                    if not isinstance(results[result_idx], Exception):
                        detected_face.quality_assessment = results[result_idx]
                        quality_time = max(quality_time, task_time)
                    result_idx += 1
                
                if detection_options.return_attributes:
                    if not isinstance(results[result_idx], Exception):
                        detected_face.face_attributes = results[result_idx]
                        attributes_time = max(attributes_time, task_time)
                    result_idx += 1
                
                if processing_options.face_alignment or processing_options.crop_faces:
                    if not isinstance(results[result_idx], Exception):
                        alignment_result = results[result_idx]
                        if processing_options.face_alignment and 'aligned_face' in alignment_result:
                            detected_face.aligned_face = alignment_result['aligned_face']
                        if processing_options.crop_faces and 'cropped_face' in alignment_result:
                            detected_face.cropped_face = alignment_result['cropped_face']
                        alignment_time = max(alignment_time, task_time)
                    result_idx += 1
            
            detected_faces.append(detected_face)
        
        # Create processing stats
        total_time = (time.time() - start_time) * 1000
        
        processing_stats = ProcessingStats(
            detection_time_ms=detection_time,
            landmarks_time_ms=landmarks_time if landmarks_time > 0 else None,
            quality_assessment_time_ms=quality_time if quality_time > 0 else None,
            attributes_time_ms=attributes_time if attributes_time > 0 else None,
            alignment_time_ms=alignment_time if alignment_time > 0 else None,
            total_time_ms=total_time
        )
        
        return detected_faces, image_info, processing_stats
    
    async def _detect_faces(
        self,
        image_data: bytes,
        options: DetectionOptions,
        method: FaceDetectionMethod
    ) -> List[Dict[str, Any]]:
        """Detect faces using specified method"""
        
        try:
            service_url = self.services[method.value]
            
            # Prepare request based on service type
            if method == FaceDetectionMethod.MEDIAPIPE:
                # MediaPipe expects JSON with base64 data
                image_b64 = base64.b64encode(image_data).decode('utf-8')
                payload = {'image': image_b64}
                
                url = f"{service_url}/detect_base64"
                async with self.session.post(url, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('faces', [])
            else:
                # YOLO and MTCNN expect multipart form data
                form_data = aiohttp.FormData()
                form_data.add_field('file', image_data, filename='image.jpg', content_type='image/jpeg')
                
                url = f"{service_url}/detect"
                async with self.session.post(url, data=form_data) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get('faces', [])
            
            return []
            
        except Exception as e:
            logger.error(f"Face detection failed with {method.value}: {e}")
            return []
    
    async def _assess_face_quality(
        self,
        face_image: bytes,
        bbox: BoundingBox
    ) -> Optional[QualityAssessment]:
        """Assess face quality using quality service"""
        
        try:
            service_url = self.services['face_quality']
            
            # Prepare face boxes for quality service
            face_boxes = [{
                "x": bbox.x,
                "y": bbox.y,
                "width": bbox.width,
                "height": bbox.height
            }]
            
            form_data = aiohttp.FormData()
            form_data.add_field('file', face_image, filename='face.jpg', content_type='image/jpeg')
            form_data.add_field('face_boxes', json.dumps(face_boxes))
            
            url = f"{service_url}/assess"
            async with self.session.post(url, data=form_data) as response:
                if response.status == 200:
                    data = await response.json()
                    faces = data.get('faces', [])
                    
                    if faces:
                        quality_data = faces[0]['quality_metrics']
                        return QualityAssessment(
                            overall_score=quality_data['overall_score'],
                            sharpness=quality_data['sharpness'],
                            brightness=quality_data['brightness'],
                            contrast=quality_data['contrast'],
                            face_angle=FaceAngle(**quality_data['face_angle']),
                            occlusion=OcclusionScores(**quality_data['occlusion']),
                            blur_score=quality_data['blur_score'],
                            noise_level=quality_data['noise_level'],
                            illumination_quality=quality_data['illumination_quality'],
                            resolution_adequacy=quality_data['resolution_adequacy']
                        )
            
            return None
            
        except Exception as e:
            logger.error(f"Face quality assessment failed: {e}")
            return None
    
    async def _analyze_face_attributes(
        self,
        face_image: bytes,
        bbox: BoundingBox
    ) -> Optional[FaceAttributes]:
        """Analyze face attributes using gender-age service"""
        
        try:
            service_url = self.services['gender_age']
            
            # Prepare face boxes for gender-age service
            face_boxes = [{
                "x": bbox.x,
                "y": bbox.y,
                "width": bbox.width,
                "height": bbox.height
            }]
            
            form_data = aiohttp.FormData()
            form_data.add_field('file', face_image, filename='face.jpg', content_type='image/jpeg')
            form_data.add_field('face_boxes', json.dumps(face_boxes))
            
            url = f"{service_url}/predict"
            async with self.session.post(url, data=form_data) as response:
                if response.status == 200:
                    data = await response.json()
                    predictions = data.get('predictions', [])
                    
                    if predictions:
                        pred = predictions[0]
                        
                        # Create emotion scores (simplified - would need emotion service)
                        emotion_scores = EmotionScores(
                            happy=0.5, sad=0.1, angry=0.1, surprised=0.1,
                            fearful=0.1, disgusted=0.1, neutral=0.1
                        )
                        
                        return FaceAttributes(
                            estimated_age=pred.get('age'),
                            estimated_gender=pred.get('gender'),
                            gender_confidence=pred.get('gender_confidence'),
                            dominant_emotion="neutral",  # Simplified
                            emotion_scores=emotion_scores
                        )
            
            return None
            
        except Exception as e:
            logger.error(f"Face attributes analysis failed: {e}")
            return None
    
    async def _process_landmarks(
        self,
        raw_face: Dict[str, Any],
        method: FaceDetectionMethod
    ) -> Optional[FaceLandmarks]:
        """Process landmarks from detection result"""
        
        try:
            if method == FaceDetectionMethod.MTCNN and 'landmarks' in raw_face:
                landmarks_data = raw_face['landmarks']
                
                if landmarks_data:
                    # Convert MTCNN landmarks to standard format
                    points = []
                    landmark_names = ['left_eye', 'right_eye', 'nose', 'mouth_left', 'mouth_right']
                    
                    for i, name in enumerate(landmark_names):
                        if i < len(landmarks_data):
                            points.append(FaceLandmark(
                                name=name,
                                x=float(landmarks_data[i][0]),
                                y=float(landmarks_data[i][1]),
                                confidence=1.0
                            ))
                    
                    return FaceLandmarks(
                        type="5_point",
                        points=points,
                        confidence=0.9
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Landmarks processing failed: {e}")
            return None
    
    async def _process_face_alignment_and_cropping(
        self,
        face_image: bytes,
        options: ProcessingOptions
    ) -> Dict[str, str]:
        """Process face alignment and cropping"""
        
        try:
            result = {}
            
            # For now, return the original face crop as both aligned and cropped
            # In a full implementation, this would include actual alignment algorithms
            face_b64 = base64.b64encode(face_image).decode('utf-8')
            
            if options.face_alignment:
                result['aligned_face'] = face_b64
            
            if options.crop_faces:
                result['cropped_face'] = face_b64
            
            return result
            
        except Exception as e:
            logger.error(f"Face alignment/cropping failed: {e}")
            return {}
    
    def _get_image_info(self, image_data: bytes) -> ImageInfo:
        """Extract image information"""
        
        try:
            # Decode image to get dimensions
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is not None:
                height, width = image.shape[:2]
                
                # Detect format from data
                image_format = "JPEG"  # Default
                if image_data.startswith(b'\x89PNG'):
                    image_format = "PNG"
                elif image_data.startswith(b'GIF'):
                    image_format = "GIF"
                
                return ImageInfo(
                    width=width,
                    height=height,
                    format=image_format,
                    file_size_bytes=len(image_data)
                )
            else:
                raise ValueError("Unable to decode image")
                
        except Exception as e:
            logger.error(f"Failed to get image info: {e}")
            # Return default info
            return ImageInfo(
                width=640,
                height=480,
                format="JPEG",
                file_size_bytes=len(image_data)
            )
    
    def _standardize_bounding_box(
        self,
        raw_face: Dict[str, Any],
        method: FaceDetectionMethod
    ) -> BoundingBox:
        """Convert detection result to standardized bounding box"""
        
        try:
            if method == FaceDetectionMethod.MEDIAPIPE:
                # MediaPipe format: bbox: [x, y, w, h]
                bbox_data = raw_face.get('bbox', [0, 0, 0, 0])
                return BoundingBox(
                    x=float(bbox_data[0]),
                    y=float(bbox_data[1]),
                    width=float(bbox_data[2]),
                    height=float(bbox_data[3]),
                    confidence=float(raw_face.get('confidence', 0.0))
                )
            
            elif method == FaceDetectionMethod.YOLO:
                # YOLO format: x, y, width, height, confidence
                return BoundingBox(
                    x=float(raw_face.get('x', 0)),
                    y=float(raw_face.get('y', 0)),
                    width=float(raw_face.get('width', 0)),
                    height=float(raw_face.get('height', 0)),
                    confidence=float(raw_face.get('confidence', 0.0))
                )
            
            elif method == FaceDetectionMethod.MTCNN:
                # MTCNN format: x, y, width, height, confidence
                return BoundingBox(
                    x=float(raw_face.get('x', 0)),
                    y=float(raw_face.get('y', 0)),
                    width=float(raw_face.get('width', 0)),
                    height=float(raw_face.get('height', 0)),
                    confidence=float(raw_face.get('confidence', 0.0))
                )
            
            else:
                # Fallback format
                return BoundingBox(
                    x=0.0, y=0.0, width=100.0, height=100.0, confidence=0.0
                )
                
        except Exception as e:
            logger.error(f"Failed to standardize bounding box: {e}")
            return BoundingBox(
                x=0.0, y=0.0, width=100.0, height=100.0, confidence=0.0
            )
    
    def _extract_face_crop(self, image_data: bytes, bbox: BoundingBox) -> bytes:
        """Extract face crop from image"""
        
        try:
            # Decode image
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is not None:
                # Extract face region
                x, y, w, h = int(bbox.x), int(bbox.y), int(bbox.width), int(bbox.height)
                
                # Ensure coordinates are within image bounds
                height, width = image.shape[:2]
                x = max(0, min(x, width - 1))
                y = max(0, min(y, height - 1))
                w = max(1, min(w, width - x))
                h = max(1, min(h, height - y))
                
                # Extract face
                face_crop = image[y:y+h, x:x+w]
                
                # Encode back to bytes
                _, encoded = cv2.imencode('.jpg', face_crop)
                return encoded.tobytes()
            
            # Fallback: return original image
            return image_data
            
        except Exception as e:
            logger.error(f"Failed to extract face crop: {e}")
            return image_data

# Service Health Checker
class ServiceHealthChecker:
    """Check health of individual AI services"""
    
    def __init__(self, http_session: aiohttp.ClientSession):
        self.session = http_session
        self.services = {
            'mediapipe': 'http://mediapipe-predetection:5000',
            'yolo': 'http://yolo10-main-detection:5000',
            'mtcnn': 'http://mtcnn-precision:5000',
            'face_quality': 'http://face-quality:5000',
            'gender_age': 'http://gender-age-service:5000'
        }
    
    async def check_service_health(self, service_name: str) -> Dict[str, Any]:
        """Check health of a specific service"""
        
        start_time = time.time()
        
        try:
            service_url = self.services.get(service_name)
            if not service_url:
                return {
                    'service': service_name,
                    'status': 'unknown',
                    'available': False,
                    'error': 'Service not configured'
                }
            
            url = f"{service_url}/health"
            async with self.session.get(url) as response:
                response_time = (time.time() - start_time) * 1000
                
                if response.status == 200:
                    data = await response.json()
                    return {
                        'service': service_name,
                        'status': 'healthy',
                        'available': True,
                        'response_time_ms': response_time,
                        'data': data
                    }
                else:
                    return {
                        'service': service_name,
                        'status': 'unhealthy',
                        'available': False,
                        'response_time_ms': response_time,
                        'status_code': response.status
                    }
                    
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return {
                'service': service_name,
                'status': 'error',
                'available': False,
                'response_time_ms': response_time,
                'error': str(e)
            }
    
    async def check_all_services(self) -> Dict[str, Dict[str, Any]]:
        """Check health of all services"""
        
        tasks = [
            self.check_service_health(service_name)
            for service_name in self.services.keys()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        health_status = {}
        for result in results:
            if isinstance(result, dict):
                service_name = result['service']
                health_status[service_name] = result
            else:
                # Handle exceptions
                logger.error(f"Health check error: {result}")
        
        return health_status