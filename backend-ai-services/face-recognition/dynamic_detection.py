#!/usr/bin/env python3
"""
Dynamic Face Detection Manager for FaceSocial
สร้างระบบการสลับ Face Detection ได้แบบ dynamic
"""

import asyncio
import aiohttp
import time
import logging
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum
import json
from dataclasses import dataclass
import cv2
import numpy as np

class DetectionMethod(Enum):
    """Available face detection methods"""
    MEDIAPIPE_INTERNAL = "mediapipe_internal"  # MediaPipe ภายใน face-recognition service
    MEDIAPIPE_SERVICE = "mediapipe_service"    # MediaPipe service แยก (port 8001)
    YOLO10 = "yolo10"                          # YOLO10 service (port 8002)
    MTCNN = "mtcnn"                           # MTCNN service (port 8003)

@dataclass
class DetectionConfig:
    """Configuration for face detection method"""
    method: DetectionMethod
    endpoint: str
    timeout: int = 30
    confidence_threshold: float = 0.7
    enabled: bool = True
    priority: int = 1  # 1 = highest, 3 = lowest
    
class FaceDetectionManager:
    """Dynamic Face Detection Manager"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Available detection methods with configurations
        self.detection_configs = {
            DetectionMethod.MEDIAPIPE_INTERNAL: DetectionConfig(
                method=DetectionMethod.MEDIAPIPE_INTERNAL,
                endpoint="internal",  # จะใช้ internal method
                priority=1,
                confidence_threshold=0.7
            ),
            DetectionMethod.YOLO10: DetectionConfig(
                method=DetectionMethod.YOLO10,
                endpoint="http://yolo10-main-detection:5000/detect",
                priority=2,
                confidence_threshold=0.6
            ),
            DetectionMethod.MTCNN: DetectionConfig(
                method=DetectionMethod.MTCNN,
                endpoint="http://mtcnn-precision:5000/detect",
                priority=3,
                confidence_threshold=0.8
            ),            DetectionMethod.MEDIAPIPE_SERVICE: DetectionConfig(
                method=DetectionMethod.MEDIAPIPE_SERVICE,
                endpoint="http://mediapipe-predetection:5000/detect_base64",
                priority=1,
                confidence_threshold=0.7
            )
        }
        
        # Current active method
        self.current_method = DetectionMethod.MEDIAPIPE_INTERNAL
        
        # Fallback chain
        self.fallback_chain = [
            DetectionMethod.MEDIAPIPE_INTERNAL,
            DetectionMethod.YOLO10,
            DetectionMethod.MTCNN,
            DetectionMethod.MEDIAPIPE_SERVICE
        ]
        
        # Performance tracking
        self.performance_stats = {}
        self.previous_method = None
        
    async def detect_faces_dynamic(self, image: np.ndarray, engine=None) -> List[Tuple[int, int, int, int]]:
        """
        Dynamic face detection with automatic fallback
        """
        # Try current method first
        try:
            faces = await self._detect_with_method(self.current_method, image, engine)
            if faces:
                self._update_performance_stats(self.current_method, success=True)
                return faces
        except Exception as e:
            self.logger.warning(f"Current method {self.current_method.value} failed: {e}")
            self._update_performance_stats(self.current_method, success=False)
        
        # Try fallback methods
        for method in self.fallback_chain:
            if method == self.current_method:
                continue  # Already tried
                
            try:
                faces = await self._detect_with_method(method, image, engine)
                if faces:
                    self.logger.info(f"Fallback to {method.value} successful")
                    self._update_performance_stats(method, success=True)
                    return faces
            except Exception as e:
                self.logger.warning(f"Fallback method {method.value} failed: {e}")
                self._update_performance_stats(method, success=False)
        
        # All methods failed
        self.logger.error("All face detection methods failed")
        return []
    
    async def _detect_with_method(self, method: DetectionMethod, image: np.ndarray, engine=None) -> List[Tuple[int, int, int, int]]:
        """Detect faces using specific method"""
        config = self.detection_configs[method]
        
        if not config.enabled:
            raise Exception(f"Method {method.value} is disabled")
        
        start_time = time.time()
        
        if method == DetectionMethod.MEDIAPIPE_INTERNAL:
            # Use internal MediaPipe detection
            if engine and hasattr(engine, '_detect_faces'):
                faces = engine._detect_faces(image)
            else:
                raise Exception("Engine not available for internal detection")
        else:
            # Use external service
            faces = await self._detect_with_service(config, image)
        
        detection_time = (time.time() - start_time) * 1000
        self.logger.info(f"{method.value} detection took {detection_time:.2f}ms, found {len(faces)} faces")
          return faces
    
    async def _detect_with_service(self, config: DetectionConfig, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces using external service"""
        import base64
        
        timeout = aiohttp.ClientTimeout(total=config.timeout)
        
        # MediaPipe service expects JSON with base64 data
        if config.method == DetectionMethod.MEDIAPIPE_SERVICE:
            # Encode image to base64
            _, buffer = cv2.imencode('.jpg', image)
            image_base64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
            
            payload = {'image': image_base64}
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(config.endpoint, json=payload) as response:
                    if response.status == 200:
                        result = await response.json()
                        return self._parse_detection_result(result, config.method)
                    else:
                        error_text = await response.text()
                        raise Exception(f"Service {config.method.value} returned {response.status}: {error_text}")
        else:
            # Other services expect multipart form data
            # Encode image to JPG bytes
            _, buffer = cv2.imencode('.jpg', image)
            image_bytes = buffer.tobytes()
            
            # Create multipart form data
            data = aiohttp.FormData()
            data.add_field('file', image_bytes, filename='image.jpg', content_type='image/jpeg')
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(config.endpoint, data=data) as response:
                    if response.status == 200:
                        result = await response.json()
                        return self._parse_detection_result(result, config.method)
                    else:
                        error_text = await response.text()
                        raise Exception(f"Service {config.method.value} returned {response.status}: {error_text}")
    
    def _parse_detection_result(self, result: Dict, method: DetectionMethod) -> List[Tuple[int, int, int, int]]:
        """Parse detection result from different services"""
        faces = []
        
        if method == DetectionMethod.YOLO10:
            # YOLO10 format: {"faces": [{"x": ..., "y": ..., "width": ..., "height": ...}]}
            detections = result.get('faces', [])
            for detection in detections:
                x = int(detection.get('x', 0))
                y = int(detection.get('y', 0))
                w = int(detection.get('width', 0))
                h = int(detection.get('height', 0))
                faces.append((x, y, w, h))
                
        elif method == DetectionMethod.MTCNN:
            # MTCNN format: {"faces": [{"x": ..., "y": ..., "width": ..., "height": ...}]}
            detections = result.get('faces', [])
            for detection in detections:
                x = int(detection.get('x', 0))
                y = int(detection.get('y', 0))
                w = int(detection.get('width', 0))
                h = int(detection.get('height', 0))
                faces.append((x, y, w, h))
                  elif method == DetectionMethod.MEDIAPIPE_SERVICE:
            # MediaPipe Service format: {"faces": [{"bbox": [x, y, w, h], "confidence": ..., "id": ...}]}
            detections = result.get('faces', [])
            for detection in detections:
                bbox = detection.get('bbox', [0, 0, 0, 0])
                if len(bbox) >= 4:
                    x, y, w, h = bbox[:4]
                    faces.append((int(x), int(y), int(w), int(h)))
        
        return faces
    
    def _update_performance_stats(self, method: DetectionMethod, success: bool):
        """Update performance statistics"""
        if method not in self.performance_stats:
            self.performance_stats[method] = {
                'success_count': 0,
                'failure_count': 0,
                'total_calls': 0
            }
        
        stats = self.performance_stats[method]
        stats['total_calls'] += 1
        
        if success:
            stats['success_count'] += 1
        else:
            stats['failure_count'] += 1
    
    def set_detection_method(self, method: DetectionMethod):
        """Set current detection method"""
        if method in self.detection_configs:
            self.current_method = method
            self.logger.info(f"Switched to detection method: {method.value}")
        else:
            raise ValueError(f"Unknown detection method: {method}")
    
    def get_current_method(self) -> DetectionMethod:
        """Get current detection method"""
        return self.current_method
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = {}
        for method, data in self.performance_stats.items():
            total = data['total_calls']
            success_rate = (data['success_count'] / total * 100) if total > 0 else 0
            
            stats[method.value] = {
                'success_count': data['success_count'],
                'failure_count': data['failure_count'],
                'total_calls': total,
                'success_rate': round(success_rate, 2)
            }
        
        return stats
    
    def configure_method(self, method: DetectionMethod, **kwargs):
        """Configure specific detection method"""
        if method in self.detection_configs:
            config = self.detection_configs[method]
            
            if 'enabled' in kwargs:
                config.enabled = kwargs['enabled']
            if 'confidence_threshold' in kwargs:
                config.confidence_threshold = kwargs['confidence_threshold']
            if 'priority' in kwargs:
                config.priority = kwargs['priority']
            if 'timeout' in kwargs:
                config.timeout = kwargs['timeout']
                
            self.logger.info(f"Updated configuration for {method.value}")
        else:
            raise ValueError(f"Unknown detection method: {method}")
    
    def set_fallback_chain(self, methods: List[DetectionMethod]):
        """Set custom fallback chain"""
        self.fallback_chain = methods
        self.logger.info(f"Updated fallback chain: {[m.value for m in methods]}")
    
    async def health_check_all_methods(self) -> Dict[str, bool]:
        """Check health of all detection methods"""
        health_status = {}
        
        for method, config in self.detection_configs.items():
            if method == DetectionMethod.MEDIAPIPE_INTERNAL:
                health_status[method.value] = True  # Always available internally
            else:
                try:
                    timeout = aiohttp.ClientTimeout(total=5)  # Quick health check
                    health_url = config.endpoint.replace('/detect', '/health')
                    
                    async with aiohttp.ClientSession(timeout=timeout) as session:
                        async with session.get(health_url) as response:
                            health_status[method.value] = response.status == 200
                except:
                    health_status[method.value] = False
        
        return health_status
    
    async def switch_method(self, method: DetectionMethod) -> bool:
        """Switch to a specific detection method"""
        try:
            if method not in self.detection_configs:
                raise ValueError(f"Unknown detection method: {method}")
            
            config = self.detection_configs[method]
            if not config.enabled:
                self.logger.warning(f"Method {method.value} is disabled")
                return False
            
            # Test the method first if it's an external service
            if method != DetectionMethod.MEDIAPIPE_INTERNAL:
                health_status = await self.health_check_all_methods()
                if not health_status.get(method.value, False):
                    self.logger.warning(f"Method {method.value} is not healthy")
                    return False
            
            self.previous_method = self.current_method
            self.current_method = method
            self.logger.info(f"Switched from {self.previous_method.value if self.previous_method else 'None'} to {method.value}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to switch to method {method.value}: {e}")
            return False

    def update_method_config(self, method: DetectionMethod, config_updates: Dict):
        """Update configuration for a specific method"""
        if method not in self.detection_configs:
            raise ValueError(f"Unknown detection method: {method}")
        
        config = self.detection_configs[method]
        
        for key, value in config_updates.items():
            if hasattr(config, key):
                setattr(config, key, value)
                self.logger.info(f"Updated {method.value} config: {key} = {value}")
            else:
                self.logger.warning(f"Unknown config key for {method.value}: {key}")

    async def check_all_methods_health(self) -> Dict[str, Dict]:
        """Check health of all detection methods with detailed info"""
        health_status = {}
        
        for method, config in self.detection_configs.items():
            method_status = {
                "healthy": False,
                "enabled": config.enabled,
                "response_time": None,
                "error": None
            }
            
            if method == DetectionMethod.MEDIAPIPE_INTERNAL:
                method_status["healthy"] = True
                method_status["response_time"] = 0
            else:
                try:
                    start_time = time.time()
                    timeout = aiohttp.ClientTimeout(total=5)
                    health_url = config.endpoint.replace('/detect', '/health')
                    
                    async with aiohttp.ClientSession(timeout=timeout) as session:
                        async with session.get(health_url) as response:
                            response_time = (time.time() - start_time) * 1000
                            method_status["healthy"] = response.status == 200
                            method_status["response_time"] = round(response_time, 2)
                            
                except Exception as e:
                    method_status["error"] = str(e)
            
            health_status[method.value] = method_status
        
        return health_status

    async def detect_faces_with_method(self, method: DetectionMethod, image: np.ndarray, engine=None) -> List[Tuple[int, int, int, int]]:
        """Detect faces using a specific method"""
        try:
            return await self._detect_with_method(method, image, engine)
        except Exception as e:
            self.logger.error(f"Detection failed with method {method.value}: {e}")
            return []
