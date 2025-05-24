import httpx
import cv2
import numpy as np
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)


class FaceDetectorClient:
    """Client for Face Recognition Service"""
    
    def __init__(self, base_url: str = "http://face-recognition-service:3002"):
        self.base_url = base_url
        self.timeout = httpx.Timeout(10.0)
        
    async def detect_faces(
        self,
        image: np.ndarray,
        confidence_threshold: float = 0.7,
        return_image: bool = False
    ) -> Optional[Dict]:
        """Detect faces in image"""
        try:
            # Convert image to JPEG bytes
            _, buffer = cv2.imencode('.jpg', image)
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                files = {
                    "file": ("image.jpg", buffer.tobytes(), "image/jpeg")
                }
                data = {
                    "confidence_threshold": str(confidence_threshold),
                    "return_image": str(return_image).lower()
                }
                
                response = await client.post(
                    f"{self.base_url}/face/detect",
                    files=files,
                    data=data
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"Face detection failed: {response.status_code} - {response.text}")
                    return None
                    
        except Exception as e:
            logger.error(f"Face detection error: {str(e)}")
            return None
    
    async def identify_face(
        self,
        face_image: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.7
    ) -> Optional[Dict]:
        """Identify a face"""
        try:
            # Convert face image to JPEG bytes
            _, buffer = cv2.imencode('.jpg', face_image)
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                files = {
                    "file": ("face.jpg", buffer.tobytes(), "image/jpeg")
                }
                data = {
                    "top_k": str(top_k),
                    "threshold": str(threshold)
                }
                
                response = await client.post(
                    f"{self.base_url}/face/identify",
                    files=files,
                    data=data
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"Face identification failed: {response.status_code} - {response.text}")
                    return None
                    
        except Exception as e:
            logger.error(f"Face identification error: {str(e)}")
            return None
    
    async def verify_face(
        self,
        face_image: np.ndarray,
        user_id: str,
        threshold: float = 0.7
    ) -> Optional[Dict]:
        """Verify if face belongs to specific user"""
        try:
            # Convert face image to JPEG bytes
            _, buffer = cv2.imencode('.jpg', face_image)
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                files = {
                    "file": ("face.jpg", buffer.tobytes(), "image/jpeg")
                }
                data = {
                    "user_id": user_id,
                    "threshold": str(threshold)
                }
                
                response = await client.post(
                    f"{self.base_url}/face/verify",
                    files=files,
                    data=data
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"Face verification failed: {response.status_code} - {response.text}")
                    return None
                    
        except Exception as e:
            logger.error(f"Face verification error: {str(e)}")
            return None
    
    async def check_liveness(
        self,
        face_image: np.ndarray,
        check_spoofing: bool = True,
        check_texture: bool = True
    ) -> Optional[Dict]:
        """Check if face is live (anti-spoofing)"""
        try:
            # Convert face image to JPEG bytes
            _, buffer = cv2.imencode('.jpg', face_image)
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                files = {
                    "file": ("face.jpg", buffer.tobytes(), "image/jpeg")
                }
                data = {
                    "check_spoofing": str(check_spoofing).lower(),
                    "check_texture": str(check_texture).lower()
                }
                
                response = await client.post(
                    f"{self.base_url}/face/check-liveness",
                    files=files,
                    data=data
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"Liveness check failed: {response.status_code} - {response.text}")
                    return None
                    
        except Exception as e:
            logger.error(f"Liveness check error: {str(e)}")
            return None
    
    async def batch_identify_faces(
        self,
        face_images: List[np.ndarray],
        threshold: float = 0.7
    ) -> List[Optional[Dict]]:
        """Identify multiple faces in parallel"""
        import asyncio
        
        tasks = [
            self.identify_face(face_img, threshold=threshold)
            for face_img in face_images
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Batch identification error: {str(result)}")
                processed_results.append(None)
            else:
                processed_results.append(result)
        
        return processed_results
