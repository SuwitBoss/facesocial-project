import cv2
import asyncio
import numpy as np
from datetime import datetime
import logging
import json
import httpx
import base64
from typing import Optional, Dict, Any
import time
from io import BytesIO
from PIL import Image
import uuid

from models.monitoring import DetectionResult, MonitoringStatus

logger = logging.getLogger(__name__)

class StreamProcessor:
    def __init__(
        self, 
        monitoring_id: str,
        stream_url: str,
        stream_type: str,
        redis_client,
        alert_manager,
        config: Dict[str, Any]
    ):
        self.monitoring_id = monitoring_id
        self.stream_url = stream_url
        self.stream_type = stream_type
        self.redis_client = redis_client
        self.alert_manager = alert_manager
        self.config = config
        
        self.is_running = False
        self.cap = None
        self.frame_count = 0
        self.last_detection_time = 0
        self.statistics = {
            "total_frames": 0,
            "total_detections": 0,
            "known_persons": 0,
            "unknown_persons": 0,
            "alerts_sent": 0,
            "fps": 0
        }
        
        # Face recognition service URL
        self.face_service_url = "http://face-recognition-service:3002"
        
    async def start_processing(self):
        """Start processing the video stream"""
        self.is_running = True
        logger.info(f"Starting stream processing for {self.monitoring_id}")

        # MOCK MODE: generate fake detections for testing
        if self.config.get("mock_mode", False):
            logger.info(f"[MOCK_MODE] Generating fake detections for {self.monitoring_id}")
            try:
                while self.is_running:
                    await asyncio.sleep(2)
                    now = datetime.utcnow().isoformat()
                    detection = {
                        "type": "detection",
                        "monitoring_id": self.monitoring_id,
                        "person_name": f"Test Person {np.random.randint(1,10)}",
                        "confidence": round(np.random.uniform(0.7, 0.99), 2),
                        "timestamp": now,
                        "face_id": str(uuid.uuid4()),
                        "image": None
                    }
                    # Publish to WebSocket via Redis pubsub
                    await self.redis_client.publish(f"detections:{self.monitoring_id}", json.dumps(detection))
                    # Save to results history
                    await self.redis_client.lpush(f"results:{self.monitoring_id}", json.dumps(detection))
                    # Update statistics
                    self.statistics["total_detections"] += 1
            except Exception as e:
                logger.error(f"[MOCK_MODE] Error: {str(e)}")
            finally:
                await self.cleanup()
            return

        try:
            # Initialize video capture
            if self.stream_type == "webcam":
                self.cap = cv2.VideoCapture(0)  # Default webcam
            else:
                self.cap = cv2.VideoCapture(self.stream_url)
            
            if not self.cap.isOpened():
                raise Exception("Failed to open video stream")
            
            # Set buffer size to reduce lag
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Get stream properties
            fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            if fps == 0:
                fps = 30  # Default FPS
            
            logger.info(f"Stream opened successfully. FPS: {fps}")
            
            # Process frames
            await self.process_frames()
            
        except Exception as e:
            logger.error(f"Stream processing error: {str(e)}")
            await self.handle_error(str(e))
        finally:
            await self.cleanup()
    
    async def process_frames(self):
        """Process video frames"""
        detection_interval_ms = self.config.get("detection_interval", 1000)
        detection_interval_frames = max(1, int(detection_interval_ms / 33))  # Assuming ~30fps
        
        frame_times = []
        
        while self.is_running:
            try:
                start_time = time.time()
                
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("Failed to read frame, attempting reconnect...")
                    await self.reconnect_stream()
                    continue
                
                self.frame_count += 1
                self.statistics["total_frames"] += 1
                
                # Process frame at intervals
                if self.frame_count % detection_interval_frames == 0:
                    await self.process_single_frame(frame)
                
                # Calculate FPS
                frame_times.append(time.time() - start_time)
                if len(frame_times) > 30:
                    frame_times.pop(0)
                    avg_time = sum(frame_times) / len(frame_times)
                    self.statistics["fps"] = int(1.0 / avg_time) if avg_time > 0 else 0
                
                # Small delay to prevent CPU overload
                await asyncio.sleep(0.001)
                
            except Exception as e:
                logger.error(f"Frame processing error: {str(e)}")
                await asyncio.sleep(0.1)
    
    async def process_single_frame(self, frame: np.ndarray):
        """Process a single frame for face detection and recognition"""
        try:
            current_time = time.time()
            
            # Skip if too soon since last detection
            if current_time - self.last_detection_time < (self.config.get("detection_interval", 1000) / 1000.0):
                return
            
            self.last_detection_time = current_time
            
            # Resize frame for faster processing
            resize_width = self.config.get("resize_width", 640)
            resize_height = self.config.get("resize_height", 480)
            
            height, width = frame.shape[:2]
            if width > resize_width:
                scale = resize_width / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame_resized = cv2.resize(frame, (new_width, new_height))
            else:
                frame_resized = frame
                scale = 1.0
            
            # Detect faces
            detections = await self.detect_faces(frame_resized)
            
            if detections and detections.get("faces_count", 0) > 0:
                # Process each detected face
                for face_info in detections["faces"]:
                    await self.process_face_detection(frame, face_info, scale)
            
        except Exception as e:
            logger.error(f"Frame processing error: {str(e)}")
    
    async def detect_faces(self, frame: np.ndarray) -> Optional[Dict]:
        """Detect faces in frame using face recognition service"""
        try:
            # Convert frame to JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            
            # Send to face detection service
            async with httpx.AsyncClient() as client:
                files = {"file": ("frame.jpg", buffer.tobytes(), "image/jpeg")}
                data = {"confidence_threshold": str(self.config.get("min_detection_confidence", 0.7))}
                
                response = await client.post(
                    f"{self.face_service_url}/face/detect",
                    files=files,
                    data=data,
                    timeout=5.0
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"Face detection failed: {response.status_code}")
                    return None
                    
        except Exception as e:
            logger.error(f"Face detection error: {str(e)}")
            return None
    
    async def process_face_detection(self, frame: np.ndarray, face_info: Dict, scale: float):
        """Process a single face detection"""
        try:
            # Adjust bbox coordinates based on scale
            bbox = face_info["bbox"]
            bbox_scaled = [int(coord / scale) for coord in bbox]
            
            # Extract face region
            x1, y1, x2, y2 = bbox_scaled
            face_img = frame[y1:y2, x1:x2]
            
            if face_img.size == 0:
                return
            
            # Identify the face
            identification = await self.identify_face(face_img)
            
            # Create detection result
            detection = DetectionResult(
                timestamp=datetime.utcnow(),
                frame_number=self.frame_count,
                confidence=face_info["confidence"],
                bbox=bbox_scaled,
                is_known_person=False
            )
            
            if identification and identification.get("identified"):
                best_match = identification.get("best_match")
                if best_match and best_match["similarity"] >= self.config.get("alert_threshold", 0.8):
                    detection.is_known_person = True
                    detection.user_id = best_match["user_id"]
                    detection.person_name = f"User_{best_match['user_id'][:8]}"
                    detection.confidence = best_match["similarity"]
                    
                    self.statistics["known_persons"] += 1
                    
                    # Save detection image
                    image_url = await self.save_detection_image(frame, bbox_scaled, detection)
                    detection.image_url = image_url
                    
                    # Send alert if enabled
                    if self.config.get("notify_on_match", True):
                        await self.send_alert(detection)
                else:
                    self.statistics["unknown_persons"] += 1
            else:
                self.statistics["unknown_persons"] += 1
            
            self.statistics["total_detections"] += 1
            
            # Save detection result
            if self.config.get("save_detections", True):
                await self.save_detection(detection)
            
            # Publish to WebSocket
            await self.publish_detection(detection)
            
        except Exception as e:
            logger.error(f"Face processing error: {str(e)}")
    
    async def identify_face(self, face_img: np.ndarray) -> Optional[Dict]:
        """Identify face using face recognition service"""
        try:
            # Convert face image to JPEG
            _, buffer = cv2.imencode('.jpg', face_img)
            
            # Send to face identification service
            async with httpx.AsyncClient() as client:
                files = {"file": ("face.jpg", buffer.tobytes(), "image/jpeg")}
                data = {
                    "top_k": "5",
                    "threshold": str(self.config.get("alert_threshold", 0.8))
                }
                
                response = await client.post(
                    f"{self.face_service_url}/face/identify",
                    files=files,
                    data=data,
                    timeout=5.0
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"Face identification failed: {response.status_code}")
                    return None
                    
        except Exception as e:
            logger.error(f"Face identification error: {str(e)}")
            return None
    
    async def save_detection_image(self, frame: np.ndarray, bbox: list, detection: DetectionResult) -> str:
        """Save detection image and return URL"""
        try:
            # Draw bbox on frame
            x1, y1, x2, y2 = bbox
            color = (0, 255, 0) if detection.is_known_person else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Add label
            label = f"{detection.person_name or 'Unknown'} ({detection.confidence:.2f})"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Convert to base64
            _, buffer = cv2.imencode('.jpg', frame)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Store in Redis with TTL
            image_key = f"detection_image:{self.monitoring_id}:{detection.frame_number}"
            await self.redis_client.setex(image_key, 3600, img_base64)  # 1 hour TTL
            
            return f"/images/{image_key}"
            
        except Exception as e:
            logger.error(f"Failed to save detection image: {str(e)}")
            return None
    
    async def save_detection(self, detection: DetectionResult):
        """Save detection result to Redis"""
        try:
            # Convert to JSON
            detection_data = {
                "timestamp": detection.timestamp.isoformat(),
                "frame_number": detection.frame_number,
                "confidence": detection.confidence,
                "bbox": detection.bbox,
                "is_known_person": detection.is_known_person,
                "user_id": detection.user_id,
                "person_name": detection.person_name,
                "image_url": detection.image_url
            }
            
            # Save to Redis list
            results_key = f"results:{self.monitoring_id}"
            await self.redis_client.lpush(results_key, json.dumps(detection_data))
            
            # Trim to keep only recent results
            await self.redis_client.ltrim(results_key, 0, 999)  # Keep last 1000 detections
            
            # Set expiry
            await self.redis_client.expire(results_key, 86400)  # 24 hours
            
        except Exception as e:
            logger.error(f"Failed to save detection: {str(e)}")
    
    async def publish_detection(self, detection: DetectionResult):
        """Publish detection to WebSocket subscribers"""
        try:
            # Create message
            message = {
                "type": "detection",
                "monitoring_id": self.monitoring_id,
                "timestamp": detection.timestamp.isoformat(),
                "frame_number": detection.frame_number,
                "is_known_person": detection.is_known_person,
                "person_name": detection.person_name,
                "confidence": detection.confidence,
                "bbox": detection.bbox,
                "image_url": detection.image_url
            }
            
            # Publish to Redis channel
            channel = f"detections:{self.monitoring_id}"
            await self.redis_client.publish(channel, json.dumps(message))
            
        except Exception as e:
            logger.error(f"Failed to publish detection: {str(e)}")
    
    async def send_alert(self, detection: DetectionResult):
        """Send alert for known person detection"""
        try:
            if self.alert_manager:
                await self.alert_manager.send_alert(
                    monitoring_id=self.monitoring_id,
                    detection=detection
                )
                
                detection.alert_sent = True
                self.statistics["alerts_sent"] += 1
                
        except Exception as e:
            logger.error(f"Failed to send alert: {str(e)}")
    
    async def reconnect_stream(self):
        """Attempt to reconnect to stream"""
        max_attempts = self.config.get("reconnect_attempts", 3)
        delay = self.config.get("reconnect_delay", 5)
        
        for attempt in range(max_attempts):
            logger.info(f"Reconnection attempt {attempt + 1}/{max_attempts}")
            
            if self.cap:
                self.cap.release()
            
            await asyncio.sleep(delay)
            
            try:
                if self.stream_type == "webcam":
                    self.cap = cv2.VideoCapture(0)
                else:
                    self.cap = cv2.VideoCapture(self.stream_url)
                
                if self.cap.isOpened():
                    logger.info("Reconnection successful")
                    return
                    
            except Exception as e:
                logger.error(f"Reconnection failed: {str(e)}")
        
        raise Exception("Failed to reconnect to stream")
    
    async def stop_processing(self):
        """Stop processing the stream"""
        logger.info(f"Stopping stream processing for {self.monitoring_id}")
        self.is_running = False
        await self.cleanup()
    
    async def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
            self.cap = None
    
    async def handle_error(self, error_message: str):
        """Handle processing errors"""
        # Update session status in Redis
        session_key = f"monitoring:{self.monitoring_id}"
        session_data = await self.redis_client.get(session_key)
        
        if session_data:
            session = json.loads(session_data)
            session["status"] = MonitoringStatus.ERROR.value
            session["error_message"] = error_message
            await self.redis_client.setex(session_key, 3600, json.dumps(session))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get current statistics"""
        return self.statistics.copy()
