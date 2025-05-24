from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel

class MonitoringStatus(Enum):
    STARTING = "starting"
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"

class DetectionResult(BaseModel):
    timestamp: datetime
    frame_number: int
    person_detected: Optional[Dict[str, Any]] = None
    confidence: float
    bbox: List[int]  # [x1, y1, x2, y2]
    face_id: Optional[str] = None
    user_id: Optional[str] = None
    person_name: Optional[str] = None
    image_url: Optional[str] = None
    is_known_person: bool = False
    alert_sent: bool = False

class MonitoringSession(BaseModel):
    id: str
    stream_url: str
    stream_type: str  # rtsp, http, webcam
    status: MonitoringStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    stopped_at: Optional[datetime] = None
    last_detection_at: Optional[datetime] = None
    config: Dict[str, Any]
    statistics: Dict[str, int] = {
        "total_frames": 0,
        "total_detections": 0,
        "known_persons": 0,
        "unknown_persons": 0,
        "alerts_sent": 0
    }
    error_message: Optional[str] = None
    
    def to_dict(self):
        return {
            "id": self.id,
            "stream_url": self.stream_url,
            "stream_type": self.stream_type,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "stopped_at": self.stopped_at.isoformat() if self.stopped_at else None,
            "last_detection_at": self.last_detection_at.isoformat() if self.last_detection_at else None,
            "config": self.config,
            "statistics": self.statistics,
            "error_message": self.error_message
        }

class Alert(BaseModel):
    id: str
    monitoring_id: str
    detection_id: str
    timestamp: datetime
    alert_type: str  # "known_person", "unknown_person", "multiple_faces"
    person_info: Dict[str, Any]
    confidence: float
    message: str
    image_url: Optional[str] = None
    sent_to: List[str] = []  # List of notification channels
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None

class StreamConfig(BaseModel):
    detection_interval: int = 1000  # milliseconds
    min_detection_confidence: float = 0.7
    notify_on_match: bool = True
    save_detections: bool = True
    alert_threshold: float = 0.8
    max_fps: int = 30
    resize_width: int = 640
    resize_height: int = 480
    buffer_size: int = 10
    reconnect_attempts: int = 3
    reconnect_delay: int = 5  # seconds
