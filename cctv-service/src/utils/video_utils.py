import cv2
import numpy as np
import base64
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def resize_frame(frame: np.ndarray, max_width: int = 640) -> Tuple[np.ndarray, float]:
    """Resize frame while maintaining aspect ratio"""
    height, width = frame.shape[:2]
    
    if width <= max_width:
        return frame, 1.0
    
    scale = max_width / width
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    resized = cv2.resize(frame, (new_width, new_height))
    
    return resized, scale


def draw_detection_box(
    frame: np.ndarray,
    bbox: list,
    label: str,
    confidence: float,
    is_known: bool = False
) -> np.ndarray:
    """Draw detection box and label on frame"""
    x1, y1, x2, y2 = bbox
    
    # Choose color based on detection type
    if is_known:
        color = (0, 255, 0)  # Green for known persons
    else:
        color = (0, 0, 255)  # Red for unknown
    
    # Draw rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # Prepare label
    label_text = f"{label} ({confidence:.2f})"
    
    # Get label size
    (label_width, label_height), baseline = cv2.getTextSize(
        label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
    )
    
    # Draw label background
    cv2.rectangle(
        frame,
        (x1, y1 - label_height - 10),
        (x1 + label_width, y1),
        color,
        -1
    )
    
    # Draw label text
    cv2.putText(
        frame,
        label_text,
        (x1, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2
    )
    
    return frame


def frame_to_base64(frame: np.ndarray, quality: int = 80) -> str:
    """Convert frame to base64 encoded JPEG"""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, buffer = cv2.imencode('.jpg', frame, encode_param)
    
    return base64.b64encode(buffer).decode('utf-8')


def base64_to_frame(base64_string: str) -> Optional[np.ndarray]:
    """Convert base64 string to OpenCV frame"""
    try:
        img_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        return frame
        
    except Exception as e:
        logger.error(f"Failed to decode base64 image: {str(e)}")
        return None


def validate_stream_url(url: str, stream_type: str) -> bool:
    """Validate stream URL format"""
    if stream_type == "webcam":
        return url.isdigit() or url == "0"
    
    elif stream_type == "rtsp":
        return url.startswith("rtsp://")
    
    elif stream_type == "http":
        return url.startswith("http://") or url.startswith("https://")
    
    return False


def get_video_info(cap: cv2.VideoCapture) -> dict:
    """Get video stream information"""
    info = {
        "fps": int(cap.get(cv2.CAP_PROP_FPS)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "codec": int(cap.get(cv2.CAP_PROP_FOURCC)),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    }
    
    # Convert codec to string
    if info["codec"] > 0:
        codec_chars = [
            chr((info["codec"] >> 8 * i) & 0xFF)
            for i in range(4)
        ]
        info["codec_string"] = "".join(codec_chars)
    else:
        info["codec_string"] = "unknown"
    
    return info


def calculate_fps(frame_times: list) -> float:
    """Calculate FPS from frame processing times"""
    if len(frame_times) < 2:
        return 0.0
    
    avg_time = sum(frame_times) / len(frame_times)
    
    if avg_time > 0:
        return 1.0 / avg_time
    
    return 0.0
