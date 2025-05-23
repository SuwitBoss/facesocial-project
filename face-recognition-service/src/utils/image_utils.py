import cv2
import numpy as np
from PIL import Image
import base64
import io
from typing import Union

def decode_base64_image(base64_string: str) -> np.ndarray:
    """
    Decode base64 string to OpenCV image
    """
    # Remove data URL prefix if present
    if "data:image" in base64_string:
        base64_string = base64_string.split(",")[1]
    
    # Decode base64
    image_bytes = base64.b64decode(base64_string)
    
    # Convert to PIL Image
    pil_image = Image.open(io.BytesIO(image_bytes))
    
    # Convert to OpenCV format (BGR)
    opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    return opencv_image

def encode_image_to_base64(image: np.ndarray, format='JPEG') -> str:
    """
    Encode OpenCV image to base64 string
    """
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    pil_image = Image.fromarray(rgb_image)
    
    # Save to bytes
    img_buffer = io.BytesIO()
    pil_image.save(img_buffer, format=format)
    
    # Encode to base64
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
    
    return f"data:image/{format.lower()};base64,{img_base64}"

def bytes_to_opencv(image_bytes: bytes) -> np.ndarray:
    """
    Convert image bytes to OpenCV image
    """
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    
    # Decode image
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise ValueError("Could not decode image from bytes")
    
    return image

def draw_faces_on_image(image: np.ndarray, faces: list) -> np.ndarray:
    """
    Draw bounding boxes on detected faces
    """
    result_image = image.copy()
    
    for i, face in enumerate(faces):
        bbox = face["bbox"]
        confidence = face["confidence"]
        
        # Draw bounding box
        cv2.rectangle(result_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        
        # Draw label
        label = f"Face {i+1}: {confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        
        # Draw background for text
        cv2.rectangle(result_image, 
                     (bbox[0], bbox[1] - label_size[1] - 10),
                     (bbox[0] + label_size[0], bbox[1]),
                     (0, 255, 0), -1)
        
        # Draw text
        cv2.putText(result_image, label, (bbox[0], bbox[1] - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    return result_image

