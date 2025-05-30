# YOLO Face Detection Models Documentation

## Overview

ระบบ Face Detection ใช้ YOLO (You Only Look Once) models ในการตรวจจับใบหน้าแบบ real-time โดยมี 4 models หลักที่รองรับ:

- **YOLOv8n-face.onnx** - YOLOv8 Nano version สำหรับ face detection
- **YOLOv8s-face-lindevs.onnx** - YOLOv8 Small version จาก LinDevs
- **YOLOv10n-face.onnx** - YOLOv10 Nano version 
- **YOLOv11n-face.onnx** - YOLOv11 Nano version (ล่าสุด)

## Model Specifications

### YOLOv8 Face Detection Models

#### YOLOv8n-face.onnx
- **Model Type**: YOLOv8 Nano
- **Input Size**: 640x640 pixels
- **Input Format**: RGB, NCHW (Batch, Channels, Height, Width)
- **Output Format**: [1, 5, 8400] - [batch, features, detections]
- **Features**: [x_center, y_center, width, height, confidence]
- **Performance**: ~2-5ms inference time (GPU)
- **Memory Usage**: ~50MB GPU memory
- **Use Case**: Real-time applications, mobile deployment

#### YOLOv8s-face-lindevs.onnx  
- **Model Type**: YOLOv8 Small (LinDevs custom)
- **Input Size**: 640x640 pixels
- **Input Format**: RGB, NCHW
- **Output Format**: [1, 5, 8400]
- **Performance**: ~5-10ms inference time (GPU)
- **Memory Usage**: ~120MB GPU memory
- **Use Case**: Higher accuracy requirements

### YOLOv10 Face Detection Model

#### YOLOv10n-face.onnx
- **Model Type**: YOLOv10 Nano
- **Input Size**: 640x640 pixels
- **Input Format**: RGB, NCHW
- **Output Format**: [1, 300, 6] - [batch, max_detections, features]
- **Features**: [x1, y1, x2, y2, confidence, class]
- **Special**: NMS (Non-Maximum Suppression) already applied in model
- **Performance**: ~3-6ms inference time (GPU)
- **Memory Usage**: ~60MB GPU memory
- **Use Case**: Production deployment with built-in post-processing

### YOLOv11 Face Detection Model

#### YOLOv11n-face.onnx
- **Model Type**: YOLOv11 Nano (Latest)
- **Input Size**: 640x640 pixels
- **Input Format**: RGB, NCHW
- **Output Format**: [1, 5, 8400]
- **Features**: [x_center, y_center, width, height, confidence]
- **Performance**: ~2-4ms inference time (GPU)
- **Memory Usage**: ~45MB GPU memory
- **Use Case**: Best balance of speed and accuracy

## Implementation Details

### Model Loading and Initialization

```python
class YOLOv8FaceDetector(BaseFaceDetector):
    def __init__(self, model_path: str, confidence_threshold: float = 0.25):
        self.input_size = (640, 640)
        self.iou_threshold = 0.45
        self.model_type = self._detect_model_type(model_path)
        self._setup_providers()
        self.load_model()
    
    def _detect_model_type(self, model_path: str) -> str:
        """Auto-detect YOLO version from model path"""
        path_lower = model_path.lower()
        if 'yolov10' in path_lower:
            return 'yolov10'
        elif 'yolov11' in path_lower:
            return 'yolov11'
        else:
            return 'yolov8'
```

### ONNX Runtime Optimization

#### GPU Acceleration (CUDA)
```python
cuda_options = {
    'device_id': 0,
    'arena_extend_strategy': 'kSameAsRequested',
    'gpu_mem_limit': 3 * 1024 * 1024 * 1024,  # 3GB
    'cudnn_conv_algo_search': 'HEURISTIC',
    'do_copy_in_default_stream': True,
    'enable_cuda_graph': '0',
    'enable_mem_arena': '1',
    'memory_limit_mb': '3072',
    'prefer_nhwc': '1',
    'use_tf32': '1'
}
```

#### Session Options
```python
session_options = ort.SessionOptions()
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session_options.enable_mem_pattern = True
session_options.enable_mem_reuse = True
session_options.enable_cpu_mem_arena = True
session_options.intra_op_num_threads = 4
session_options.inter_op_num_threads = 2
```

### Image Preprocessing

#### Standard Preprocessing Pipeline
```python
def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    1. Resize with aspect ratio preservation
    2. Padding to square (640x640)
    3. BGR to RGB conversion
    4. Normalization (0-1)
    5. Transpose to NCHW format
    """
    original_height, original_width = image.shape[:2]
    
    # Calculate scale factor
    scale = min(self.input_size[0] / original_width, 
                self.input_size[1] / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    
    # Resize with optimal interpolation
    resized = cv2.resize(image, (new_width, new_height), 
                        interpolation=cv2.INTER_LINEAR)
    
    # Create padded image (114 = gray padding value)
    padded = np.full((self.input_size[1], self.input_size[0], 3), 114, dtype=np.uint8)
    
    # Center the resized image
    pad_x = (self.input_size[0] - new_width) // 2
    pad_y = (self.input_size[1] - new_height) // 2
    padded[pad_y:pad_y + new_height, pad_x:pad_x + new_width] = resized
    
    # Convert BGR to RGB and normalize
    padded_rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    normalized = padded_rgb.astype(np.float32) / 255.0
    
    # Transpose to NCHW format for ONNX
    input_tensor = normalized.transpose(2, 0, 1)
    input_tensor = np.expand_dims(input_tensor, axis=0)
    
    # Calculate scale factors for coordinate conversion
    scale_x = original_width / self.input_size[0]
    scale_y = original_height / self.input_size[1]
    
    return input_tensor, scale_x, scale_y
```

### Model Inference

#### Optimized Inference Pipeline
```python
def detect_faces(self, image: np.ndarray) -> List[DetectedFace]:
    """Main detection method with optimized inference"""
    
    # 1. Input validation
    if image is None or image.size == 0:
        return []
    
    original_height, original_width = image.shape[:2]
    
    # 2. Preprocess image
    input_tensor, scale_x, scale_y = self.preprocess_image(image)
    
    # 3. Run inference with timing
    start_time = cv2.getTickCount()
    outputs = self.session.run(self.output_names, 
                              {self.input_name: input_tensor})
    inference_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency() * 1000
    
    # 4. Postprocess results
    faces = self.postprocess_detections(outputs, scale_x, scale_y, 
                                       original_width, original_height)
    
    logger.info(f"🎯 Detected {len(faces)} faces in {inference_time:.2f}ms using {self.model_type}")
    return faces
```

### Output Post-processing

#### YOLOv8/YOLOv11 Post-processing
```python
# Output format: [1, 5, 8400] -> [8400, 5]
# Features: [center_x, center_y, width, height, confidence]

for detection in detections:
    if len(detection) < 5:
        continue
        
    center_x, center_y, width, height, confidence = detection[:5]
    
    if confidence < self.confidence_threshold:
        continue
    
    # Convert center format to corner format
    x1 = center_x - width / 2
    y1 = center_y - height / 2
    x2 = center_x + width / 2
    y2 = center_y + height / 2
    
    # Scale back to original image coordinates
    x1 = max(0, int(x1 * scale_x))
    y1 = max(0, int(y1 * scale_y))
    x2 = min(original_width, int(x2 * scale_x))
    y2 = min(original_height, int(y2 * scale_y))
```

#### YOLOv10 Post-processing
```python
# Output format: [1, 300, 6] -> [300, 6]
# Features: [x1, y1, x2, y2, confidence, class]
# Note: NMS already applied in model

for detection in detections:
    x1, y1, x2, y2, confidence, class_id = detection[:6]
    
    if confidence < self.confidence_threshold:
        continue
    
    # Coordinates are already in corner format
    # Scale back to original image coordinates
    x1 = max(0, int(x1 * scale_x))
    y1 = max(0, int(y1 * scale_y))
    x2 = min(original_width, int(x2 * scale_x))
    y2 = min(original_height, int(y2 * scale_y))
```

### Non-Maximum Suppression (NMS)

#### Custom NMS Implementation
```python
def _apply_nms(self, faces: List[DetectedFace]) -> List[DetectedFace]:
    """Apply Non-Maximum Suppression to remove overlapping detections"""
    if len(faces) <= 1:
        return faces
    
    # Convert to format for cv2.dnn.NMSBoxes
    boxes = []
    confidences = []
    
    for face in faces:
        boxes.append([face.bbox.x1, face.bbox.y1, 
                     face.bbox.x2 - face.bbox.x1, face.bbox.y2 - face.bbox.y1])
        confidences.append(face.confidence)
    
    # Apply NMS with optimized parameters
    indices = cv2.dnn.NMSBoxes(
        boxes, confidences, 
        self.confidence_threshold, 
        self.iou_threshold
    )
    
    if len(indices) > 0:
        return [faces[i] for i in indices.flatten()]
    
    return []
```

## Performance Benchmarks

### Inference Speed (GPU - RTX 3070)

| Model | Input Size | Inference Time | Memory Usage | Accuracy |
|-------|------------|----------------|--------------|----------|
| YOLOv8n-face | 640x640 | 2-5ms | ~50MB | Good |
| YOLOv8s-face-lindevs | 640x640 | 5-10ms | ~120MB | Better |
| YOLOv10n-face | 640x640 | 3-6ms | ~60MB | Good |
| YOLOv11n-face | 640x640 | 2-4ms | ~45MB | Best |

### Inference Speed (CPU - Intel i7-10700K)

| Model | Input Size | Inference Time | Memory Usage |
|-------|------------|----------------|--------------|
| YOLOv8n-face | 640x640 | 15-25ms | ~80MB |
| YOLOv8s-face-lindevs | 640x640 | 35-50ms | ~150MB |
| YOLOv10n-face | 640x640 | 20-30ms | ~90MB |
| YOLOv11n-face | 640x640 | 12-20ms | ~75MB |

## Configuration Options

### Model Selection
```python
# Environment variable for model selection
FACE_DETECTION_MODEL = "yolov11n-face.onnx"  # Recommended

# Available options:
# - yolov8n-face.onnx (fastest, good accuracy)
# - yolov8s-face-lindevs.onnx (slower, better accuracy)
# - yolov10n-face.onnx (built-in NMS)
# - yolov11n-face.onnx (latest, best balance)
```

### Detection Thresholds
```python
# Confidence threshold (0.0 - 1.0)
CONFIDENCE_THRESHOLD = 0.25  # Default: 0.25

# IoU threshold for NMS (0.0 - 1.0)  
IOU_THRESHOLD = 0.45  # Default: 0.45

# Minimum face size (pixels)
MIN_FACE_SIZE = 20  # Skip faces smaller than 20x20
```

### Device Configuration
```python
# Device selection
DEVICE = "cuda"  # Options: "cuda", "cpu"

# GPU memory limit (MB)
GPU_MEMORY_LIMIT = 3072  # 3GB

# CPU thread configuration
INTRA_OP_NUM_THREADS = 4
INTER_OP_NUM_THREADS = 2
```

## Usage Examples

### Basic Face Detection
```python
from app.ai_models.yolov8_face_optimized import YOLOv8FaceDetector

# Initialize detector
detector = YOLOv8FaceDetector(
    model_path="models/yolov11n-face.onnx",
    confidence_threshold=0.25
)

# Load image
image = cv2.imread("test_image.jpg")

# Detect faces
faces = detector.detect_faces(image)

# Process results
for face in faces:
    print(f"Face detected: confidence={face.confidence:.3f}")
    print(f"Bounding box: ({face.bbox.x1}, {face.bbox.y1}, {face.bbox.x2}, {face.bbox.y2})")
```

### Batch Processing
```python
import time

# Process multiple images
images = [cv2.imread(f"image_{i}.jpg") for i in range(10)]
total_time = 0

for i, image in enumerate(images):
    start_time = time.time()
    faces = detector.detect_faces(image)
    inference_time = (time.time() - start_time) * 1000
    total_time += inference_time
    
    print(f"Image {i}: {len(faces)} faces detected in {inference_time:.2f}ms")

avg_time = total_time / len(images)
print(f"Average inference time: {avg_time:.2f}ms")
```

### Real-time Video Processing
```python
import cv2

# Initialize video capture
cap = cv2.VideoCapture(0)  # Webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect faces
    faces = detector.detect_faces(frame)
    
    # Draw bounding boxes
    for face in faces:
        x1, y1, x2, y2 = face.bbox.x1, face.bbox.y1, face.bbox.x2, face.bbox.y2
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{face.confidence:.2f}", (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Display result
    cv2.imshow('Face Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## Troubleshooting

### Common Issues

#### 1. Model Loading Errors
```python
# Error: Model file not found
# Solution: Check model path and file existence
if not os.path.exists(model_path):
    print(f"Model file not found: {model_path}")
    
# Error: ONNX Runtime provider issues
# Solution: Install proper ONNX Runtime version
pip install onnxruntime-gpu  # For GPU
pip install onnxruntime      # For CPU only
```

#### 2. Performance Issues
```python
# Issue: Slow inference on GPU
# Solution: Check CUDA installation and GPU memory
import onnxruntime as ort
print("Available providers:", ort.get_available_providers())

# Issue: High memory usage
# Solution: Reduce GPU memory limit
gpu_mem_limit = 2 * 1024 * 1024 * 1024  # 2GB instead of 3GB
```

#### 3. Detection Quality Issues
```python
# Issue: Too many false positives
# Solution: Increase confidence threshold
confidence_threshold = 0.5  # Increase from 0.25

# Issue: Missing small faces
# Solution: Decrease confidence threshold and minimum face size
confidence_threshold = 0.15
min_face_size = 10
```

### Debug Mode
```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Monitor inference timing
start_time = cv2.getTickCount()
faces = detector.detect_faces(image)
inference_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency() * 1000
print(f"Debug: Inference time: {inference_time:.2f}ms")
```

## Model Comparison and Selection Guide

### Choose YOLOv8n-face when:
- Maximum speed is required
- Running on limited hardware
- Real-time applications (>30 FPS)
- Mobile/edge deployment

### Choose YOLOv8s-face-lindevs when:
- Higher accuracy is more important than speed
- Have sufficient GPU memory (>2GB)
- Processing high-resolution images
- Batch processing scenarios

### Choose YOLOv10n-face when:
- Want built-in NMS (no post-processing needed)
- Balanced speed and accuracy
- Production deployment
- Simplified inference pipeline

### Choose YOLOv11n-face when:
- Want the latest technology
- Best balance of speed and accuracy
- Future-proof solution
- Optimal memory usage

## Best Practices

1. **Model Selection**: Use YOLOv11n-face for new projects (best balance)
2. **Confidence Threshold**: Start with 0.25, adjust based on use case
3. **Batch Processing**: Process multiple images together for better GPU utilization
4. **Memory Management**: Monitor GPU memory usage and set appropriate limits
5. **Preprocessing**: Maintain consistent preprocessing pipeline across models
6. **Error Handling**: Always validate inputs and handle edge cases
7. **Performance Monitoring**: Log inference times for performance optimization

## Integration with FaceSocial System

### Model Deployment
```yaml
# docker-compose.yml
face-detection:
  environment:
    - FACE_DETECTION_MODEL=yolov11n-face.onnx
    - CONFIDENCE_THRESHOLD=0.25
    - DEVICE=cuda
    - GPU_MEMORY_LIMIT=3072
```

### API Usage
```python
# FastAPI endpoint
@app.post("/detect-faces")
async def detect_faces(file: UploadFile):
    image = await process_upload(file)
    faces = detector.detect_faces(image)
    return {
        "faces_count": len(faces),
        "faces": [face.dict() for face in faces],
        "model_type": detector.model_type
    }
```

This documentation provides comprehensive coverage of all YOLO face detection models used in the FaceSocial system, including implementation details, performance characteristics, and usage examples.
