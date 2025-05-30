# คู่มือการใช้งาน MN3_antispoof.onnx
## Face Anti-Spoofing Model (11.7 MB)

---

## 📋 ข้อมูลโมเดล

| รายการ | ค่า |
|--------|-----|
| **ชื่อโมเดล** | MN3_antispoof.onnx |
| **ขนาดไฟล์** | 11.7 MB |
| **Architecture** | MobileNetV3 Large |
| **Parameters** | ~3.02M |
| **Input Size** | 128x128x3 |
| **Output Classes** | 2 (Real/Spoof) |

---

## 🚀 การติดตั้ง Dependencies

### สำหรับ Python

```bash
# ONNX Runtime
pip install onnxruntime>=1.8.0
# หรือสำหรับ GPU
pip install onnxruntime-gpu>=1.8.0

# Image Processing
pip install opencv-python>=4.5.0
pip install numpy>=1.19.0
pip install Pillow>=8.0.0
pip install matplotlib>=3.3.0

# Optional: Model analysis
pip install onnx>=1.9.0
```

### ตรวจสอบการติดตั้ง

```python
import onnxruntime as ort
import cv2
import numpy as np
print(f"ONNX Runtime version: {ort.__version__}")
print(f"Available providers: {ort.get_available_providers()}")
```

---

## 🔧 การตั้งค่าโมเดล

### 1. โหลดโมเดล ONNX

```python
import onnxruntime as ort
import numpy as np
import cv2
from pathlib import Path

class FaceAntiSpoofing:
    def __init__(self, model_path, providers=None):
        """
        Initialize Face Anti-Spoofing model
        
        Args:
            model_path (str): Path to ONNX model file
            providers (list): Execution providers ['CPUExecutionProvider', 'CUDAExecutionProvider']
        """
        self.model_path = Path(model_path)
        
        # Set default providers
        if providers is None:
            available_providers = ort.get_available_providers()
            if 'CUDAExecutionProvider' in available_providers:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']
        
        # Create inference session
        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=providers
        )
        
        # Get model info
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        
        print(f"Model loaded: {self.model_path.name}")
        print(f"Input shape: {self.input_shape}")
        print(f"Providers: {self.session.get_providers()}")

# สร้าง instance
model = FaceAntiSpoofing('MN3_antispoof.onnx')
```

### 2. พารามิเตอร์การตั้งค่า

```python
# การตั้งค่าพารามิเตอร์หลัก
CONFIG = {
    # Image preprocessing
    'INPUT_SIZE': (128, 128),
    'MEAN': [0.5931, 0.4690, 0.4229],  # ImageNet mean values
    'STD': [0.2471, 0.2214, 0.2157],   # ImageNet std values
    
    # Model parameters
    'BATCH_SIZE': 1,
    'NUM_CLASSES': 2,
    'CLASS_NAMES': ['Real', 'Spoof'],
    
    # Inference settings
    'CONFIDENCE_THRESHOLD': 0.5,
    'TEMPERATURE': 1.0,  # For softmax temperature scaling
    
    # Performance settings
    'USE_GPU': True,
    'NUM_THREADS': 4,  # For CPU inference
}
```

---

## 🖼️ การประมวลผลภาพ (Preprocessing)

### ฟังก์ชัน Preprocessing

```python
def preprocess_image(image, config=CONFIG):
    """
    Preprocess image for model inference
    
    Args:
        image: Input image (BGR format from cv2 or RGB from PIL)
        config: Configuration dictionary
    
    Returns:
        preprocessed_image: Numpy array ready for inference
    """
    # Convert BGR to RGB if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        if isinstance(image, np.ndarray):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize image
    image = cv2.resize(
        image, 
        config['INPUT_SIZE'], 
        interpolation=cv2.INTER_CUBIC
    )
    
    # Normalize to [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # Apply mean and std normalization
    mean = np.array(config['MEAN']).reshape(1, 1, 3)
    std = np.array(config['STD']).reshape(1, 1, 3)
    image = (image - mean) / std
    
    # Convert to CHW format and add batch dimension
    image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
    image = np.expand_dims(image, axis=0)   # Add batch dimension
    
    return image.astype(np.float32)

# ตัวอย่างการใช้งาน
def load_and_preprocess(image_path):
    """Load and preprocess image from file"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot load image: {image_path}")
    
    return preprocess_image(image)
```

### ฟังก์ชัน Postprocessing

```python
def postprocess_output(output, config=CONFIG, temperature=1.0):
    """
    Process model output to get predictions
    
    Args:
        output: Raw model output
        config: Configuration dictionary
        temperature: Temperature for softmax scaling
    
    Returns:
        dict: Prediction results
    """
    # Apply temperature scaling
    if temperature != 1.0:
        output = output / temperature
    
    # Apply softmax
    exp_scores = np.exp(output - np.max(output, axis=1, keepdims=True))
    probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    # Get predictions
    predicted_class = np.argmax(probabilities, axis=1)[0]
    confidence = np.max(probabilities, axis=1)[0]
    
    # Get individual probabilities
    real_prob = probabilities[0][0]
    spoof_prob = probabilities[0][1]
    
    return {
        'prediction': config['CLASS_NAMES'][predicted_class],
        'confidence': float(confidence),
        'real_probability': float(real_prob),
        'spoof_probability': float(spoof_prob),
        'predicted_class': int(predicted_class),
        'raw_output': output[0].tolist()
    }
```

---

## 🔍 การใช้งานหลัก (Inference)

### 1. Inference แบบพื้นฐาน

```python
def predict_single_image(model, image_path, config=CONFIG):
    """
    Predict single image
    
    Args:
        model: FaceAntiSpoofing model instance
        image_path: Path to image file
        config: Configuration dictionary
    
    Returns:
        dict: Prediction results
    """
    try:
        # Preprocess image
        input_data = load_and_preprocess(image_path)
        
        # Run inference
        output = model.session.run(
            [model.output_name], 
            {model.input_name: input_data}
        )[0]
        
        # Postprocess output
        result = postprocess_output(output, config)
        
        return result
        
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

# ตัวอย่างการใช้งาน
result = predict_single_image(model, 'path/to/your/image.jpg')
if result:
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Real probability: {result['real_probability']:.4f}")
    print(f"Spoof probability: {result['spoof_probability']:.4f}")
```

### 2. Batch Inference

```python
def predict_batch(model, image_paths, config=CONFIG, batch_size=8):
    """
    Predict multiple images in batches
    
    Args:
        model: FaceAntiSpoofing model instance
        image_paths: List of image paths
        config: Configuration dictionary
        batch_size: Batch size for processing
    
    Returns:
        list: List of prediction results
    """
    results = []
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_inputs = []
        
        # Prepare batch
        for path in batch_paths:
            try:
                input_data = load_and_preprocess(path)
                batch_inputs.append(input_data[0])  # Remove batch dimension
            except Exception as e:
                print(f"Error processing {path}: {e}")
                batch_inputs.append(None)
        
        # Filter out None values
        valid_inputs = [inp for inp in batch_inputs if inp is not None]
        if not valid_inputs:
            continue
            
        # Stack inputs
        batch_array = np.stack(valid_inputs, axis=0)
        
        # Run inference
        batch_output = model.session.run(
            [model.output_name], 
            {model.input_name: batch_array}
        )[0]
        
        # Process each output
        for j, output in enumerate(batch_output):
            result = postprocess_output(
                output.reshape(1, -1), 
                config
            )
            result['image_path'] = batch_paths[j]
            results.append(result)
    
    return results
```

### 3. Real-time Video Processing

```python
def process_video_stream(model, config=CONFIG, source=0):
    """
    Process video stream in real-time
    
    Args:
        model: FaceAntiSpoofing model instance
        config: Configuration dictionary
        source: Video source (0 for webcam, or video file path)
    """
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video source {source}")
        return
    
    print("Press 'q' to quit, 's' to save frame")
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        try:
            # Preprocess frame
            input_data = preprocess_image(frame, config)
            
            # Run inference
            output = model.session.run(
                [model.output_name], 
                {model.input_name: input_data}
            )[0]
            
            # Get results
            result = postprocess_output(output, config)
            
            # Draw results on frame
            label = f"{result['prediction']}: {result['confidence']:.3f}"
            color = (0, 255, 0) if result['prediction'] == 'Real' else (0, 0, 255)
            
            cv2.putText(frame, label, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Display frame
            cv2.imshow('Face Anti-Spoofing', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f'frame_{frame_count:06d}.jpg'
                cv2.imwrite(filename, frame)
                print(f"Saved: {filename}")
                
        except Exception as e:
            print(f"Error processing frame {frame_count}: {e}")
            continue
    
    cap.release()
    cv2.destroyAllWindows()

# เรียกใช้
# process_video_stream(model)
```

---

## ⚙️ การปรับแต่งพารามิเตอร์

### 1. Performance Tuning

```python
# สำหรับ CPU Performance
def optimize_cpu_inference(model_path):
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 4
    session_options.inter_op_num_threads = 4
    session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    session = ort.InferenceSession(
        model_path,
        sess_options=session_options,
        providers=['CPUExecutionProvider']
    )
    return session

# สำหรับ GPU Performance
def optimize_gpu_inference(model_path):
    providers = [
        ('CUDAExecutionProvider', {
            'device_id': 0,
            'arena_extend_strategy': 'kNextPowerOfTwo',
            'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB
            'cudnn_conv_algo_search': 'EXHAUSTIVE',
            'do_copy_in_default_stream': True,
        }),
        'CPUExecutionProvider'
    ]
    
    session = ort.InferenceSession(model_path, providers=providers)
    return session
```

### 2. Confidence Threshold Tuning

```python
def evaluate_thresholds(model, test_images, true_labels, thresholds=None):
    """
    Evaluate different confidence thresholds
    
    Args:
        model: Model instance
        test_images: List of test image paths
        true_labels: List of true labels (0=Real, 1=Spoof)
        thresholds: List of thresholds to test
    
    Returns:
        dict: Evaluation results for each threshold
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.1)
    
    results = {}
    
    # Get predictions for all images
    predictions = []
    for img_path in test_images:
        result = predict_single_image(model, img_path)
        if result:
            predictions.append(result)
    
    # Evaluate each threshold
    for threshold in thresholds:
        correct = 0
        total = len(predictions)
        
        for i, pred in enumerate(predictions):
            if pred['confidence'] >= threshold:
                predicted_label = pred['predicted_class']
                if predicted_label == true_labels[i]:
                    correct += 1
        
        accuracy = correct / total if total > 0 else 0
        results[threshold] = {
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        }
    
    return results
```

### 3. Temperature Scaling

```python
def calibrate_temperature(model, validation_data, validation_labels):
    """
    Calibrate temperature parameter for better probability estimates
    
    Args:
        model: Model instance
        validation_data: Validation image paths
        validation_labels: True labels
    
    Returns:
        float: Optimal temperature value
    """
    from scipy.optimize import minimize_scalar
    
    def temperature_loss(temperature):
        total_loss = 0
        count = 0
        
        for img_path, true_label in zip(validation_data, validation_labels):
            result = predict_single_image(model, img_path)
            if result:
                # Get raw output and apply temperature
                raw_output = np.array([result['raw_output']])
                calibrated_result = postprocess_output(
                    raw_output, 
                    temperature=temperature
                )
                
                # Calculate negative log likelihood
                prob = calibrated_result['real_probability'] if true_label == 0 else calibrated_result['spoof_probability']
                total_loss += -np.log(max(prob, 1e-15))
                count += 1
        
        return total_loss / count if count > 0 else float('inf')
    
    # Find optimal temperature
    result = minimize_scalar(temperature_loss, bounds=(0.1, 10.0), method='bounded')
    optimal_temperature = result.x
    
    print(f"Optimal temperature: {optimal_temperature:.3f}")
    return optimal_temperature
```

---

## 📊 การวิเคราะห์และ Debugging

### 1. Model Analysis

```python
def analyze_model(model_path):
    """Analyze ONNX model structure and properties"""
    import onnx
    
    # Load ONNX model
    onnx_model = onnx.load(model_path)
    
    print("=== Model Information ===")
    print(f"IR Version: {onnx_model.ir_version}")
    print(f"Producer: {onnx_model.producer_name}")
    print(f"Model Version: {onnx_model.model_version}")
    
    # Input/Output info
    print("\n=== Input Information ===")
    for input_info in onnx_model.graph.input:
        print(f"Name: {input_info.name}")
        shape = [dim.dim_value for dim in input_info.type.tensor_type.shape.dim]
        print(f"Shape: {shape}")
        print(f"Type: {input_info.type.tensor_type.elem_type}")
    
    print("\n=== Output Information ===")
    for output_info in onnx_model.graph.output:
        print(f"Name: {output_info.name}")
        shape = [dim.dim_value for dim in output_info.type.tensor_type.shape.dim]
        print(f"Shape: {shape}")
        print(f"Type: {output_info.type.tensor_type.elem_type}")
    
    # Node statistics
    node_types = {}
    for node in onnx_model.graph.node:
        node_type = node.op_type
        node_types[node_type] = node_types.get(node_type, 0) + 1
    
    print("\n=== Node Statistics ===")
    for node_type, count in sorted(node_types.items()):
        print(f"{node_type}: {count}")

# เรียกใช้
# analyze_model('MN3_antispoof.onnx')
```

### 2. Performance Benchmarking

```python
import time

def benchmark_model(model, num_iterations=100, input_size=(1, 3, 128, 128)):
    """
    Benchmark model inference speed
    
    Args:
        model: Model instance
        num_iterations: Number of inference iterations
        input_size: Input tensor size
    
    Returns:
        dict: Benchmark results
    """
    # Prepare dummy input
    dummy_input = np.random.randn(*input_size).astype(np.float32)
    
    # Warmup
    for _ in range(10):
        _ = model.session.run([model.output_name], {model.input_name: dummy_input})
    
    # Benchmark
    start_time = time.time()
    for _ in range(num_iterations):
        _ = model.session.run([model.output_name], {model.input_name: dummy_input})
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time = total_time / num_iterations
    fps = 1.0 / avg_time
    
    results = {
        'total_time': total_time,
        'average_time': avg_time,
        'fps': fps,
        'iterations': num_iterations
    }
    
    print(f"=== Benchmark Results ===")
    print(f"Total time: {total_time:.3f}s")
    print(f"Average inference time: {avg_time*1000:.2f}ms")
    print(f"FPS: {fps:.1f}")
    
    return results

# เรียกใช้
# benchmark_results = benchmark_model(model)
```

---

## 🛠️ Troubleshooting

### ปัญหาที่พบบ่อยและวิธีแก้ไข

```python
# 1. Memory Issues
def check_memory_usage():
    """Check system memory usage"""
    import psutil
    memory = psutil.virtual_memory()
    print(f"Total memory: {memory.total / (1024**3):.2f} GB")
    print(f"Available memory: {memory.available / (1024**3):.2f} GB")
    print(f"Memory usage: {memory.percent}%")

# 2. Provider Issues
def troubleshoot_providers():
    """Troubleshoot execution providers"""
    available = ort.get_available_providers()
    print(f"Available providers: {available}")
    
    # Test each provider
    for provider in available:
        try:
            session = ort.InferenceSession('MN3_antispoof.onnx', providers=[provider])
            print(f"✓ {provider}: Working")
        except Exception as e:
            print(f"✗ {provider}: Error - {e}")

# 3. Input Shape Issues
def validate_input_shape(model, input_data):
    """Validate input data shape"""
    expected_shape = model.input_shape
    actual_shape = input_data.shape
    
    print(f"Expected shape: {expected_shape}")
    print(f"Actual shape: {actual_shape}")
    
    if list(actual_shape) != expected_shape:
        print("⚠️ Shape mismatch detected!")
        return False
    else:
        print("✓ Shape validation passed")
        return True
```

---

## 📝 ตัวอย่างการใช้งานครบถ้วน

```python
def main_example():
    """Complete usage example"""
    
    # 1. Initialize model
    print("Loading model...")
    model = FaceAntiSpoofing('MN3_antispoof.onnx')
    
    # 2. Single image prediction
    print("\n=== Single Image Test ===")
    result = predict_single_image(model, 'test_image.jpg')
    if result:
        print(f"Result: {result}")
    
    # 3. Batch prediction
    print("\n=== Batch Test ===")
    image_list = ['image1.jpg', 'image2.jpg', 'image3.jpg']
    batch_results = predict_batch(model, image_list)
    for result in batch_results:
        print(f"{result['image_path']}: {result['prediction']} ({result['confidence']:.3f})")
    
    # 4. Performance benchmark
    print("\n=== Performance Test ===")
    benchmark_model(model)
    
    # 5. Model analysis
    print("\n=== Model Analysis ===")
    analyze_model('MN3_antispoof.onnx')

if __name__ == "__main__":
    main_example()
```

---

## 📋 สรุปพารามิเตอร์สำคัญ

| พารามิเตอร์ | ค่าเริ่มต้น | คำอธิบาย | การปรับแต่ง |
|------------|-------------|----------|-------------|
| **INPUT_SIZE** | (128, 128) | ขนาดภาพอินพุต | ไม่ควรเปลี่ยน |
| **CONFIDENCE_THRESHOLD** | 0.5 | เกณฑ์ความมั่นใจ | 0.3-0.8 |
| **TEMPERATURE** | 1.0 | Temperature scaling | 0.5-3.0 |
| **BATCH_SIZE** | 1 | ขนาด batch | 1-32 |
| **NUM_THREADS** | 4 | จำนวน CPU threads | 1-8 |

**หมายเหตุ**: โมเดลนี้ได้รับการ train มาแล้วและพร้อมใช้งาน ขนาด 11.7 MB เหมาะสำหรับการใช้งาน production ที่ต้องการความเร็วและประสิทธิภาพสูง