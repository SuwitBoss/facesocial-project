# คู่มือการแบ่ง VRAM สำหรับ Face Social AI Services
## VRAM Allocation Guide for RTX 3060 6GB

---

## 📋 ข้อมูลระบบ

| รายการ | ค่า |
|--------|-----|
| **GPU** | NVIDIA GeForce RTX 3060 |
| **VRAM** | 6GB GDDR6 |
| **CPU** | AMD Ryzen 7 5800H (8 cores/16 threads) |
| **RAM** | 32GB DDR4-3200 MHz |
| **CUDA Version** | 12.9.0 + cuDNN |

---

## 🎯 AI Models Overview

### Face Detection Pipeline (3-Stage)

| โมเดล | ประเภท | Base Memory | ความเร็ว | หน้าที่ |
|-------|--------|-------------|----------|--------|
| **MediaPipe** | Google MediaPipe | ~80MB | 3-5ms | Pre-detection (screening) |
| **YOLOv10n** | YOLO v10 Nano | ~60MB | 3-6ms | Main detection (primary) |
| **MTCNN** | Multi-task CNN | ~150MB | 8-15ms | Precision detection (final) |

### Analysis Services

| โมเดล | ประเภท | Base Memory | ความเร็ว | หน้าที่ |
|-------|--------|-------------|----------|--------|
| **Face Recognition** | Ensemble (3 models) | ~200MB | 15-25ms | AdaFace + FaceNet + ArcFace |
| **Anti-spoof** | MN3 MobileNetV3 | ~12MB | 5-8ms | Liveness detection |
| **Gender & Age** | Custom CNN | ~50MB | 3-5ms | Demographic analysis |
| **Deepfake Detection** | ViT (FP16) | ~384MB | 50-100ms | AI-generated detection |

---

## 💾 VRAM Allocation Strategy

### 🎯 Optimized Distribution (6000MB Total)

```yaml
services:
  # === FACE DETECTION PIPELINE (1800MB) ===
  
  mediapipe-predetection:
    memory_limit: 400MB
    purpose: "Fast pre-detection screening"
    batch_size: 8
    confidence: 0.3
  
  yolo10-main-detection:
    memory_limit: 800MB
    purpose: "Primary face detection"
    batch_size: 3
    confidence: 0.25
  
  mtcnn-precision:
    memory_limit: 600MB
    purpose: "Precision refinement"
    batch_size: 2
    thresholds: [0.6, 0.7, 0.8]

  # === ANALYSIS SERVICES (4200MB) ===
  
  face-recognition:
    memory_limit: 1536MB
    purpose: "Face matching & identification"
    models: ["adaface_ir101", "facenet_vggface2", "arcface_r100"]
    weights: [0.25, 0.5, 0.25]
    batch_size: 2
  
  antispoof-service:
    memory_limit: 300MB
    purpose: "Liveness detection"
    model: "MN3_antispoof.onnx"
    input_size: 128x128
    batch_size: 4
  
  gender-age-service:
    memory_limit: 400MB
    purpose: "Demographic analysis"
    model: "genderage.onnx"
    batch_size: 3
  
  deepfake-detection:
    memory_limit: 2048MB
    purpose: "AI-generated content detection"
    model: "model_fp16.onnx"
    architecture: "ViT-base-patch16-224"
    accuracy: 98.84%
    batch_size: 1
```

### 📊 Memory Distribution Breakdown

| Service Category | Allocated VRAM | Percentage | Services Count |
|------------------|----------------|------------|----------------|
| **Face Detection** | 1800MB | 30% | 3 services |
| **Face Recognition** | 1536MB | 25.6% | 1 service (3 models) |
| **Deepfake Detection** | 2048MB | 34.1% | 1 service |
| **Other Analysis** | 700MB | 11.7% | 2 services |
| **System Buffer** | -84MB* | -1.4% | OS overhead |
| **Total** | **6000MB** | **100%** | **7 services** |

*หมายเหตุ: การปรับแต่ง memory allocation จริงจะใช้ 5916MB เพื่อให้ระบบมี buffer

---

## 🐳 Docker Compose Configuration

### Complete Services Setup

```yaml
version: '3.8'

services:
  # === FACE DETECTION PIPELINE ===
  
  mediapipe-predetection:
    build:
      context: ./backend-ai-services/mediapipe
      dockerfile: Dockerfile
    image: facesocial/mediapipe:latest
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=0
      - CUDA_MEM_LIMIT=400m
      - DETECTION_CONFIDENCE=0.3
      - MODEL_SELECTION=0
      - MAX_NUM_FACES=10
      - BATCH_SIZE=8
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]
              options:
                memory: 400M
    ports:
      - "8001:5000"
    volumes:
      - ./test-image:/app/test-images:ro
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  yolo10-main-detection:
    build:
      context: ./backend-ai-services/yolo-detection
      dockerfile: Dockerfile
    image: facesocial/yolo10:latest
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=0
      - CUDA_MEM_LIMIT=800m
      - MODEL_TYPE=yolov10n
      - MODEL_PATH=/app/models/yolov10n-face.onnx
      - BATCH_SIZE=3
      - CONFIDENCE_THRESHOLD=0.25
      - INPUT_SIZE=640
      - IOU_THRESHOLD=0.45
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]
              options:
                memory: 800M
    ports:
      - "8002:5000"
    volumes:
      - ./model/face-detection:/app/models:ro
      - ./test-image:/app/test-images:ro
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    depends_on:
      - mediapipe-predetection

  mtcnn-precision:
    build:
      context: ./backend-ai-services/mtcnn
      dockerfile: Dockerfile
    image: facesocial/mtcnn:latest
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=0
      - CUDA_MEM_LIMIT=600m
      - MIN_FACE_SIZE=20
      - SCALE_FACTOR=0.7
      - THRESHOLDS=0.6,0.7,0.8
      - BATCH_SIZE=2
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]
              options:
                memory: 600M
    ports:
      - "8003:5000"
    volumes:
      - ./test-image:/app/test-images:ro
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    depends_on:
      - yolo10-main-detection

  # === ANALYSIS SERVICES ===

  face-recognition:
    build:
      context: ./backend-ai-services/face-recognition
      dockerfile: Dockerfile
    image: facesocial/face-recognition:latest
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=0
      - CUDA_MEM_LIMIT=1536m
      - ENSEMBLE_MODE=true
      - ADAFACE_MODEL=/app/models/adaface_ir101.onnx
      - FACENET_MODEL=/app/models/facenet_vggface2.onnx
      - ARCFACE_MODEL=/app/models/arcface_r100.onnx
      - ENSEMBLE_WEIGHTS=0.25,0.5,0.25
      - BATCH_SIZE=2
      - SIMILARITY_THRESHOLD=0.6
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]
              options:
                memory: 1536M
    ports:
      - "8004:5000"
    volumes:
      - ./model/face-recognition:/app/models:ro
      - ./test-image:/app/test-images:ro
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  antispoof-service:
    build:
      context: ./backend-ai-services/antispoof
      dockerfile: Dockerfile
    image: facesocial/antispoof:latest
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=0
      - CUDA_MEM_LIMIT=300m
      - MODEL_PATH=/app/models/anti-spoof-mn3.onnx
      - INPUT_SIZE=128
      - BATCH_SIZE=4
      - CONFIDENCE_THRESHOLD=0.5
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]
              options:
                memory: 300M
    ports:
      - "8005:5000"
    volumes:
      - ./model/anti-spoofing:/app/models:ro
      - ./test-image:/app/test-images:ro
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  gender-age-service:
    build:
      context: ./backend-ai-services/gender-age
      dockerfile: Dockerfile
    image: facesocial/gender-age:latest
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=0
      - CUDA_MEM_LIMIT=400m
      - MODEL_PATH=/app/models/genderage.onnx
      - BATCH_SIZE=3
      - INPUT_SIZE=224
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]
              options:
                memory: 400M
    ports:
      - "8006:5000"
    volumes:
      - ./model/gender-age:/app/models:ro
      - ./test-image:/app/test-images:ro
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  deepfake-detection:
    build:
      context: ./backend-ai-services/deepfake
      dockerfile: Dockerfile
    image: facesocial/deepfake:latest
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=0
      - CUDA_MEM_LIMIT=2048m
      - MODEL_PATH=/app/models/model_fp16.onnx
      - MODEL_VERSION=fp16
      - VIT_BASE_PATCH16=true
      - INPUT_SIZE=224
      - BATCH_SIZE=1
      - ACCURACY_TARGET=98.84
      - CONFIDENCE_THRESHOLD=0.5
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: [gpu]
              options:
                memory: 2048M
    ports:
      - "8007:5000"
    volumes:
      - ./model/deepfake-detection:/app/models:ro
      - ./test-image:/app/test-images:ro
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # === API GATEWAY ===
  
  api-gateway:
    build:
      context: ./backend-ai-services/gateway
      dockerfile: Dockerfile
    image: facesocial/api-gateway:latest
    environment:
      - MEDIAPIPE_URL=http://mediapipe-predetection:5000
      - YOLO10_URL=http://yolo10-main-detection:5000
      - MTCNN_URL=http://mtcnn-precision:5000
      - FACE_RECOGNITION_URL=http://face-recognition:5000
      - ANTISPOOF_URL=http://antispoof-service:5000
      - GENDER_AGE_URL=http://gender-age-service:5000
      - DEEPFAKE_URL=http://deepfake-detection:5000
    ports:
      - "8000:8000"
    depends_on:
      - mediapipe-predetection
      - yolo10-main-detection
      - mtcnn-precision
      - face-recognition
      - antispoof-service
      - gender-age-service
      - deepfake-detection
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # === MONITORING ===
  
  nvidia-smi-exporter:
    image: mindprober/nvidia_gpu_prometheus_exporter:0.1
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    ports:
      - "9445:9445"
    restart: unless-stopped

volumes:
  gpu_monitoring_data:

networks:
  default:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

---

## 🔄 Detection Pipeline Modes

### 1. Fast Mode (Real-time)
- **Service**: MediaPipe only
- **Memory**: 400MB
- **Speed**: 3-5ms
- **Use Case**: Webcam, live streaming
- **API**: `POST /detect/fast`

```bash
curl -X POST http://localhost:8000/detect/fast \
  -F "image=@test-image/test_01.jpg" \
  -H "Content-Type: multipart/form-data"
```

### 2. Standard Mode (Balanced)
- **Services**: MediaPipe → YOLOv10n
- **Memory**: 1200MB
- **Speed**: 8-12ms
- **Use Case**: Photo processing, mobile apps
- **API**: `POST /detect/standard`

```bash
curl -X POST http://localhost:8000/detect/standard \
  -F "image=@test-image/group_01.jpg" \
  -H "Content-Type: multipart/form-data"
```

### 3. Precision Mode (High Accuracy)
- **Services**: MediaPipe → YOLOv10n → MTCNN
- **Memory**: 1800MB
- **Speed**: 15-25ms
- **Use Case**: Professional photography, security
- **API**: `POST /detect/precision`

```bash
curl -X POST http://localhost:8000/detect/precision \
  -F "image=@test-image/group_02.jpg" \
  -H "Content-Type: multipart/form-data"
```

### 4. Full Analysis Mode (Complete)
- **Services**: All detection + analysis services
- **Memory**: 6000MB
- **Speed**: 100-200ms
- **Use Case**: Complete face analysis, research
- **API**: `POST /analyze/full`

```bash
curl -X POST http://localhost:8000/analyze/full \
  -F "image=@test-image/real_0.jpg" \
  -H "Content-Type: multipart/form-data"
```

---

## 📊 Performance Benchmarks

### RTX 3060 6GB Performance

| Pipeline Mode | Memory Usage | Processing Time | Throughput | Accuracy |
|---------------|-------------|----------------|------------|----------|
| **Fast** | 400MB | 3-5ms | 200-300 FPS | Good |
| **Standard** | 1200MB | 8-12ms | 80-120 FPS | Better |
| **Precision** | 1800MB | 15-25ms | 40-60 FPS | Best |
| **Full Analysis** | 6000MB | 100-200ms | 5-10 FPS | Complete |

### Individual Service Performance

| Service | Memory | Inference Time | Batch Size | Throughput |
|---------|--------|----------------|------------|------------|
| **MediaPipe** | 400MB | 3-5ms | 8 | 300+ FPS |
| **YOLOv10n** | 800MB | 3-6ms | 3 | 150-200 FPS |
| **MTCNN** | 600MB | 8-15ms | 2 | 60-80 FPS |
| **Face Recognition** | 1536MB | 15-25ms | 2 | 40-60 FPS |
| **Anti-spoof** | 300MB | 5-8ms | 4 | 120-150 FPS |
| **Gender/Age** | 400MB | 3-5ms | 3 | 180-250 FPS |
| **Deepfake** | 2048MB | 50-100ms | 1 | 10-20 FPS |

---

## 🛠️ Memory Optimization Techniques

### 1. Dynamic Memory Management

```python
import torch
import gc
from contextlib import contextmanager

class GPUMemoryManager:
    def __init__(self, max_memory_mb: int):
        self.max_memory = max_memory_mb * 1024 * 1024
        self.setup_memory_limit()
    
    def setup_memory_limit(self):
        if torch.cuda.is_available():
            # Set memory fraction
            total_memory = torch.cuda.get_device_properties(0).total_memory
            fraction = self.max_memory / total_memory
            torch.cuda.set_per_process_memory_fraction(fraction)
            
            # Configure memory allocation
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
    
    @contextmanager
    def memory_context(self):
        """Context manager for memory cleanup"""
        try:
            yield
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
    def get_memory_stats(self):
        """Get current GPU memory statistics"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**2  # MB
            cached = torch.cuda.memory_reserved() / 1024**2     # MB
            return {
                'allocated_mb': allocated,
                'cached_mb': cached,
                'free_mb': self.max_memory / 1024**2 - allocated
            }
        return {'allocated_mb': 0, 'cached_mb': 0, 'free_mb': 0}

# Usage in each service
memory_manager = GPUMemoryManager(max_memory_mb=400)  # MediaPipe

with memory_manager.memory_context():
    result = model.predict(image)
```

### 2. Model Loading Optimization

```python
import onnxruntime as ort

def create_optimized_session(model_path: str, memory_limit_mb: int):
    """Create optimized ONNX session with memory limits"""
    
    # Session options
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.enable_mem_pattern = True
    session_options.enable_mem_reuse = True
    session_options.enable_cpu_mem_arena = True
    session_options.intra_op_num_threads = 4
    session_options.inter_op_num_threads = 2
    
    # CUDA provider options
    cuda_options = {
        'device_id': 0,
        'arena_extend_strategy': 'kSameAsRequested',
        'gpu_mem_limit': memory_limit_mb * 1024 * 1024,
        'cudnn_conv_algo_search': 'HEURISTIC',
        'do_copy_in_default_stream': True,
        'enable_cuda_graph': '0',
        'enable_mem_arena': '1',
        'memory_limit_mb': str(memory_limit_mb),
        'prefer_nhwc': '1',
        'use_tf32': '1'
    }
    
    providers = [
        ('CUDAExecutionProvider', cuda_options),
        'CPUExecutionProvider'
    ]
    
    return ort.InferenceSession(
        model_path,
        sess_options=session_options,
        providers=providers
    )
```

### 3. Batch Processing Optimization

```python
import numpy as np
from typing import List, Tuple
import asyncio

class BatchProcessor:
    def __init__(self, model_session, batch_size: int, memory_limit_mb: int):
        self.session = model_session
        self.batch_size = batch_size
        self.memory_limit = memory_limit_mb
        self.queue = asyncio.Queue()
    
    async def process_batch(self, images: List[np.ndarray]) -> List[dict]:
        """Process images in optimal batches"""
        results = []
        
        for i in range(0, len(images), self.batch_size):
            batch = images[i:i + self.batch_size]
            
            # Prepare batch tensor
            batch_tensor = np.stack(batch)
            
            # Run inference
            outputs = self.session.run(None, {'input': batch_tensor})
            
            # Process outputs
            batch_results = self._process_outputs(outputs)
            results.extend(batch_results)
            
            # Memory cleanup after each batch
            del batch_tensor, outputs
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return results
    
    def _process_outputs(self, outputs) -> List[dict]:
        """Process model outputs to structured results"""
        # Implementation depends on specific model
        pass
```

---

## 🔍 Monitoring & Health Checks

### 1. GPU Memory Monitoring

```python
import psutil
import pynvml
from datetime import datetime

class GPUMonitor:
    def __init__(self):
        pynvml.nvmlInit()
        self.device_count = pynvml.nvmlDeviceGetCount()
    
    def get_gpu_stats(self):
        """Get comprehensive GPU statistics"""
        stats = []
        
        for i in range(self.device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            
            # Memory info
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            # Utilization
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            
            # Temperature
            temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            
            stats.append({
                'device_id': i,
                'memory_total_mb': memory_info.total / 1024**2,
                'memory_used_mb': memory_info.used / 1024**2,
                'memory_free_mb': memory_info.free / 1024**2,
                'memory_usage_percent': (memory_info.used / memory_info.total) * 100,
                'gpu_utilization_percent': utilization.gpu,
                'memory_utilization_percent': utilization.memory,
                'temperature_celsius': temperature,
                'timestamp': datetime.now().isoformat()
            })
        
        return stats
    
    def check_memory_threshold(self, threshold_percent: float = 90.0):
        """Check if memory usage exceeds threshold"""
        stats = self.get_gpu_stats()
        for stat in stats:
            if stat['memory_usage_percent'] > threshold_percent:
                return False, f"GPU {stat['device_id']} memory usage: {stat['memory_usage_percent']:.1f}%"
        return True, "Memory usage normal"

# Usage
monitor = GPUMonitor()
stats = monitor.get_gpu_stats()
print(f"GPU Memory Usage: {stats[0]['memory_usage_percent']:.1f}%")
```

### 2. Service Health Checks

```python
import aiohttp
import asyncio
from typing import Dict, List

class ServiceHealthChecker:
    def __init__(self):
        self.services = {
            'mediapipe': 'http://mediapipe-predetection:5000/health',
            'yolo10': 'http://yolo10-main-detection:5000/health',
            'mtcnn': 'http://mtcnn-precision:5000/health',
            'face_recognition': 'http://face-recognition:5000/health',
            'antispoof': 'http://antispoof-service:5000/health',
            'gender_age': 'http://gender-age-service:5000/health',
            'deepfake': 'http://deepfake-detection:5000/health'
        }
    
    async def check_service_health(self, service_name: str, url: str) -> Dict:
        """Check individual service health"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            'service': service_name,
                            'status': 'healthy',
                            'response_time_ms': data.get('response_time', 0),
                            'memory_usage_mb': data.get('memory_usage', 0),
                            'model_loaded': data.get('model_loaded', False)
                        }
                    else:
                        return {
                            'service': service_name,
                            'status': 'unhealthy',
                            'error': f"HTTP {response.status}"
                        }
        except Exception as e:
            return {
                'service': service_name,
                'status': 'error',
                'error': str(e)
            }
    
    async def check_all_services(self) -> List[Dict]:
        """Check all services health concurrently"""
        tasks = []
        for name, url in self.services.items():
            tasks.append(self.check_service_health(name, url))
        
        results = await asyncio.gather(*tasks)
        return results
    
    async def get_system_status(self) -> Dict:
        """Get overall system health status"""
        service_results = await self.check_all_services()
        gpu_monitor = GPUMonitor()
        gpu_stats = gpu_monitor.get_gpu_stats()
        
        healthy_services = sum(1 for r in service_results if r['status'] == 'healthy')
        total_services = len(service_results)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'healthy' if healthy_services == total_services else 'degraded',
            'services': {
                'total': total_services,
                'healthy': healthy_services,
                'unhealthy': total_services - healthy_services
            },
            'gpu': gpu_stats[0] if gpu_stats else {},
            'service_details': service_results
        }

# Usage
health_checker = ServiceHealthChecker()
status = await health_checker.get_system_status()
print(f"System Status: {status['overall_status']}")
```

---

## 🚀 Deployment Commands

### 1. Start All Services

```powershell
# Build all images
docker-compose build

# Start services in order
docker-compose up -d mediapipe-predetection
docker-compose up -d yolo10-main-detection
docker-compose up -d mtcnn-precision
docker-compose up -d face-recognition
docker-compose up -d antispoof-service
docker-compose up -d gender-age-service
docker-compose up -d deepfake-detection
docker-compose up -d api-gateway

# Check services status
docker-compose ps

# View logs
docker-compose logs -f api-gateway
```

### 2. Test Pipeline

```powershell
# Test fast detection
Invoke-WebRequest -Uri "http://localhost:8000/detect/fast" `
  -Method POST `
  -InFile "test-image/test_01.jpg" `
  -ContentType "multipart/form-data"

# Test full analysis
Invoke-WebRequest -Uri "http://localhost:8000/analyze/full" `
  -Method POST `
  -InFile "test-image/group_01.jpg" `
  -ContentType "multipart/form-data"
```

### 3. Monitor GPU Usage

```powershell
# Watch GPU memory usage
nvidia-smi -l 1

# View service metrics
docker stats

# Check container logs for memory issues
docker-compose logs --tail=50 deepfake-detection
```

---

## 📈 Scaling Considerations

### 1. Horizontal Scaling

สำหรับการขยายระบบเมื่อมี GPU หลายตัว:

```yaml
# Multi-GPU configuration
services:
  yolo10-main-detection-gpu0:
    environment:
      - NVIDIA_VISIBLE_DEVICES=0
      - CUDA_MEM_LIMIT=800m
  
  yolo10-main-detection-gpu1:
    environment:
      - NVIDIA_VISIBLE_DEVICES=1
      - CUDA_MEM_LIMIT=800m
  
  load-balancer:
    image: nginx:alpine
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    ports:
      - "8080:80"
```

### 2. Memory Optimization for Production

```yaml
# Production optimized configuration
environment:
  - PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
  - CUDA_LAUNCH_BLOCKING=0
  - CUDNN_BENCHMARK=1
  - TENSORRT_OPTIMIZATION=1
```

### 3. Failover Strategy

```yaml
# Health check and restart configuration
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 60s

restart: unless-stopped
```

---

## 🔧 Troubleshooting

### Common Memory Issues

| Error | Cause | Solution |
|-------|-------|----------|
| `CUDA out of memory` | Memory limit exceeded | Reduce batch size or memory limit |
| `Container killed (OOMKilled)` | Host memory exhausted | Increase swap or reduce containers |
| `Model loading failed` | Insufficient VRAM | Check available memory before loading |
| `Slow inference` | Memory fragmentation | Restart container or clear cache |

### Performance Tuning

1. **Batch Size Optimization**
   ```python
   # Test different batch sizes
   for batch_size in [1, 2, 4, 8]:
       test_performance(batch_size)
   ```

2. **Memory Cleanup**
   ```python
   # Regular cleanup in long-running services
   if step % 100 == 0:
       torch.cuda.empty_cache()
       gc.collect()
   ```

3. **Model Quantization**
   ```python
   # Use FP16 models for memory efficiency
   session_options.add_session_config_entry('use_fp16', '1')
   ```

---

## 📚 References

- [NVIDIA Docker Documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
- [ONNX Runtime GPU Performance](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
- [Docker Compose GPU Support](https://docs.docker.com/compose/gpu-support/)
- [CUDA Memory Management](https://pytorch.org/docs/stable/notes/cuda.html)

---

**เอกสารนี้อัปเดตล่าสุด**: 29 พฤษภาคม 2025  
**เวอร์ชัน**: 1.0  
**ผู้จัดทำ**: Face Social AI Team
