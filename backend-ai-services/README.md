# FaceSocial AI Services Infrastructure

A comprehensive Docker-based AI microservices infrastructure for face detection, recognition, anti-spoofing, gender/age analysis, and deepfake detection optimized for RTX 3060 6GB VRAM.

## 🏗️ Architecture Overview

This infrastructure consists of 7 AI microservices orchestrated through an API Gateway:

### 🤖 AI Services
1. **MediaPipe** (Port 8001) - Fast face pre-detection and screening
2. **YOLOv10n** (Port 8002) - Primary face detection with bounding boxes
3. **MTCNN** (Port 8003) - Precision face detection with landmarks
4. **Face Recognition** (Port 8004) - Identity matching with ensemble models
5. **Anti-spoofing** (Port 8005) - Liveness detection and spoof prevention
6. **Gender/Age** (Port 8006) - Demographic analysis
7. **Deepfake Detection** (Port 8007) - AI-generated content detection

### 🌐 Infrastructure Services
- **API Gateway** (Port 8000) - Service orchestration and routing
- **PostgreSQL** (Port 5432) - Primary database with pgvector support
- **Redis** (Port 6379) - Caching and session management
- **Prometheus** (Port 9090) - Metrics collection
- **Grafana** (Port 3001) - Monitoring dashboards

## 🚀 Quick Start

### Prerequisites

1. **Windows 10/11** with WSL2 (recommended) or native Windows
2. **Docker Desktop** with WSL2 backend
3. **NVIDIA GPU** with 6GB+ VRAM (RTX 3060 or better)
4. **NVIDIA Container Toolkit** for GPU support
5. **PowerShell 5.1+** or PowerShell Core

### Installation

1. **Clone and navigate to the project:**
```powershell
git clone <repository>
cd facesocial-project-final/backend-ai-services
```

2. **Deploy the infrastructure:**
```powershell
# Full deployment with image building
.\deploy.ps1 -Environment dev -BuildImages

# Or use pre-built images
.\deploy.ps1 -Environment dev -PullImages
```

3. **Verify deployment:**
```powershell
.\test-services.ps1 -Detailed
```

4. **Access services:**
   - API Gateway: http://localhost:8000
   - API Documentation: http://localhost:8000/docs
   - Grafana Dashboard: http://localhost:3001 (admin/admin123)

## 💾 VRAM Allocation Strategy

Optimized for RTX 3060 6GB with efficient memory distribution:

| Service | VRAM Limit | Purpose |
|---------|------------|---------|
| MediaPipe | 400MB | Fast screening |
| YOLOv10n | 800MB | Primary detection |
| MTCNN | 600MB | Precision analysis |
| Face Recognition | 1536MB | Ensemble matching |
| Anti-spoofing | 300MB | Liveness detection |
| Gender/Age | 400MB | Demographics |
| Deepfake | 2048MB | Manipulation detection |
| **Total** | **6084MB** | **Within 6GB limit** |

## 🔧 Management Commands

Use the management script for common operations:

```powershell
# Start all services
.\manage.ps1 start

# Check service status
.\manage.ps1 status

# View logs (with follow)
.\manage.ps1 logs -Service api-gateway -Follow

# Scale a service
.\manage.ps1 scale -Service yolo10-main-detection -Scale 2

# Monitor GPU usage
.\manage.ps1 gpu

# Create backup
.\manage.ps1 backup

# Open monitoring dashboard
.\manage.ps1 monitor
```

## 📊 API Usage Examples

### Single Image Analysis
```powershell
# Comprehensive analysis
curl -X POST "http://localhost:8000/analyze/comprehensive" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test-image/test_01.jpg"

# Individual service analysis
curl -X POST "http://localhost:8001/detect" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test-image/test_01.jpg"
```

### Batch Processing
```powershell
curl -X POST "http://localhost:8000/analyze/batch" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@test-image/test_01.jpg" \
  -F "files=@test-image/test_02.jpg" \
  -F "files=@test-image/test_03.jpg"
```

### Service Health Check
```powershell
curl "http://localhost:8000/health"
```

## 🧪 Testing

### Automated Testing Suite
```powershell
# Basic functionality tests
.\test-services.ps1

# Detailed output with performance metrics
.\test-services.ps1 -Detailed

# Load testing with concurrent requests
.\test-services.ps1 -LoadTest
```

### Manual Testing
1. Place test images in `../test-image/` directory
2. Use API documentation at http://localhost:8000/docs
3. Monitor performance in Grafana dashboard

## 📈 Monitoring & Analytics

### Grafana Dashboard Features
- **GPU Utilization**: Real-time VRAM and GPU usage
- **Service Performance**: Response times and throughput
- **Error Rates**: Failed requests and health status
- **Database Metrics**: PostgreSQL performance
- **System Resources**: CPU, memory, and disk usage

### Prometheus Metrics
- Custom AI service metrics
- NVIDIA GPU metrics via nvidia-smi-exporter
- Container and system metrics
- API Gateway performance metrics

### Health Monitoring
Each service provides comprehensive health endpoints:
```json
{
  "status": "healthy",
  "service": "mediapipe-predetection",
  "version": "1.0.0",
  "gpu_memory_used_mb": 384.5,
  "gpu_memory_limit_mb": 400,
  "cpu_usage_percent": 15.2,
  "uptime_seconds": 3600,
  "requests_processed": 1234,
  "average_response_time_ms": 45.3
}
```

## 🗄️ Database Schema

Comprehensive PostgreSQL schema with pgvector support:

### Core Tables
- **users**: User management and authentication
- **faces**: Face embeddings and metadata
- **analysis_sessions**: Image processing sessions
- **face_detections**: Detection results from all services
- **face_recognitions**: Identity matching results
- **antispoof_results**: Liveness detection results
- **gender_age_results**: Demographic analysis
- **deepfake_results**: Manipulation detection results

### Performance Tables
- **service_metrics**: Service performance data
- **health_logs**: System health monitoring
- **api_usage**: API analytics and usage tracking

### Advanced Features
- **Vector similarity search** for face matching
- **Automated indexing** for performance optimization
- **Real-time analytics views**
- **Comprehensive audit trail**

## 🔒 Security Features

### Data Protection
- Password hashing with bcrypt
- JWT token authentication
- SQL injection prevention
- Input validation and sanitization

### Network Security
- Internal Docker network isolation
- Service-to-service authentication
- Rate limiting and request throttling
- CORS configuration

### Monitoring Security
- Health check endpoints with authentication
- Secure metrics collection
- Access logging and audit trails

## 🎯 Performance Optimization

### GPU Memory Management
- **Dynamic memory allocation** based on workload
- **Memory pooling** to prevent fragmentation
- **Automatic cleanup** of unused resources
- **VRAM limit enforcement** per service

### Processing Optimization
- **Batch processing** for multiple images
- **Async processing** with FastAPI
- **Connection pooling** for database access
- **Redis caching** for frequent queries

### Scalability Features
- **Horizontal scaling** for AI services
- **Load balancing** through API Gateway
- **Database connection pooling**
- **Microservice architecture** for independent scaling

## 🛠️ Troubleshooting

### Common Issues

1. **GPU not detected:**
```powershell
# Check NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:12.9.0-base-ubuntu22.04 nvidia-smi
```

2. **Service startup failures:**
```powershell
# Check service logs
.\manage.ps1 logs -Service <service-name>

# Check GPU memory usage
.\manage.ps1 gpu
```

3. **Database connection issues:**
```powershell
# Check PostgreSQL status
docker compose exec postgres pg_isready -U facesocial -d facesocial_ai
```

4. **Performance issues:**
```powershell
# Monitor resource usage
.\manage.ps1 status

# Check Grafana dashboard
.\manage.ps1 monitor
```

### Log Locations
- Service logs: `docker compose logs <service>`
- Deployment logs: `deployment-*.log`
- Test results: `test-results-*.json`
- Database logs: `docker compose logs postgres`

## 📋 Maintenance

### Regular Tasks
1. **Weekly**: Database backup using `.\manage.ps1 backup`
2. **Monthly**: Update services with `.\manage.ps1 update`
3. **Quarterly**: Review and optimize VRAM allocation
4. **As needed**: Scale services based on load

### Backup Strategy
- **Database backups**: Automated PostgreSQL dumps
- **Configuration backups**: Docker Compose and config files
- **Model backups**: AI model files and weights
- **Monitoring data**: Prometheus and Grafana data

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Test thoroughly with the test suite
4. Submit a pull request with detailed description

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **NVIDIA** for CUDA and GPU acceleration support
- **FastAPI** for high-performance API framework
- **Docker** for containerization platform
- **PostgreSQL** and **pgvector** for vector database support
- **Prometheus** and **Grafana** for monitoring infrastructure

---

**🚀 Ready to deploy your FaceSocial AI infrastructure? Start with `.\deploy.ps1` and begin analyzing faces in minutes!**
