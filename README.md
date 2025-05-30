# FaceSocial Project

🎯 **AI-Powered Social Media Platform with Advanced Face Recognition & Security**

A comprehensive social media platform integrated with multiple AI services for face recognition, anti-spoofing, deepfake detection, and demographic analysis.

## 🚀 Features

### 🤖 AI Services
- **Face Recognition**: Advanced face detection and recognition using FaceNet
- **Anti-Spoofing**: Real-time detection of fake/spoofed faces
- **Deepfake Detection**: AI-powered deepfake content identification
- **Face Detection**: Multiple algorithms (MTCNN, YOLO, MediaPipe)
- **Gender & Age Detection**: Demographic analysis from facial features

### 🏗️ Architecture
- **Microservices**: Independent AI services with Docker containerization
- **API Gateway**: Centralized routing and load balancing
- **Real-time Processing**: Async processing with high performance
- **Monitoring**: Comprehensive logging and metrics with Prometheus/Grafana

## 📂 Project Structure

```
facesocial-project-final/
├── backend-ai-services/           # AI Microservices
│   ├── face-recognition/          # Face recognition service
│   ├── antispoof/                 # Anti-spoofing service
│   ├── deepfake/                  # Deepfake detection service
│   ├── gender-age/                # Gender & age detection
│   ├── mediapipe/                 # MediaPipe face detection
│   ├── mtcnn/                     # MTCNN face detection
│   ├── yolo-detection/            # YOLO face detection
│   ├── api-gateway/               # API Gateway service
│   ├── monitoring/                # Prometheus & Grafana configs
│   └── docker-compose.yml         # Container orchestration
├── doc/                           # Complete documentation
│   ├── backend-ai-services/       # Technical documentation
│   ├── frontend-pages/            # UI/UX specifications
│   └── model/                     # AI model documentation
├── model/                         # AI model files and configs
├── test-image/                    # Test images for AI services
└── test_complete_system.py        # Comprehensive system testing
```

## 🛠️ Technology Stack

### Backend AI Services
- **Python 3.11+**
- **FastAPI**: High-performance async web framework
- **TensorFlow/PyTorch**: Deep learning frameworks
- **OpenCV**: Computer vision processing
- **MediaPipe**: Google's ML framework
- **ONNX Runtime**: Optimized model inference

### Infrastructure
- **Docker & Docker Compose**: Containerization
- **Nginx**: Reverse proxy and load balancing
- **PostgreSQL**: Primary database
- **Redis**: Caching and session management
- **Prometheus & Grafana**: Monitoring and analytics

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.11+
- Git

### 1. Clone Repository
```bash
git clone https://github.com/SuwitBoss/facesocial-project.git
cd facesocial-project
```

### 2. Download AI Models & Test Images
⚠️ **ไฟล์ AI Models และรูปภาพทดสอบไม่ได้รวมใน repository นี้เนื่องจากมีขนาดใหญ่ (1.0+ GB)**

📥 **ดูวิธีการดาวน์โหลดได้ที่:** [MODELS_DOWNLOAD.md](MODELS_DOWNLOAD.md)

**ไฟล์ที่ต้องดาวน์โหลดเพิ่มเติม:**
- 🧠 **AI Models** (~1.0 GB): Face Recognition, Deepfake Detection, Anti-Spoofing, Gender-Age
- 🖼️ **Test Images** (~50 MB): ภาพทดสอบสำหรับแต่ละ AI service
- 📦 **Model Configs**: Configuration files สำหรับแต่ละโมเดล

**ตัวเลือกการดาวน์โหลด:**
- 📦 **GitHub Releases**: [Download from Releases](https://github.com/SuwitBoss/facesocial-project/releases) (แนะนำ)
- 🔗 **Direct Download**: ลิงก์ดาวน์โหลดโดยตรงใน [MODELS_DOWNLOAD.md](MODELS_DOWNLOAD.md)
- 🤗 **Hugging Face**: [SuwitBoss/facesocial-models](https://huggingface.co/SuwitBoss/facesocial-models)

### 3. Start AI Services
```bash
cd backend-ai-services
docker-compose up -d
```

### 4. Verify Services
```bash
# Check all services are running
docker-compose ps

# Test individual services
curl http://localhost:5001/health  # Face Recognition
curl http://localhost:5002/health  # Anti-Spoofing
curl http://localhost:5003/health  # Deepfake Detection
curl http://localhost:8080/health  # API Gateway
```

### 5. Run System Tests
```bash
cd ..
pip install -r requirements.txt
python test_complete_system.py
```

## 📋 Service Endpoints

| Service | Port | Health Check | Main Endpoint |
|---------|------|--------------|---------------|
| Face Recognition | 5001 | `/health` | `/recognize`, `/register`, `/compare` |
| Anti-Spoofing | 5002 | `/health` | `/detect_spoof` |
| Deepfake Detection | 5003 | `/health` | `/detect_deepfake` |
| MTCNN Detection | 5004 | `/health` | `/detect_faces` |
| Gender & Age | 5005 | `/health` | `/analyze` |
| MediaPipe | 5006 | `/health` | `/detect_faces` |
| YOLO Detection | 5007 | `/health` | `/detect_faces` |
| API Gateway | 8080 | `/health` | `/api/*` |

## 🧪 Testing

### Comprehensive System Test
```bash
python test_complete_system.py
```

The test suite includes:
- ✅ Service health checks
- ✅ Face registration and recognition
- ✅ Anti-spoofing validation
- ✅ Deepfake detection accuracy
- ✅ Multi-algorithm face detection
- ✅ Gender & age analysis
- ✅ API Gateway routing
- ✅ End-to-end pipeline testing
- ✅ Performance metrics

### Test Images
Located in `test-image/` directory:
- `real_*.jpg` - Authentic face images
- `fake_*.jpg` - Synthetic/fake images  
- `spoof_*.jpg` - Spoofed face images
- `group_*.jpg` - Multiple faces
- `test_*.jpg` - General test images

## 📊 Performance Metrics

- **Face Recognition**: < 1.5s response time
- **Anti-Spoofing**: < 0.8s response time
- **Deepfake Detection**: < 2.0s response time
- **Face Detection**: < 0.5s response time
- **Gender & Age**: < 1.0s response time

## 🔧 Configuration

### Environment Variables
Create `.env` file in `backend-ai-services/`:
```env
# Database
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=facesocial
POSTGRES_USER=admin
POSTGRES_PASSWORD=password

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# Services
FACE_RECOGNITION_PORT=5001
ANTISPOOF_PORT=5002
DEEPFAKE_PORT=5003
```

### Model Configuration
AI models are automatically downloaded on first run or can be pre-loaded:
- FaceNet model for face recognition
- Anti-spoofing CNN models
- Deepfake detection models
- YOLO face detection weights

## 📚 Documentation

### Technical Documentation
- [AI Services Architecture](doc/backend-ai-services/01-ai-services-architecture.md)
- [Face Recognition Service](doc/backend-ai-services/03-face-recognition-service.md)
- [Anti-Spoofing Service](doc/backend-ai-services/04-antispoofing-service.md)
- [Deepfake Detection](doc/backend-ai-services/05-deepfake-detection-service.md)
- [Infrastructure Setup](doc/backend-ai-services/08-infrastructure-configuration.md)

### Frontend Specifications
- [UI/UX Flow](doc/frontend-pages/16-ui-flow-navigation.md)
- [AI Features Hub](doc/frontend-pages/10-ai-features-hub.md)
- [Dashboard Design](doc/frontend-pages/06-dashboard.md)

## 🐛 Troubleshooting

### Common Issues

**Services not starting:**
```bash
# Check Docker status
docker-compose logs

# Restart specific service
docker-compose restart face-recognition
```

**Memory issues:**
```bash
# Increase Docker memory limit
# Or use lazy loading in services
```

**Port conflicts:**
```bash
# Check port usage
netstat -tulpn | grep :5001

# Change ports in docker-compose.yml
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **SuwitBoss** - *Initial work* - [SuwitBoss](https://github.com/SuwitBoss)

## 🙏 Acknowledgments

- TensorFlow and PyTorch communities
- OpenCV contributors
- MediaPipe team at Google
- MTCNN and YOLO researchers
- FastAPI framework developers

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Email: [your-email@example.com]
- Documentation: [Project Wiki](https://github.com/SuwitBoss/facesocial-project/wiki)

---

**⭐ Star this repo if you find it helpful!**

Last Updated: May 31, 2025
