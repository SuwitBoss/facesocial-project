# 🤖 FaceSocial AI Models & Test Images Download

เนื่องจากไฟล์ AI Models และรูปภาพทดสอบมีขนาดใหญ่ (รวม ~1.0 GB) จึงไม่ได้รวมอยู่ใน Git repository นี้

## 📦 ไฟล์ที่ต้องดาวน์โหลดเพิ่มเติม

### 🧠 AI Models (ขนาดรวม ~900 MB)

**Face Recognition Models:**
- `adaface_ir101.onnx` (249 MB) - โมเดลหลักสำหรับ Face Recognition
- `arcface_r100.onnx` (249 MB) - โมเดลสำรองสำหรับ Face Recognition 
- `facenet_vggface2.onnx` (89 MB) - โมเดล lightweight สำหรับ Face Recognition

**Deepfake Detection Models:**
- `model.onnx` (44 MB) - โมเดลตรวจจับ Deepfake

**Face Detection Models:**
- `yolov5s-face.onnx` (27 MB) - YOLO Face Detection โมเดล
- `yolov10n-face.onnx` (9 MB) - YOLO v10 Face Detection โมเดล

**Anti-Spoofing Models:**
- `AntiSpoofing_bin_1.5_128.onnx` (1.8 MB) - Anti-Spoofing Binary
- `AntiSpoofing_print-replay_1.5_128.onnx` (1.8 MB) - Anti-Spoofing Print/Replay

**Gender & Age Detection:**
- `genderage.onnx` (1.3 MB) - Gender และ Age Detection โมเดล

### 🖼️ Test Images (~200 MB)

**Test Images สำหรับทดสอบระบบ:**
- `test-image/fake_*.jpg` - ภาพปลอม (deepfake)
- `test-image/real_*.jpg` - ภาพจริง
- `test-image/spoof_*.jpg` - ภาพหลอกลวง
- `test-image/group_*.jpg` - ภาพกลุ่มคน
- `test-image/test_*.jpg` - ภาพทดสอบทั่วไป

## 📥 วิธีการดาวน์โหลด

### ตัวเลือกที่ 1: ดาวน์โหลดจาก Google Drive
```bash
# สำหรับ Windows PowerShell
Invoke-WebRequest -Uri "https://drive.google.com/download?id=YOUR_FILE_ID" -OutFile "models.zip"

# แตกไฟล์
Expand-Archive -Path "models.zip" -DestinationPath "./"
```

### ตัวเลือกที่ 2: ดาวน์โหลดจาก Hugging Face
```bash
# ติดตั้ง huggingface-hub
pip install huggingface-hub

# ดาวน์โหลด models
huggingface-cli download SuwitBoss/facesocial-models --local-dir ./model/
```

### ตัวเลือกที่ 3: ดาวน์โหลดจาก Release
ไปที่ [Releases](https://github.com/SuwitBoss/facesocial-project/releases) และดาวน์โหลด:
- `facesocial-models.zip`
- `facesocial-test-images.zip`

## 🔧 การติดตั้งหลังดาวน์โหลด

1. **แตกไฟล์ models และ images ไปยัง directory ที่ถูกต้อง:**
```bash
# ตัวอย่างโครงสร้างไฟล์
facesocial-project/
├── model/                    # AI Models ที่ดาวน์โหลด
│   ├── face-recognition/
│   ├── deepfake-detection/
│   ├── face-detection/
│   ├── anti-spoofing/
│   └── gender-age/
├── test-image/              # Test Images ที่ดาวน์โหลด
│   ├── fake_0.jpg
│   ├── real_0.jpg
│   └── ...
└── backend-ai-services/     # โค้ดหลัก (มีอยู่ใน repo นี้)
```

2. **ตรวจสอบการติดตั้ง:**
```bash
# รันการทดสอบ
python test_complete_system.py
```

## ⚠️ ข้อกำหนดระบบ

- **พื้นที่ว่าง:** ต้องการพื้นที่ว่างอย่างน้อย 2 GB
- **RAM:** แนะนำ 8 GB ขึ้นไป
- **GPU:** ไม่จำเป็น แต่จะช่วยเพิ่มประสิทธิภาพ

## 🆘 การแก้ไขปัญหา

**ปัญหา: ไฟล์ model ไม่ถูกต้อง**
```bash
# ตรวจสอบ checksum
certutil -hashfile model/face-recognition/adaface_ir101.onnx SHA256
```

**ปัญหา: เมมอรี่ไม่พอ**
```bash
# ใช้ quantized models (ขนาดเล็กกว่า)
# ใช้ model_int8.onnx แทน model.onnx
```

## 📞 ติดต่อสอบถาม

หากมีปัญหาในการดาวน์โหลดหรือติดตั้ง กรุณาสร้าง [GitHub Issue](https://github.com/SuwitBoss/facesocial-project/issues)
