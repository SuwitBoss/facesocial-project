from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
from datetime import datetime
import logging
import asyncio
from typing import Optional

# Import our utilities
from utils.face_detector import YOLOv5FaceDetector
from utils.image_utils import bytes_to_opencv, draw_faces_on_image, encode_image_to_base64

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="FaceSocial Face Recognition Service",
    description="AI-powered face recognition service with YOLOv5 face detection",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global face detector instance
face_detector = None

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global face_detector
    try:
        logger.info("Loading Face Detection model...")
        # ตรวจสอบ path ของโมเดล
        model_path = "/app/models/yolov5s-face.onnx"
        logger.info(f"Model path: {model_path}")
        
        # ตรวจสอบว่าไฟล์มีอยู่หรือไม่
        import os
        if os.path.exists(model_path):
            logger.info("Model file exists!")
        else:
            logger.error(f"Model file not found at {model_path}")
            # ลองหา path อื่น
            for root, dirs, files in os.walk("/app"):
                for file in files:
                    if file == "yolov5s-face.onnx":
                        found_path = os.path.join(root, file)
                        logger.info(f"Found model at: {found_path}")
        
        face_detector = YOLOv5FaceDetector(model_path)
        logger.info("Face Detection model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load Face Detection model: {e}")
        # Don't fail startup, just log the error

@app.get("/")
def read_root():
    return {
        "service": "Face Recognition Service",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": {
            "face_detector": face_detector is not None
        },
        "endpoints": [
            "GET /health",
            "GET /test-milvus",
            "GET /test-redis",
            "POST /face/detect",
            "POST /face/detect-with-image",
            "POST /face/recognize",
            "POST /face/register"
        ]
    }

@app.get("/health")
def health_check():
    return {
        "status": "OK",
        "service": "Face Recognition Service",
        "timestamp": datetime.now().isoformat(),
        "models_status": {
            "face_detector": "loaded" if face_detector is not None else "not_loaded"
        }
    }

@app.get("/test-milvus")
async def test_milvus():
    """ทดสอบการเชื่อมต่อ Milvus"""
    try:
        from pymilvus import connections
        
        # เชื่อมต่อ Milvus
        connections.connect(
            alias="default",
            host="milvus",
            port="19530"
        )
        
        return {
            "status": "success",
            "message": "Milvus connection successful (basic)",
            "host": "milvus:19530"
        }
        
    except Exception as e:
        logger.error(f"Milvus connection failed: {e}")
        return {
            "status": "warning",
            "message": f"Milvus connection issue (will implement later): {str(e)}",
            "host": "milvus:19530"
        }

@app.get("/test-redis")
async def test_redis():
    """ทดสอบการเชื่อมต่อ Redis"""
    try:
        import redis
        
        r = redis.Redis(host='redis', port=6379, decode_responses=True)
        r.ping()
        
        return {
            "status": "success", 
            "message": "Redis connection successful",
            "host": "redis:6379"
        }
        
    except Exception as e:
        logger.error(f"Redis connection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Redis connection failed: {str(e)}")

@app.post("/face/detect")
async def detect_faces(file: UploadFile = File(...), 
                      confidence_threshold: Optional[float] = 0.7,
                      return_image: Optional[bool] = False):
    """
    ตรวจจับใบหน้าในรูปภาพ
    
    Parameters:
    - file: ไฟล์รูปภาพ
    - confidence_threshold: ค่าความมั่นใจขั้นต่ำ (0.1-0.9)
    - return_image: คืนค่ารูปภาพที่มีกรอบใบหน้าหรือไม่
    """
    try:
        # ตรวจสอบว่าโหลด face detector แล้วหรือไม่
        if face_detector is None:
            raise HTTPException(status_code=503, detail="Face detection model not loaded")
        
        # ตรวจสอบประเภทไฟล์
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # ตรวจสอบ confidence threshold
        if not (0.1 <= confidence_threshold <= 0.9):
            confidence_threshold = 0.7
        
        # อ่านข้อมูลไฟล์
        contents = await file.read()
        
        # แปลงเป็น OpenCV image
        image = bytes_to_opencv(contents)
        
        # ตรวจจับใบหน้า
        detection_result = face_detector.detect_faces(image, conf_threshold=confidence_threshold)
        
        if not detection_result["success"]:
            raise HTTPException(status_code=500, detail=f"Face detection failed: {detection_result.get('error', 'Unknown error')}")
        
        # สร้าง response
        response = {
            "success": True,
            "filename": file.filename,
            "file_size": len(contents),
            "content_type": file.content_type,
            "faces_detected": detection_result["faces_count"],
            "faces": detection_result["faces"],
            "confidence_threshold": confidence_threshold,
            "processing_info": {
                "image_dimensions": {
                    "height": image.shape[0],
                    "width": image.shape[1],
                    "channels": image.shape[2]
                }
            }
        }
        
        # เพิ่มรูปภาพที่มีกรอบใบหน้าถ้าต้องการ
        if return_image and detection_result["faces_count"] > 0:
            image_with_faces = draw_faces_on_image(image, detection_result["faces"])
            response["image_with_faces"] = encode_image_to_base64(image_with_faces)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Face detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Face detection failed: {str(e)}")

@app.post("/detect")
async def detect_faces_proxy(file: UploadFile = File(...), 
                            confidence_threshold: Optional[float] = 0.7,
                            return_image: Optional[bool] = False):
    return await detect_faces(file, confidence_threshold, return_image)

@app.post("/detect-with-image")
async def detect_faces_with_image_proxy(file: UploadFile = File(...), 
                                 confidence_threshold: Optional[float] = 0.7):
    return await detect_faces(file, confidence_threshold, return_image=True)

@app.post("/face/detect-with-image")
async def detect_faces_with_image(file: UploadFile = File(...), 
                                 confidence_threshold: Optional[float] = 0.7):
    """
    ตรวจจับใบหน้าและคืนค่ารูปภาพที่มีกรอบใบหน้า
    """
    return await detect_faces(file, confidence_threshold, return_image=True)

@app.post("/face/register")
async def register_face(file: UploadFile = File(...)):
    """
    ลงทะเบียนใบหน้า (placeholder - จะพัฒนาต่อไป)
    """
    try:
        if face_detector is None:
            raise HTTPException(status_code=503, detail="Face detection model not loaded")
        
        # ตรวจสอบประเภทไฟล์
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # อ่านและตรวจจับใบหน้า
        contents = await file.read()
        image = bytes_to_opencv(contents)
        detection_result = face_detector.detect_faces(image, conf_threshold=0.7)
        
        if detection_result["faces_count"] == 0:
            raise HTTPException(status_code=400, detail="No face detected in image")
        elif detection_result["faces_count"] > 1:
            raise HTTPException(status_code=400, detail="Multiple faces detected. Please provide image with single face")
        
        # TODO: สกัด face embedding และบันทึกลง Milvus
        
        return {
            "success": True,
            "message": "Face registration endpoint (will implement face embedding extraction)",
            "filename": file.filename,
            "face_detected": True,
            "face_info": detection_result["faces"][0]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Face registration failed: {e}")
        raise HTTPException(status_code=500, detail=f"Face registration failed: {str(e)}")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 3002))
    uvicorn.run(app, host="0.0.0.0", port=port)