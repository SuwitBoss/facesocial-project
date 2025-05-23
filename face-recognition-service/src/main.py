from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
from datetime import datetime
import logging
import asyncio
from typing import Optional
import uuid
import numpy as np
import cv2

# Import our utilities
from utils.face_detector import YOLOv5FaceDetector
from utils.image_utils import bytes_to_opencv, draw_faces_on_image, encode_image_to_base64
from utils.face_embedder import FaceEmbedder
from utils.milvus_manager import MilvusManager
from utils.anti_spoofing import AntiSpoofingDetector, LivenessDetector
from glasses_detector import AnyglassesClassifier

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
face_embedder = None
milvus_manager = None
anti_spoofing_detector = None
liveness_detector = None
glasses_classifier = None

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global face_detector, face_embedder, milvus_manager, anti_spoofing_detector, liveness_detector, glasses_classifier
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

    # Load face embedding models
    try:
        logger.info("Loading Face Embedding models...")
        model_configs = {
            "adaface": {"path": "/app/models/adaface_ir101.onnx", "input_size": (112, 112)},
            "arcface": {"path": "/app/models/arcface_r100.onnx", "input_size": (112, 112)},
            "facenet": {"path": "/app/models/facenet_vggface2.onnx", "input_size": (160, 160)}
        }
        face_embedder = FaceEmbedder(model_configs)
        logger.info("Face Embedding models loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load Face Embedding models: {e}")

    # Initialize Milvus
    try:
        logger.info("Connecting to Milvus...")
        milvus_manager = MilvusManager()
        logger.info("Milvus connected successfully!")
    except Exception as e:
        logger.error(f"Failed to connect to Milvus: {e}")

    # Load Anti-Spoofing model
    try:
        logger.info("Loading Anti-Spoofing model...")
        anti_spoofing_detector = AntiSpoofingDetector("/app/models/AntiSpoofing_bin_1.5_128.onnx")
        logger.info("Anti-Spoofing model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load Anti-Spoofing model: {e}")

    # Initialize Liveness Detector
    try:
        liveness_detector = LivenessDetector()
        logger.info("Liveness Detector initialized!")
    except Exception as e:
        logger.error(f"Failed to initialize Liveness Detector: {e}")

    # Load Glasses Detector
    try:
        logger.info("Loading Glasses Detector model...")
        glasses_classifier = AnyglassesClassifier()  # ไม่ต้องใส่ device
        logger.info("Glasses Detector loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load Glasses Detector: {e}")
        glasses_classifier = None

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
async def register_face(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    model_type: str = Form(default="ensemble")
):
    """Register a face for a user"""
    try:
        if face_detector is None:
            raise HTTPException(status_code=503, detail="Face detection model not loaded")
        # ตรวจสอบประเภทไฟล์
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        contents = await file.read()
        image = bytes_to_opencv(contents)
        # ใช้ confidence threshold สูงขึ้นสำหรับ registration
        detection_result = face_detector.detect_faces(image, conf_threshold=0.8)
        logger.info(f"Face detection result: {detection_result['faces_count']} faces detected")
        for i, face in enumerate(detection_result["faces"]):
            logger.info(f"Face {i+1}: confidence={face['confidence']:.3f}, bbox={face['bbox']}")
        if detection_result["faces_count"] == 0:
            raise HTTPException(status_code=400, detail="No face detected in image")
        elif detection_result["faces_count"] > 1:
            best_face = max(detection_result["faces"], key=lambda x: x["confidence"])
            logger.info(f"Multiple faces detected ({detection_result['faces_count']}), using face with highest confidence: {best_face['confidence']:.3f}")
            raise HTTPException(
                status_code=400,
                detail=f"Multiple faces detected ({detection_result['faces_count']} faces). Please provide image with single face or use face/register-best endpoint"
            )
        face_bbox = detection_result["faces"][0]["bbox"]
        face_image = image[face_bbox[1]:face_bbox[3], face_bbox[0]:face_bbox[2]]
        if face_embedder is None:
            raise HTTPException(status_code=503, detail="Face embedding models not loaded")
        if model_type == "ensemble":
            embedding = face_embedder.extract_ensemble_embedding(face_image)
        else:
            embedding = face_embedder.extract_embedding(face_image, model_type)
        if embedding is None:
            raise HTTPException(status_code=500, detail="Failed to extract face embedding")
        face_id = str(uuid.uuid4())
        success = milvus_manager.insert_face(user_id, face_id, embedding, model_type)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to save face embedding")
        return {
            "success": True,
            "message": "Face registered successfully",
            "face_id": str(face_id),
            "user_id": user_id,
            "model_type": model_type,
            "face_confidence": float(detection_result["faces"][0]["confidence"])
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Face registration failed: {e}")
        raise HTTPException(status_code=500, detail=f"Face registration failed: {str(e)}")

@app.post("/face/register-best")
async def register_best_face(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    model_type: str = Form(default="ensemble")
):
    """Register the face with highest confidence score"""
    try:
        if face_detector is None:
            raise HTTPException(status_code=503, detail="Face detection model not loaded")
        contents = await file.read()
        image = bytes_to_opencv(contents)
        detection_result = face_detector.detect_faces(image, conf_threshold=0.8)
        if detection_result["faces_count"] == 0:
            detection_result = face_detector.detect_faces(image, conf_threshold=0.7)
        if detection_result["faces_count"] == 0:
            raise HTTPException(status_code=400, detail="No face detected in image")
        best_face = max(detection_result["faces"], key=lambda x: x["confidence"])
        face_bbox = best_face["bbox"]
        face_image = image[face_bbox[1]:face_bbox[3], face_bbox[0]:face_bbox[2]]
        if face_embedder is None:
            raise HTTPException(status_code=503, detail="Face embedding models not loaded")
        if model_type == "ensemble":
            embedding = face_embedder.extract_ensemble_embedding(face_image)
        else:
            embedding = face_embedder.extract_embedding(face_image, model_type)
        if embedding is None:
            raise HTTPException(status_code=500, detail="Failed to extract face embedding")
        face_id = str(uuid.uuid4())
        success = milvus_manager.insert_face(user_id, face_id, embedding, model_type)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to save face embedding")
        return {
            "success": True,
            "message": f"Face registered successfully (best of {int(detection_result['faces_count'])} detected)",
            "face_id": str(face_id),
            "user_id": user_id,
            "model_type": model_type,
            "face_confidence": float(best_face["confidence"]),
            "total_faces_detected": int(detection_result["faces_count"])
        }
    except Exception as e:
        logger.error(f"Face registration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/register")
async def register_face_alias(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    model_type: str = Form(default="ensemble")
):
    """Alias for Kong Gateway routing"""
    return await register_face(file, user_id, model_type)

@app.post("/register-best")
async def register_best_face_alias(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    model_type: str = Form(default="ensemble")
):
    """Alias for Kong Gateway routing"""
    return await register_best_face(file, user_id, model_type)

@app.post("/face/verify")
async def verify_face(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    threshold: float = Form(default=0.7)
):
    """Verify if the face belongs to a specific user"""
    try:
        if face_detector is None or face_embedder is None:
            raise HTTPException(status_code=503, detail="Models not loaded")
        contents = await file.read()
        image = bytes_to_opencv(contents)
        detection_result = face_detector.detect_faces(image, conf_threshold=0.7)
        if detection_result["faces_count"] == 0:
            raise HTTPException(status_code=400, detail="No face detected")
        best_face = max(detection_result["faces"], key=lambda x: x["confidence"])
        face_bbox = best_face["bbox"]
        face_image = image[face_bbox[1]:face_bbox[3], face_bbox[0]:face_bbox[2]]
        query_embedding = face_embedder.extract_ensemble_embedding(face_image)
        if query_embedding is None:
            raise HTTPException(status_code=500, detail="Failed to extract face embedding")
        search_results = milvus_manager.search_faces(query_embedding, top_k=10, threshold=0.0)
        user_matches = [r for r in search_results if r["user_id"] == user_id]
        if not user_matches:
            return {
                "verified": False,
                "user_id": user_id,
                "message": "No matching face found for this user",
                "confidence": 0.0
            }
        best_match = max(user_matches, key=lambda x: x["similarity"])
        # ตรวจจับแว่นตา (ใช้ Glasses Detector หรือ fallback)
        has_glasses = False
        try:
            has_glasses = has_glasses(face_image)
        except Exception as e:
            logger.warning(f"Glasses detection failed: {e}")
        # ปรับ threshold ถ้าพบแว่นตา
        adjusted_threshold = threshold
        auto_adjust = True  # สามารถตั้งค่าตรงนี้หรือรับจาก form ได้
        if has_glasses and auto_adjust:
            adjusted_threshold = threshold * 0.7  # ลดลง 30%
        is_verified = best_match["similarity"] >= adjusted_threshold
        return {
            "verified": bool(is_verified),
            "user_id": user_id,
            "similarity": float(best_match["similarity"]),
            "threshold": float(adjusted_threshold),
            "face_id": str(best_match["face_id"]),
            "message": "Face verified successfully" if is_verified else "Face verification failed"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Face verification failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/verify")
async def verify_face_alias(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    threshold: float = Form(default=0.7)
):
    return await verify_face(file, user_id, threshold)

@app.post("/face/identify")
async def identify_face(
    file: UploadFile = File(...),
    top_k: int = Form(default=5),
    threshold: float = Form(default=0.7)
):
    """Identify who the face belongs to"""
    try:
        if face_detector is None or face_embedder is None:
            raise HTTPException(status_code=503, detail="Models not loaded")
        contents = await file.read()
        image = bytes_to_opencv(contents)
        detection_result = face_detector.detect_faces(image, conf_threshold=0.7)
        if detection_result["faces_count"] == 0:
            raise HTTPException(status_code=400, detail="No face detected")
        best_face = max(detection_result["faces"], key=lambda x: x["confidence"])
        face_bbox = best_face["bbox"]
        face_image = image[face_bbox[1]:face_bbox[3], face_bbox[0]:face_bbox[2]]
        query_embedding = face_embedder.extract_ensemble_embedding(face_image)
        if query_embedding is None:
            raise HTTPException(status_code=500, detail="Failed to extract face embedding")
        search_results = milvus_manager.search_faces(query_embedding, top_k=top_k, threshold=threshold)
        if not search_results:
            return {
                "identified": False,
                "message": "No matching face found",
                "candidates": []
            }
        user_matches = {}
        for result in search_results:
            user_id = result["user_id"]
            if user_id not in user_matches or result["similarity"] > user_matches[user_id]["similarity"]:
                user_matches[user_id] = result
        candidates = sorted(user_matches.values(), key=lambda x: x["similarity"], reverse=True)
        # Convert all candidate fields to python native types
        candidates = [
            {
                **c,
                "similarity": float(c["similarity"]),
                "face_id": str(c["face_id"]),
                "user_id": str(c["user_id"])
            } for c in candidates
        ]
        best_match = candidates[0] if candidates else None
        # ตรวจจับแว่นตา (ใช้ Glasses Detector หรือ fallback)
        has_glasses = False
        try:
            has_glasses = has_glasses(face_image)
        except Exception as e:
            logger.warning(f"Glasses detection failed: {e}")
        # ปรับ threshold ถ้าพบแว่นตา
        adjusted_threshold = threshold
        auto_adjust = True  # สามารถตั้งค่าตรงนี้หรือรับจาก form ได้
        if has_glasses and auto_adjust:
            adjusted_threshold = threshold * 0.7  # ลดลง 30%
        return {
            "identified": True,
            "message": f"Found {len(candidates)} potential matches",
            "best_match": best_match,
            "candidates": candidates
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Face identification failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/identify")
async def identify_face_alias(
    file: UploadFile = File(...),
    top_k: int = Form(default=5),
    threshold: float = Form(default=0.7)
):
    return await identify_face(file, top_k, threshold)

@app.get("/face/stats")
async def get_face_stats():
    """Get statistics about registered faces"""
    try:
        from pymilvus import utility
        from pymilvus import Collection
        # Get collection stats
        stats = utility.get_query_segment_info("face_embeddings")
        collection = Collection("face_embeddings")
        return {
            "total_embeddings": collection.num_entities,
            "collection_name": "face_embeddings",
            "embedding_dimension": 512,
            "index_type": "IVF_FLAT"
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/stats")
async def get_face_stats_alias():
    return await get_face_stats()

@app.post("/face/check-liveness")
async def check_face_liveness(
    file: UploadFile = File(...),
    check_spoofing: bool = Form(default=True),
    check_texture: bool = Form(default=True)
):
    """Check if face is real (anti-spoofing and liveness detection)"""
    try:
        if face_detector is None:
            raise HTTPException(status_code=503, detail="Face detector not loaded")
        contents = await file.read()
        image = bytes_to_opencv(contents)
        detection_result = face_detector.detect_faces(image, conf_threshold=0.7)
        if detection_result["faces_count"] == 0:
            raise HTTPException(status_code=400, detail="No face detected")
        best_face = max(detection_result["faces"], key=lambda x: x["confidence"])
        face_bbox = best_face["bbox"]
        face_image = image[face_bbox[1]:face_bbox[3], face_bbox[0]:face_bbox[2]]
        result = {
            "face_detected": True,
            "face_confidence": float(best_face["confidence"])
        }
        # ตรวจจับแว่นตา (ใช้ Glasses Detector หรือ fallback Haar Cascade)
        has_glasses_flag = False
        try:
            has_glasses_flag = has_glasses(face_image)
        except Exception as e:
            logger.warning(f"Glasses detection failed: {e}")
        # ปรับ threshold ถ้าพบแว่นตา
        adjusted_confidence_threshold = 0.7
        if has_glasses_flag:
            adjusted_confidence_threshold = 0.7 * 0.7  # ลดลง 30%
        if check_spoofing and anti_spoofing_detector is not None:
            spoofing_result = anti_spoofing_detector.detect_spoofing(face_image)
            # Convert all numpy types to python types in spoofing_result
            spoofing_result = {
                k: (bool(v) if isinstance(v, (np.bool_, bool)) else float(v) if isinstance(v, (np.floating, float)) else v)
                for k, v in spoofing_result.items()
            }
            result["anti_spoofing"] = spoofing_result
        if check_texture and liveness_detector is not None:
            texture_result = liveness_detector.analyze_texture(face_image)
            texture_result = {
                k: (bool(v) if isinstance(v, (np.bool_, bool)) else float(v) if isinstance(v, (np.floating, float)) else v)
                for k, v in texture_result.items()
            }
            face_size_ok = liveness_detector.check_face_size(face_bbox, image.shape)
            result["liveness_checks"] = {
                "texture_analysis": texture_result,
                "face_size_adequate": bool(face_size_ok)
            }
        is_live = True
        confidence_scores = []
        if "anti_spoofing" in result:
            # ปรับ logic: ถ้า confidence > adjusted_confidence_threshold ให้ผ่าน แม้ is_real จะเป็น False
            spoofing_is_real = bool(result["anti_spoofing"]["is_real"])
            spoofing_conf = float(result["anti_spoofing"]["confidence"])
            is_live = is_live and (spoofing_is_real or spoofing_conf > adjusted_confidence_threshold)
            confidence_scores.append(spoofing_conf)
        if "liveness_checks" in result:
            is_live = is_live and bool(result["liveness_checks"]["face_size_adequate"])
            if result["liveness_checks"]["texture_analysis"]["is_sharp"]:
                confidence_scores.append(0.8)
            else:
                confidence_scores.append(0.3)
                # ผ่อนปรน: ถ้า has_glasses และ anti-spoofing ผ่าน ให้ไม่ตัด is_live เป็น False
                if not (has_glasses_flag and "anti_spoofing" in result and result["anti_spoofing"]["confidence"] > adjusted_confidence_threshold):
                    is_live = False
        result["is_live"] = bool(is_live)
        result["overall_confidence"] = float(np.mean(confidence_scores)) if confidence_scores else 0.0
        result["recommendation"] = "PASS" if is_live else "FAIL - Possible spoofing detected"
        result["has_glasses"] = bool(has_glasses_flag)
        result["adjusted_confidence_threshold"] = float(adjusted_confidence_threshold)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Liveness check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/check-liveness")
async def check_face_liveness_alias(
    file: UploadFile = File(...),
    check_spoofing: bool = Form(default=True),
    check_texture: bool = Form(default=True)
):
    return await check_face_liveness(file, check_spoofing, check_texture)

@app.post("/face/register-secure")
async def register_face_with_liveness(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    model_type: str = Form(default="ensemble"),
    require_liveness: bool = Form(default=True)
):
    """Register face with liveness check"""
    try:
        if require_liveness:
            liveness_result = await check_face_liveness(file, check_spoofing=True, check_texture=True)
            await file.seek(0)
            if not liveness_result["is_live"]:
                raise HTTPException(
                    status_code=400,
                    detail=f"Liveness check failed: {liveness_result['recommendation']}"
                )
        return await register_face(file, user_id, model_type)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Secure registration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/register-secure")
async def register_face_secure_alias(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    model_type: str = Form(default="ensemble"),
    require_liveness: bool = Form(default=True)
):
    return await register_face_with_liveness(file, user_id, model_type, require_liveness)

def has_glasses(face_img):
    """ตรวจจับแว่นตาด้วย Glasses Detector หรือ fallback เป็น Haar Cascade"""
    global glasses_classifier
    try:
        if glasses_classifier is not None:
            temp_path = "temp_face.jpg"
            cv2.imwrite(temp_path, face_img)
            result = glasses_classifier.predict(temp_path)
            os.remove(temp_path)
            return bool(result)
        else:
            return has_glasses_using_haar(face_img)
    except Exception as e:
        logger.warning(f"Glasses detection failed, fallback to Haar: {e}")
        return has_glasses_using_haar(face_img)

def has_glasses_using_haar(face_img):
    """ตรวจจับแว่นตาด้วย Haar Cascade (ใช้ path ที่ถูกต้องใน src/)"""
    cascade_path = os.path.join(os.path.dirname(__file__), 'haarcascade_eye_tree_eyeglasses.xml')
    eye_cascade = cv2.CascadeClassifier(cascade_path)
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    return len(eyes) >= 2

if __name__ == "__main__":
    port = int(os.getenv("PORT", 3002))
    uvicorn.run(app, host="0.0.0.0", port=port)