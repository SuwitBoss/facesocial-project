from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import os
from datetime import datetime
import logging
import numpy as np
import cv2
from typing import Optional, List
import json
from src.utils.gender_age_detector import GenderAgeDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Gender & Age Detection Service",
    description="AI-powered gender and age detection service",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

gender_age_detector = None

@app.on_event("startup")
async def startup_event():
    global gender_age_detector
    try:
        logger.info("Loading Gender & Age Detection model...")
        model_path = "/app/models/genderage.onnx"
        if os.path.exists(model_path):
            gender_age_detector = GenderAgeDetector(model_path)
            logger.info("Gender & Age Detection model loaded successfully!")
        else:
            logger.error(f"Model not found at {model_path}")
            model_dir = "/app/models"
            if os.path.exists(model_dir):
                files = os.listdir(model_dir)
                logger.info(f"Available files in {model_dir}: {files}")
            else:
                logger.error(f"Models directory {model_dir} does not exist")
    except Exception as e:
        logger.error(f"Failed to load Gender & Age Detection model: {e}")

def bytes_to_opencv(image_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image from bytes")
    return image

@app.get("/")
def read_root():
    return {
        "service": "Gender & Age Detection Service",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": {
            "gender_age_detector": gender_age_detector is not None
        },
        "endpoints": [
            "GET /health",
            "POST /demographics/analyze",
            "POST /demographics/analyze-batch",
            "POST /demographics/face-analysis"
        ]
    }

@app.get("/health")
def health_check():
    return {
        "status": "OK",
        "service": "Gender & Age Detection Service",
        "timestamp": datetime.now().isoformat(),
        "models_status": {
            "gender_age_detector": "loaded" if gender_age_detector is not None else "not_loaded"
        }
    }

@app.post("/demographics/analyze")
async def analyze_demographics(
    file: UploadFile = File(...),
    return_face_info: Optional[bool] = Form(default=False)
):
    try:
        if gender_age_detector is None:
            raise HTTPException(status_code=503, detail="Gender & Age detection model not loaded")
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        contents = await file.read()
        image = bytes_to_opencv(contents)
        result = gender_age_detector.detect_gender_age(image)
        response = {
            "success": True,
            "filename": file.filename,
            "file_size": len(contents),
            "content_type": file.content_type,
            "analysis": result,
            "analysis_timestamp": datetime.utcnow().isoformat()
        }
        if return_face_info:
            response["image_info"] = {
                "dimensions": {
                    "height": image.shape[0],
                    "width": image.shape[1],
                    "channels": image.shape[2]
                }
            }
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Demographics analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/demographics/analyze-batch")
async def analyze_demographics_batch(
    files: List[UploadFile] = File(...),
    include_summary: Optional[bool] = Form(default=True)
):
    try:
        if gender_age_detector is None:
            raise HTTPException(status_code=503, detail="Gender & Age detection model not loaded")
        if len(files) > 20:
            raise HTTPException(status_code=400, detail="Maximum 20 files per batch")
        results = []
        face_images = []
        for i, file in enumerate(files):
            try:
                if not file.content_type or not file.content_type.startswith('image/'):
                    results.append({
                        "filename": file.filename,
                        "error": "Invalid file type",
                        "success": False
                    })
                    continue
                contents = await file.read()
                image = bytes_to_opencv(contents)
                face_images.append(image)
                analysis = gender_age_detector.detect_gender_age(image)
                results.append({
                    "filename": file.filename,
                    "file_index": i,
                    "file_size": len(contents),
                    "analysis": analysis,
                    "success": True
                })
            except Exception as e:
                results.append({
                    "filename": file.filename,
                    "error": str(e),
                    "success": False
                })
        response = {
            "success": True,
            "total_files": len(files),
            "successful_analyses": sum(1 for r in results if r.get("success", False)),
            "results": results,
            "analysis_timestamp": datetime.utcnow().isoformat()
        }
        if include_summary and face_images:
            successful_results = [r["analysis"] for r in results if r.get("success", False)]
            if successful_results:
                response["demographics_summary"] = gender_age_detector.get_demographics_summary(successful_results)
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch demographics analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")

@app.post("/demographics/face-analysis")
async def analyze_face_with_demographics(
    file: UploadFile = File(...),
    detect_faces: Optional[bool] = Form(default=True),
    face_confidence_threshold: Optional[float] = Form(default=0.7)
):
    try:
        if gender_age_detector is None:
            raise HTTPException(status_code=503, detail="Gender & Age detection model not loaded")
        contents = await file.read()
        image = bytes_to_opencv(contents)
        result = gender_age_detector.detect_gender_age(image)
        response = {
            "success": True,
            "filename": file.filename,
            "image_info": {
                "dimensions": {
                    "height": image.shape[0],
                    "width": image.shape[1],
                    "channels": image.shape[2]
                }
            },
            "faces_detected": 1,
            "faces_analysis": [
                {
                    "face_id": 0,
                    "face_region": "entire_image",
                    "demographics": result
                }
            ],
            "analysis_timestamp": datetime.utcnow().isoformat()
        }
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Face analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Face analysis failed: {str(e)}")

@app.get("/demographics/stats")
async def get_demographics_stats():
    return {
        "service": "Gender & Age Detection Service",
        "model_info": {
            "model_type": "ONNX",
            "input_size": "224x224" if gender_age_detector else "Unknown",
            "outputs": ["gender", "age"]
        },
        "capabilities": [
            "Gender classification (Male/Female)",
            "Age estimation (0-100 years)",
            "Batch processing",
            "Demographics summary statistics"
        ],
        "supported_formats": ["jpg", "jpeg", "png", "bmp"],
        "max_batch_size": 20
    }

@app.post("/analyze")
async def analyze_alias(
    file: UploadFile = File(...),
    return_face_info: Optional[bool] = Form(default=False)
):
    return await analyze_demographics(file, return_face_info)

@app.post("/analyze-batch")
async def analyze_batch_alias(
    files: List[UploadFile] = File(...),
    include_summary: Optional[bool] = Form(default=True)
):
    return await analyze_demographics_batch(files, include_summary)

@app.options("/analyze")
@app.options("/analyze-batch")
@app.options("/demographics/analyze")
@app.options("/demographics/analyze-batch")
async def options_handlers(request: Request):
    return JSONResponse(content={}, status_code=200)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)
