from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
import onnxruntime as ort
import numpy as np
import cv2
from PIL import Image
import io
import base64
import os
import logging
import redis
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Deepfake Detection Service")

# Global variables
deepfake_detector = None
redis_client = None

class DeepfakeDetector:
    def __init__(self, model_path, threshold=0.31):
        """Initialize Deepfake Detector with ONNX model"""
        try:
            # Check available providers
            providers = ['CPUExecutionProvider']
            if 'CUDAExecutionProvider' in ort.get_available_providers():
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            
            self.session = ort.InferenceSession(model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            self.threshold = threshold
            
            # Model expects 224x224 images
            self.input_height = 224
            self.input_width = 224
            
            logger.info(f"Deepfake Detector loaded successfully")
            logger.info(f"Provider: {self.session.get_providers()[0]}")
            logger.info(f"Threshold: {self.threshold}")
            
        except Exception as e:
            logger.error(f"Failed to load deepfake model: {str(e)}")
            raise

    def preprocess_image(self, image):
        """Preprocess image for model input"""
        # Resize to model input size
        img_resized = cv2.resize(image, (self.input_width, self.input_height))
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        img_normalized = img_rgb.astype(np.float32) / 255.0
        
        # Add batch dimension
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        return img_batch

    def predict(self, image):
        """Predict if image is deepfake"""
        try:
            # Preprocess image
            processed_img = self.preprocess_image(image)
            
            # Run inference
            outputs = self.session.run([self.output_name], {self.input_name: processed_img})
            probability = float(outputs[0][0][0])
            
            # Determine if fake
            is_fake = probability > self.threshold
            confidence = max(probability, 1 - probability)
            
            return {
                "is_deepfake": is_fake,
                "deepfake_probability": probability,
                "confidence": confidence,
                "threshold": self.threshold
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global deepfake_detector, redis_client
    
    try:
        # Load Deepfake model
        model_path = "/app/models/model.onnx"
        if os.path.exists(model_path):
            deepfake_detector = DeepfakeDetector(model_path)
            logger.info("Deepfake detector initialized")
        else:
            logger.error(f"Model not found at {model_path}")
        
        # Initialize Redis
        redis_client = redis.Redis(
            host=os.getenv('REDIS_HOST', 'redis'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            decode_responses=True
        )
        redis_client.ping()
        logger.info("Redis connected")
        
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")

@app.get("/")
async def root():
    return {"service": "Deepfake Detection Service", "status": "running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    health_status = {
        "status": "healthy",
        "service": "deepfake-detection",
        "models_loaded": {
            "deepfake_detector": deepfake_detector is not None
        },
        "redis_connected": False
    }
    
    # Check Redis
    try:
        if redis_client:
            redis_client.ping()
            health_status["redis_connected"] = True
    except:
        pass
    
    return health_status

@app.post("/deepfake/detect")
async def detect_deepfake(
    file: UploadFile = File(...),
    threshold: float = 0.31
):
    """Detect if uploaded image is a deepfake"""
    if not deepfake_detector:
        raise HTTPException(status_code=503, detail="Deepfake detector not initialized")
    
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Update threshold if provided
        original_threshold = deepfake_detector.threshold
        deepfake_detector.threshold = threshold
        
        # Detect deepfake
        result = deepfake_detector.predict(image)
        
        # Restore original threshold
        deepfake_detector.threshold = original_threshold
        
        # Add metadata
        result["filename"] = file.filename
        result["image_shape"] = image.shape
        result["analysis_timestamp"] = datetime.utcnow().isoformat()
        
        # Cache result if Redis available
        if redis_client:
            cache_key = f"deepfake:analysis:{file.filename}:{datetime.utcnow().timestamp()}"
            redis_client.setex(cache_key, 3600, json.dumps(result))
        
        return result
        
    except Exception as e:
        logger.error(f"Detection error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.options("/detect")
@app.options("/api/deepfake/detect")
async def options_detect(request: Request):
    """Handle CORS preflight for /detect and /api/deepfake/detect"""
    return JSONResponse(content={}, status_code=200)

@app.post("/detect")
async def detect_deepfake_alias(
    file: UploadFile = File(...),
    threshold: float = 0.31
):
    return await detect_deepfake(file, threshold)

@app.post("/api/deepfake/detect")
async def detect_deepfake_api_alias(
    file: UploadFile = File(...),
    threshold: float = 0.31
):
    return await detect_deepfake(file, threshold)

@app.post("/deepfake/detect-with-visualization")
async def detect_deepfake_with_visualization(
    file: UploadFile = File(...),
    threshold: float = 0.31
):
    """Detect deepfake and return visualization"""
    if not deepfake_detector:
        raise HTTPException(status_code=503, detail="Deepfake detector not initialized")
    
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Detect deepfake
        original_threshold = deepfake_detector.threshold
        deepfake_detector.threshold = threshold
        result = deepfake_detector.predict(image)
        deepfake_detector.threshold = original_threshold
        
        # Create visualization
        img_copy = image.copy()
        h, w = img_copy.shape[:2]
        
        # Add detection result
        color = (0, 0, 255) if result["is_deepfake"] else (0, 255, 0)
        label = "DEEPFAKE" if result["is_deepfake"] else "REAL"
        confidence_text = f"{label} ({result['confidence']:.2%})"
        
        # Add background rectangle for text
        cv2.rectangle(img_copy, (10, 10), (w-10, 60), (0, 0, 0), -1)
        cv2.putText(img_copy, confidence_text, (20, 45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
        
        # Add probability bar
        bar_width = int((w - 40) * result["deepfake_probability"])
        cv2.rectangle(img_copy, (20, 70), (w-20, 90), (100, 100, 100), -1)
        cv2.rectangle(img_copy, (20, 70), (20 + bar_width, 90), color, -1)
        cv2.putText(img_copy, f"Deepfake Probability: {result['deepfake_probability']:.2%}", 
                    (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Convert to base64
        _, buffer = cv2.imencode('.jpg', img_copy)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Return result with visualization
        return {
            **result,
            "visualization": f"data:image/jpeg;base64,{img_base64}"
        }
        
    except Exception as e:
        logger.error(f"Detection error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/deepfake/batch-detect")
async def batch_detect_deepfakes(
    files: list[UploadFile] = File(...),
    threshold: float = 0.31
):
    """Detect deepfakes in multiple images"""
    if not deepfake_detector:
        raise HTTPException(status_code=503, detail="Deepfake detector not initialized")
    
    results = []
    
    for file in files:
        try:
            # Read image
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                results.append({
                    "filename": file.filename,
                    "error": "Invalid image file"
                })
                continue
            
            # Detect deepfake
            original_threshold = deepfake_detector.threshold
            deepfake_detector.threshold = threshold
            result = deepfake_detector.predict(image)
            deepfake_detector.threshold = original_threshold
            
            result["filename"] = file.filename
            results.append(result)
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    # Summary statistics
    total_files = len(files)
    deepfakes = sum(1 for r in results if r.get("is_deepfake", False))
    real_images = sum(1 for r in results if not r.get("is_deepfake", False) and "error" not in r)
    errors = sum(1 for r in results if "error" in r)
    
    return {
        "summary": {
            "total_files": total_files,
            "deepfakes_detected": deepfakes,
            "real_images": real_images,
            "errors": errors,
            "deepfake_percentage": (deepfakes / total_files * 100) if total_files > 0 else 0
        },
        "results": results
    }

@app.get("/deepfake/stats")
async def get_deepfake_stats():
    """Get deepfake detection statistics from cache"""
    if not redis_client:
        return {"error": "Redis not connected"}
    
    try:
        # Get all deepfake analysis keys
        keys = redis_client.keys("deepfake:analysis:*")
        
        if not keys:
            return {
                "total_analyses": 0,
                "message": "No analyses found"
            }
        
        # Aggregate statistics
        total_analyses = len(keys)
        deepfake_count = 0
        total_confidence = 0
        
        for key in keys:
            data = redis_client.get(key)
            if data:
                result = json.loads(data)
                if result.get("is_deepfake"):
                    deepfake_count += 1
                total_confidence += result.get("confidence", 0)
        
        return {
            "total_analyses": total_analyses,
            "deepfakes_detected": deepfake_count,
            "real_images": total_analyses - deepfake_count,
            "deepfake_percentage": (deepfake_count / total_analyses * 100) if total_analyses > 0 else 0,
            "average_confidence": total_confidence / total_analyses if total_analyses > 0 else 0
        }
        
    except Exception as e:
        logger.error(f"Stats error: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)