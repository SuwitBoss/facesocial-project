import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class YOLOv5FaceDetector:
    def __init__(self, model_path: str = "/app/models/yolov5s-face.onnx"):
        """
        Initialize YOLO Face Detector
        """
        try:
            # Setup ONNX Runtime
            providers = ['CPUExecutionProvider']
            if 'CUDAExecutionProvider' in ort.get_available_providers():
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            
            self.session = ort.InferenceSession(model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [output.name for output in self.session.get_outputs()]
            
            # Input shape
            input_shape = self.session.get_inputs()[0].shape
            self.input_height = input_shape[2] if input_shape[2] > 0 else 640
            self.input_width = input_shape[3] if input_shape[3] > 0 else 640
            
            logger.info(f"YOLOv5 Face Detector loaded successfully")
            logger.info(f"Input shape: {self.input_width}x{self.input_height}")
            logger.info(f"Providers: {self.session.get_providers()}")
            
        except Exception as e:
            logger.error(f"Failed to load YOLOv5 Face Detector: {e}")
            raise

    def preprocess_image(self, image: np.ndarray):
        """
        Preprocess image for YOLO inference
        """
        # Resize image
        img_resized = cv2.resize(image, (self.input_width, self.input_height))
        
        # Convert BGR to RGB and normalize
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_normalized = img_rgb.astype(np.float32) / 255.0
        
        # Convert to CHW format and add batch dimension
        img_transposed = img_normalized.transpose(2, 0, 1)  # HWC to CHW
        img_batch = np.expand_dims(img_transposed, axis=0)  # Add batch dimension
        
        return img_batch

    def apply_nms(self, boxes, scores, iou_threshold=0.5):
        """Apply Non-Maximum Suppression to remove duplicate detections"""
        if len(boxes) == 0:
            return []
        boxes = np.array(boxes)
        scores = np.array(scores)
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= iou_threshold)[0]
            order = order[inds + 1]
        return keep

    def postprocess_detections(self, outputs, original_shape, conf_threshold=0.7):
        """
        Post-process YOLO outputs to get face bounding boxes
        """
        predictions = outputs[0]  # Assuming first output is detections
        
        # Get original image dimensions
        orig_h, orig_w = original_shape[:2]
        
        # Scale factors
        scale_x = orig_w / self.input_width
        scale_y = orig_h / self.input_height
        
        faces = []
        
        # Process predictions (assuming YOLO format: x, y, w, h, conf, ...)
        for detection in predictions[0]:  # Remove batch dimension
            if len(detection) >= 5:
                confidence = detection[4]
                
                if confidence > conf_threshold:
                    # Get bounding box coordinates
                    x_center, y_center, width, height = detection[:4]
                    
                    # Convert from center format to corner format
                    x1 = int((x_center - width/2) * scale_x)
                    y1 = int((y_center - height/2) * scale_y)
                    x2 = int((x_center + width/2) * scale_x)
                    y2 = int((y_center + height/2) * scale_y)
                    
                    # Ensure coordinates are within image bounds
                    x1 = max(0, min(x1, orig_w))
                    y1 = max(0, min(y1, orig_h))
                    x2 = max(0, min(x2, orig_w))
                    y2 = max(0, min(y2, orig_h))
                    
                    faces.append({
                        "bbox": [x1, y1, x2, y2],
                        "confidence": float(confidence),
                        "width": x2 - x1,
                        "height": y2 - y1
                    })
        
        # Apply NMS
        if faces:
            boxes = [f["bbox"] for f in faces]
            scores = [f["confidence"] for f in faces]
            keep_indices = self.apply_nms(boxes, scores, iou_threshold=0.5)
            faces = [faces[i] for i in keep_indices]
        
        return faces

    def detect_faces(self, image: np.ndarray, conf_threshold=0.7):
        """
        Detect faces in image
        """
        try:
            # Preprocess
            input_data = self.preprocess_image(image)
            
            # Inference
            outputs = self.session.run(self.output_names, {self.input_name: input_data})
            
            # Post-process
            faces = self.postprocess_detections(outputs, image.shape, conf_threshold)
            
            return {
                "success": True,
                "faces_count": len(faces),
                "faces": faces
            }
            
        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "faces_count": 0,
                "faces": []
            }

