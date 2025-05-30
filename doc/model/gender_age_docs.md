# เอกสารการใช้งานโมเดลทำนายเพศและอายุ

## ภาพรวม
โมเดลนี้ใช้สำหรับทำนายเพศ (Male/Female) และอายุของใบหน้าที่ตรวจพบแล้ว โดยรับ input เป็นรูปภาพและ bounding box ของใบหน้า

## การติดตั้ง

### 1. ติดตั้ง Dependencies
```bash
pip install onnxruntime opencv-python numpy scikit-image
```

หรือถ้าต้องการใช้ GPU:
```bash
pip install onnxruntime-gpu opencv-python numpy scikit-image
```

### 2. โหลดโมเดล
```bash
# สร้างโฟลเดอร์สำหรับเก็บโมเดล
mkdir -p weights

# โหลดโมเดล gender & age (1.26 MB)
wget -O weights/genderage.onnx https://github.com/yakhyo/facial-analysis/releases/download/v0.0.1/genderage.onnx
```

## โครงสร้างไฟล์ที่จำเป็น

```
your_project/
├── weights/
│   └── genderage.onnx
├── models/
│   └── gender_age.py
├── utils/
│   └── helpers.py
└── your_main.py
```

## โค้ดที่จำเป็น

### 1. สร้างไฟล์ `models/gender_age.py`
```python
import cv2
import numpy as np
import onnxruntime
from typing import Tuple

from utils.helpers import image_alignment

class Attribute:
    def __init__(self, model_path: str) -> None:
        """Age and Gender Prediction

        Args:
            model_path (str): Path to .onnx file
        """
        self.model_path = model_path

        self.input_std = 1.0
        self.input_mean = 0.0

        self._initialize_model(model_path=model_path)

    def _initialize_model(self, model_path: str):
        """Initialize the model from the given path.

        Args:
            model_path (str): Path to .onnx model.
        """
        try:
            self.session = onnxruntime.InferenceSession(
                model_path,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )

            # Get model info
            metadata = self.session.get_inputs()[0]
            input_shape = metadata.shape
            self.input_size = tuple(input_shape[2:4][::-1])

            self.input_names = [x.name for x in self.session.get_inputs()]
            self.output_names = [x.name for x in self.session.get_outputs()]

        except Exception as e:
            print(f"Failed to load the model: {e}")
            raise

    def preprocess(self, image: np.ndarray, bbox: np.ndarray):
        """Preprocessing

        Args:
            image (np.ndarray): Numpy image
            bbox (np.ndarray): Bounding box coordinates: [x1, y1, x2, y2]

        Returns:
            np.ndarray: Transformed image
        """
        width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        scale = self.input_size[0] / (max(width, height)*1.5)

        transformed_image, M = image_alignment(image, center, self.input_size[0], scale)

        input_size = tuple(transformed_image.shape[0:2][::-1])

        blob = cv2.dnn.blobFromImage(
            transformed_image,
            1.0/self.input_std,
            input_size,
            (self.input_mean, self.input_mean, self.input_mean),
            swapRB=True
        )
        return blob

    def postprocess(self, predictions: np.ndarray) -> Tuple[np.int64, int]:
        """Postprocessing

        Args:
            predictions (np.ndarray): Model predictions, shape: [1, 3]

        Returns:
            Tuple[np.int64, int]: Gender and Age values
        """
        gender = np.argmax(predictions[:2])
        age = int(np.round(predictions[2]*100))  # คูณด้วย 100
        return gender, age

    def get(self, image: np.ndarray, bbox: np.ndarray) -> Tuple[np.int64, int]:
        blob = self.preprocess(image, bbox)
        predictions = self.session.run(self.output_names, {self.input_names[0]: blob})[0][0]
        gender, age = self.postprocess(predictions)

        return gender, age
```

### 2. สร้างไฟล์ `utils/helpers.py`
```python
import cv2
import numpy as np
from skimage.transform import SimilarityTransform
from typing import List, Optional
from dataclasses import dataclass, field

@dataclass
class Face:
    """
    A class to represent a detected face with its attributes.

    Attributes:
        kps (List[float]): Keypoints of the face.
        bbox (List[float]): Bounding box coordinates of the face.
        age (Optional[int]): Age of the detected face.
        gender (Optional[int]): Gender of the detected face (1 for Male, 0 for Female).
    """
    kps: List[float] = field(default_factory=list)
    bbox: List[float] = field(default_factory=list)
    embedding: List[float] = field(default_factory=list)
    age: Optional[int] = None
    gender: Optional[int] = None

    @property
    def sex(self) -> Optional[str]:
        """Returns the gender as 'M' for Male and 'F' for Female."""
        if self.gender is None:
            return None
        return 'M' if self.gender == 1 else 'F'

def image_alignment(image, center, output_size, scale):
    """Image alignment function for face preprocessing"""
    T = SimilarityTransform(
        scale=scale,
        translation=(output_size / 2-center[0] * scale, output_size / 2 - center[1] * scale)
    )
    M = T.params[0:2]
    cropped = cv2.warpAffine(image, M, (output_size, output_size), borderValue=0.0)

    return cropped, M

def draw_face_info(frame: np.ndarray, face: Face) -> None:
    """Draws face bounding box and attributes on the frame.
    
    Args:
        frame (np.ndarray): Input frame
        face (Face): Face coordinates and attributes
    """
    if len(face.bbox) >= 4:
        x1, y1, x2, y2 = map(int, face.bbox[:4])
        
        # วาดกรอบ
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # แสดงข้อมูลเพศและอายุ
        if face.sex and face.age:
            text = f"{face.sex} {face.age}"
            cv2.putText(
                frame,
                text,
                (x1, y1 - 10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(255, 255, 255),
                thickness=1,
                lineType=cv2.LINE_AA
            )
```

### 3. สร้างไฟล์ `models/__init__.py`
```python
# เพื่อให้ import ได้
```

## วิธีการใช้งาน

### 1. Import และสร้างโมเดล
```python
import cv2
import numpy as np
from models.gender_age import Attribute

# สร้าง instance ของโมเดล
gender_age_model = Attribute(model_path="weights/genderage.onnx")
```

### 2. ทำนายเพศและอายุ
```python
# สมมติว่าคุณมี face detection แล้วได้ bounding box มา
# bbox format: [x1, y1, x2, y2]
image = cv2.imread("your_image.jpg")
bbox = [100, 50, 200, 180]  # ตัวอย่าง bounding box

# ทำนายเพศและอายุ
gender, age = gender_age_model.get(image, np.array(bbox))

# แสดงผลลัพธ์
gender_text = "Male" if gender == 1 else "Female"
print(f"Gender: {gender_text}, Age: {age}")
```

### 3. ตัวอย่างโค้ดใช้งานทันที (Complete Example)
```python
import cv2
import numpy as np
from models.gender_age import Attribute
from utils.helpers import Face, draw_face_info

def analyze_single_face(image_path, bbox):
    """
    วิเคราะห์เพศและอายุของใบหน้าเดียว
    
    Args:
        image_path (str): path ของรูปภาพ
        bbox (list): bounding box [x1, y1, x2, y2]
    """
    # โหลดโมเดล
    model = Attribute(model_path="weights/genderage.onnx")
    
    # อ่านรูปภาพ
    image = cv2.imread(image_path)
    
    # ทำนายเพศและอายุ
    gender, age = model.get(image, np.array(bbox))
    
    # แสดงผลลัพธ์
    gender_text = "Male" if gender == 1 else "Female"
    print(f"Gender: {gender_text}, Age: {age}")
    
    # สร้าง Face object และวาดผลลัพธ์บนรูป
    face = Face(bbox=bbox, gender=gender, age=age)
    draw_face_info(image, face)
    
    # แสดงรูปผลลัพธ์
    cv2.imshow("Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return gender_text, age

# ตัวอย่างการใช้งาน
if __name__ == "__main__":
    # ใส่ path รูปภาพและ bounding box ที่ได้จาก face detector ของคุณ
    image_path = "test_image.jpg"
    face_bbox = [100, 50, 200, 180]  # [x1, y1, x2, y2]
    
    gender, age = analyze_single_face(image_path, face_bbox)
    print(f"Result: {gender}, {age} years old")
```

### 3. ตัวอย่างการใช้งานแบบครบวงจร
```python
import cv2
import numpy as np
from models.gender_age import Attribute

def analyze_faces_gender_age(image_path, face_bboxes):
    """
    วิเคราะห์เพศและอายุของใบหน้าในรูปภาพ
    
    Args:
        image_path (str): path ของรูปภาพ
        face_bboxes (list): list ของ bounding boxes [[x1,y1,x2,y2], ...]
    
    Returns:
        list: ผลลัพธ์ [(gender, age), ...]
    """
    # โหลดโมเดล
    model = Attribute(model_path="weights/genderage.onnx")
    
    # อ่านรูปภาพ
    image = cv2.imread(image_path)
    
    results = []
    for bbox in face_bboxes:
        try:
            gender, age = model.get(image, np.array(bbox))
            gender_text = "Male" if gender == 1 else "Female"
            results.append((gender_text, age))
            print(f"Face: {gender_text}, Age: {age}")
        except Exception as e:
            print(f"Error processing face: {e}")
            results.append(("Unknown", 0))
    
    return results

# ตัวอย่างการใช้งาน
if __name__ == "__main__":
    # สมมติว่าคุณใช้ RetinaFace, MTCNN หรือ YOLOv8-Face ได้ bboxes มาแล้ว
    detected_faces = [
        [100, 50, 200, 180],   # หน้าคนที่ 1
        [300, 100, 400, 220],  # หน้าคนที่ 2
    ]
    
    results = analyze_faces_gender_age("test_image.jpg", detected_faces)
    for i, (gender, age) in enumerate(results):
        print(f"Person {i+1}: {gender}, {age} years old")
```

## การรวมเข้ากับ Face Detector ที่มีอยู่

### กับ RetinaFace
```python
from retinaface import RetinaFace
from models.gender_age import Attribute

def detect_and_analyze(image_path):
    # Face Detection ด้วย RetinaFace
    faces = RetinaFace.detect_faces(image_path)
    
    # โหลดโมเดลวิเคราะห์เพศ/อายุ
    gender_age_model = Attribute("weights/genderage.onnx")
    image = cv2.imread(image_path)
    
    for key in faces.keys():
        face_info = faces[key]
        bbox = face_info["facial_area"]  # [x, y, w, h] format
        
        # แปลงเป็น [x1, y1, x2, y2] format
        bbox_converted = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]
        
        gender, age = gender_age_model.get(image, np.array(bbox_converted))
        gender_text = "Male" if gender == 1 else "Female"
        
        print(f"Face {key}: {gender_text}, Age: {age}")
```

### กับ MTCNN
```python
from mtcnn import MTCNN
from models.gender_age import Attribute

def detect_and_analyze_mtcnn(image_path):
    detector = MTCNN()
    gender_age_model = Attribute("weights/genderage.onnx")
    
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Face Detection
    faces = detector.detect_faces(image_rgb)
    
    for i, face in enumerate(faces):
        bbox = face['box']  # [x, y, width, height]
        
        # แปลงเป็น [x1, y1, x2, y2] format
        bbox_converted = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]
        
        gender, age = gender_age_model.get(image, np.array(bbox_converted))
        gender_text = "Male" if gender == 1 else "Female"
        
        print(f"Face {i+1}: {gender_text}, Age: {age}")
```

## Input/Output Specifications

### Input
- **image**: NumPy array (OpenCV format, BGR)
- **bbox**: NumPy array รูปแบบ [x1, y1, x2, y2]
  - x1, y1: พิกัดซ้ายบน
  - x2, y2: พิกัดขวาล่าง

### Output
- **gender**: Integer (0 = Female, 1 = Male)
- **age**: Integer (อายุที่ทำนายได้)

## ข้อมูลเทคนิค

### Model Specifications
- **Input Size**: แปรผันตาม bounding box
- **Framework**: ONNX Runtime
- **Preprocessing**: Automatic alignment และ normalization
- **Model Size**: 1.26 MB

### Output Processing
- **Gender**: ใช้ `argmax` จาก predictions[:2] (0=Female, 1=Male)
- **Age**: โมเดลส่งออกค่า normalized (0-1) ต้อง **คูณด้วย 100** เพื่อได้อายุจริง
  ```python
  raw_age = predictions[2]        # ค่าจากโมเดล เช่น 0.25
  actual_age = raw_age * 100      # อายุจริง = 25 ปี
  ```

### Performance Notes
- รองรับการประมวลผลด้วย GPU (CUDA) และ CPU
- แนะนำให้ใช้ bounding box ที่มีขนาดเหมาะสม (ไม่เล็กหรือใหญ่เกินไป)
- ความแม่นยำจะดีขึ้นเมื่อใบหน้าชัดเจนและหันหน้าตรง

### Error Handling
```python
try:
    gender, age = gender_age_model.get(image, bbox)
    if age < 0 or age > 120:  # ตรวจสอบความสมเหตุสมผล
        print("Warning: Unusual age prediction")
except Exception as e:
    print(f"Error in gender/age prediction: {e}")
    gender, age = 0, 25  # ค่า default
```

## การปรับแต่งและเพิ่มประสิทธิภาพ

### 1. Batch Processing
หากต้องการประมวลผลหลายใบหน้าพร้อมกัน:
```python
def batch_predict(model, image, bboxes):
    results = []
    for bbox in bboxes:
        gender, age = model.get(image, bbox)
        results.append((gender, age))
    return results
```

### 2. Confidence Filtering
```python
def predict_with_validation(model, image, bbox, min_face_size=30):
    # ตรวจสอบขนาดใบหน้า
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    
    if width < min_face_size or height < min_face_size:
        return None, None  # ใบหน้าเล็กเกินไป
    
    return model.get(image, bbox)
```

## ตัวอย่างการแสดงผลบนรูปภาพ

```python
def draw_results(image, bbox, gender, age):
    x1, y1, x2, y2 = map(int, bbox)
    
    # วาดกรอบ
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # แสดงข้อมูล
    gender_text = "Male" if gender == 1 else "Female"
    text = f"{gender_text}, {age}"
    
    cv2.putText(image, text, (x1, y1-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return image
```

## การแก้ไขปัญหาที่พบบ่อย

### ปัญหา: โมเดลโหลดไม่ได้
**แก้ไข**: ตรวจสอบ path ของไฟล์และติดตั้ง onnxruntime

### ปัญหา: ผลลัพธ์ไม่ถูกต้อง
**แก้ไข**: 
- ตรวจสอบ bounding box format
- ตรวจสอบคุณภาพของรูปภาพ input
- ตรวจสอบขนาดของใบหน้า

### ปัญหา: ประมวลผลช้า
**แก้ไข**: 
- ใช้ GPU provider หากมี CUDA
- ลดขนาดรูปภาพก่อนประมวลผล
- ใช้ batch processing สำหรับหลายใบหน้า