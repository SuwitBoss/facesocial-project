# คู่มือการใช้งาน Deepfake Detection Model

## 📋 ข้อมูลทั่วไป

**ชื่อโมเดล:** Deepfake-Detection-Exp-02-21-ONNX  
**เวอร์ชัน:** v1.0  
**ประเภท:** Image Classification (Binary Classification)  
**เทคโนโลยี:** Vision Transformer (ViT) based on Google's vit-base-patch16-224-in21k  
**ความแม่นยำ:** 98.84%  

## 🎯 การจำแนกประเภท

```
Class Mapping:
├── 0: "Deepfake" (ภาพปลอม/สังเคราะห์)
└── 1: "Real" (ภาพจริง)

Label Mapping:
├── "Deepfake": 0
└── "Real": 1
```

## 📊 ประสิทธิภาพโมเดล

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Deepfake | 99.62% | 98.06% | 98.83% | 1,600 |
| Real | 98.09% | 99.62% | 98.85% | 1,600 |
| **Overall** | **98.86%** | **98.84%** | **98.84%** | **3,200** |

## 🗂️ รูปแบบโมเดลที่มีให้

| ไฟล์ | ขนาด | ความเร็ว | ความแม่นยำ | แนะนำสำหรับ |
|------|------|-----------|------------|-------------|
| `model.onnx` | ใหญ่ที่สุด | ช้าที่สุด | สูงที่สุด | งานที่ต้องการความแม่นยำสูง |
| `model_fp16.onnx` | ปานกลาง | ปานกลาง | สูง | **แนะนำสำหรับใช้งานทั่วไป** |
| `model_int8.onnx` | เล็ก | เร็ว | ดี | CPU inference |
| `model_uint8.onnx` | เล็ก | เร็ว | ดี | Mobile/Embedded devices |
| `model_q4.onnx` | เล็กมาก | เร็วมาก | พอใช้ | Real-time applications |
| `model_q4f16.onnx` | เล็กมาก | เร็วมาก | พอใช้ | Hybrid precision |
| `model_bnb4.onnx` | เล็กที่สุด | เร็วที่สุด | ต่ำที่สุด | Resource-constrained |

## ⚙️ การตั้งค่าโมเดล (Model Configuration)

### พารามิเตอร์หลัก
```json
{
  "model_type": "vit",
  "architectures": ["ViTForImageClassification"],
  "hidden_size": 768,
  "num_hidden_layers": 12,
  "num_attention_heads": 12,
  "intermediate_size": 3072,
  "patch_size": 16,
  "image_size": 224,
  "num_channels": 3,
  "problem_type": "single_label_classification"
}
```

### พารามิเตอร์การประมวลผลภาพ
```json
{
  "do_resize": true,
  "do_rescale": true,
  "do_normalize": true,
  "size": {"height": 224, "width": 224},
  "rescale_factor": 0.00392156862745098,
  "image_mean": [0.5, 0.5, 0.5],
  "image_std": [0.5, 0.5, 0.5],
  "resample": 2
}
```

### พารามิเตอร์ Quantization
```json
{
  "modes": ["fp16", "q8", "int8", "uint8", "q4", "q4f16", "bnb4"],
  "per_channel": true,
  "reduce_range": true,
  "is_symmetric": true,
  "quant_type": 1
}
```

## 🚀 วิธีการใช้งาน

### 1. การติดตั้ง Dependencies

```bash
pip install transformers torch pillow onnxruntime
# สำหรับ GPU (ถ้ามี)
pip install onnxruntime-gpu
```

### 2. การใช้งานผ่าน Hugging Face Pipeline

```python
from transformers import pipeline
from PIL import Image

# โหลดโมเดล
pipe = pipeline(
    'image-classification', 
    model="prithivMLmods/Deepfake-Detection-Exp-02-21",
    device=0  # ใช้ GPU, เปลี่ยนเป็น -1 สำหรับ CPU
)

# ทำนายผล
image_path = "path_to_your_image.jpg"
result = pipe(image_path)
print(f"Result: {result}")

# ตัวอย่างผลลัพธ์
# [{'label': 'Real', 'score': 0.9876}, {'label': 'Deepfake', 'score': 0.0124}]
```

### 3. การใช้งานผ่าน PyTorch

```python
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import torch

# โหลดโมเดลและ processor
model = ViTForImageClassification.from_pretrained(
    "prithivMLmods/Deepfake-Detection-Exp-02-21"
)
processor = ViTImageProcessor.from_pretrained(
    "prithivMLmods/Deepfake-Detection-Exp-02-21"
)

# ตั้งค่าโมเดลเป็น evaluation mode
model.eval()

# โหลดและประมวลผลภาพ
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        predicted_class = torch.argmax(logits, dim=1).item()
    
    # แปลงผลลัพธ์
    label = model.config.id2label[predicted_class]
    confidence = probabilities[0][predicted_class].item()
    
    return {
        "label": label,
        "confidence": confidence,
        "probabilities": {
            "Deepfake": probabilities[0][0].item(),
            "Real": probabilities[0][1].item()
        }
    }

# ใช้งาน
result = predict_image("your_image.jpg")
print(f"Prediction: {result['label']}")
print(f"Confidence: {result['confidence']:.4f}")
print(f"Probabilities: {result['probabilities']}")
```

### 4. การใช้งานผ่าน ONNX Runtime

```python
import onnxruntime as ort
import numpy as np
from PIL import Image

# โหลดโมเดล ONNX
session = ort.InferenceSession("model_fp16.onnx")

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))
    
    # แปลงเป็น numpy array และ normalize
    image_array = np.array(image).astype(np.float32) / 255.0
    image_array = (image_array - 0.5) / 0.5  # Normalize [-1, 1]
    
    # เปลี่ยน shape เป็น (1, 3, 224, 224)
    image_array = np.transpose(image_array, (2, 0, 1))
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

def predict_onnx(image_path):
    input_data = preprocess_image(image_path)
    
    # รันการทำนาย
    outputs = session.run(None, {"pixel_values": input_data})
    logits = outputs[0]
    
    # คำนวณ probabilities
    probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    predicted_class = np.argmax(probabilities, axis=1)[0]
    
    labels = {0: "Deepfake", 1: "Real"}
    
    return {
        "label": labels[predicted_class],
        "confidence": probabilities[0][predicted_class],
        "probabilities": {
            "Deepfake": probabilities[0][0],
            "Real": probabilities[0][1]
        }
    }

# ใช้งาน
result = predict_onnx("your_image.jpg")
print(result)
```

## 📝 พารามิเตอร์สำคัญในการใช้งาน

### การปรับแต่งประสิทธิภาพ

```python
# สำหรับ GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Mixed Precision (ประหยัดหน่วยความจำ)
with torch.cuda.amp.autocast():
    outputs = model(**inputs)

# Batch Processing (ประมวลผลหลายภาพพร้อมกัน)
images = [Image.open(path) for path in image_paths]
inputs = processor(images=images, return_tensors="pt")
```

### Threshold Setting

```python
def classify_with_threshold(probabilities, threshold=0.5):
    """
    กำหนด threshold สำหรับการจำแนก
    - threshold สูง: ลด False Positive (ปลอดภัยมากขึ้น)
    - threshold ต่ำ: ลด False Negative (ตรวจจับได้มากขึ้น)
    """
    deepfake_prob = probabilities["Deepfake"]
    
    if deepfake_prob > threshold:
        return "Deepfake", deepfake_prob
    else:
        return "Real", probabilities["Real"]

# ตัวอย่างการใช้งาน
result = predict_image("image.jpg")
classification, confidence = classify_with_threshold(
    result["probabilities"], 
    threshold=0.7  # ปรับตามความต้องการ
)
```

## ⚠️ ข้อจำกัดและข้อควรระวัง

### 1. ข้อจำกัดทางเทคนิค
- **ขนาดภาพ:** จำกัดที่ 224x224 pixels
- **รูปแบบภาพ:** รองรับ RGB เท่านั้น
- **ประเภทไฟล์:** JPG, PNG, BMP

### 2. ข้อจำกัดในการใช้งาน
- อาจไม่เหมาะกับเทคนิค deepfake ใหม่ที่ไม่เคยเห็น
- ประสิทธิภาพอาจลดลงกับภาพความละเอียดสูง
- อาจมี bias จากข้อมูลที่ใช้ฝึก

### 3. คำแนะนำสำหรับการใช้งานจริง
```python
# ตรวจสอบคุณภาพภาพก่อนการทำนาย
def validate_image(image_path):
    try:
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # ตรวจสอบขนาดภาพ
        if min(image.size) < 224:
            print(f"Warning: Image resolution is low ({image.size})")
        
        return True, image
    except Exception as e:
        print(f"Error loading image: {e}")
        return False, None

# การใช้งานที่ปลอดภัย
def safe_predict(image_path, confidence_threshold=0.8):
    is_valid, image = validate_image(image_path)
    if not is_valid:
        return None
    
    result = predict_image(image_path)
    
    if result["confidence"] < confidence_threshold:
        return {
            "prediction": "Uncertain",
            "confidence": result["confidence"],
            "recommendation": "Manual review required"
        }
    
    return result
```

## 🔧 การแก้ไขปัญหาที่พบบ่อย

### ปัญหา Memory หมด
```python
# ลด batch size
inputs = processor(images=image, return_tensors="pt", max_length=1)

# ใช้ gradient checkpointing
model.gradient_checkpointing_enable()

# Clear cache
torch.cuda.empty_cache()
```

### ปัญหาความเร็ว
```python
# ใช้โมเดล quantized
model_path = "model_int8.onnx"  # หรือ model_q4.onnx

# ปิด gradient computation
with torch.no_grad():
    outputs = model(**inputs)
```

## 📞 การสนับสนุนและข้อมูลเพิ่มเติม

- **Repository:** `prithivMLmods/Deepfake-Detection-Exp-02-21`
- **License:** Apache 2.0
- **Dataset:** `prithivMLmods/Deepfake-vs-Real`
- **Base Model:** `google/vit-base-patch16-224-in21k`

---

**หมายเหตุ:** เอกสารนี้อัปเดตล่าสุดวันที่ 28 พฤษภาคม 2025 ให้ตรวจสอบ repository หลักเพื่อดูข้อมูลและการอัปเดตล่าสุด