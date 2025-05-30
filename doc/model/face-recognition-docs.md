genderage# คู่มือการใช้งานระบบทดสอบโมเดลจดจำใบหน้า

## 1. ภาพรวมระบบ

ระบบนี้ใช้สำหรับการทดสอบและเปรียบเทียบประสิทธิภาพของโมเดลจดจำใบหน้า 3 รูปแบบ (`AdaFace`, `FaceNet`, `ArcFace`) รวมถึงการทดสอบแบบ `Ensemble` (ผสมผสานทั้ง 3 โมเดล) โดยมีคุณสมบัติหลัก:

- ทดสอบความแม่นยำในการจับคู่ใบหน้า
- วัดประสิทธิภาพของโมเดลด้วยเมทริกซ์มาตรฐาน (Accuracy, F1 Score, ROC AUC)
- ทดสอบความทนทานของโมเดลต่อการเปลี่ยนแปลงของภาพ (ความสว่าง, ความคมชัด, ความเบลอ, สัญญาณรบกวน)
- หาค่าเกณฑ์ (Threshold) ที่เหมาะสมและน้ำหนักที่ดีที่สุดสำหรับโมเดล Ensemble
- แสดงผลการทดสอบในรูปแบบกราฟและตาราง
- วิเคราะห์เชิงลึกของข้อผิดพลาดในการจดจำใบหน้า

## 2. ขั้นตอนการติดตั้ง

### 2.1 ติดตั้งแพ็คเกจที่จำเป็น

```bash
pip install numpy opencv-python onnxruntime scikit-learn pandas matplotlib seaborn tqdm
```

### 2.2 ดาวน์โหลดโมเดล

โค้ดนี้ใช้ ONNX เป็นรูปแบบไฟล์โมเดล ให้ดาวน์โหลดโมเดลที่ได้แปลงเป็น ONNX แล้วโดยสร้างโฟลเดอร์ `/content/models` และดาวน์โหลดไฟล์โมเดลต่อไปนี้:

1. **AdaFace**: [ดาวน์โหลด adaface_ir101.onnx](https://drive.google.com/file/d/1KJE5IeCi9T-TzXThYHGR4p4k1jEV85vk/view?usp=sharing)
2. **FaceNet**: [ดาวน์โหลด facenet_vggface2.onnx](https://drive.google.com/file/d/1YIRvp7Yrp9UXr9JV8P1HZgHIQfIGLea4/view?usp=sharing)  
3. **ArcFace**: [ดาวน์โหลด arcface_r100.onnx](https://drive.google.com/file/d/1Y0OQAYuXz5V-hET3HKGlQ2QRqiX0iB8F/view?usp=sharing)

(หมายเหตุ: ลิงก์ดาวน์โหลดอาจปรับเปลี่ยนตามแหล่งที่เก็บไฟล์)

### 2.3 เตรียมชุดข้อมูลทดสอบ

ระบบนี้ออกแบบมาให้ใช้กับชุดข้อมูล VGGFace2 แต่สามารถปรับให้ใช้กับชุดข้อมูลอื่นได้ โดยชุดข้อมูลควรมีโครงสร้างดังนี้:

```
dataset_path/
├── train/
│   ├── person1_id/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── person2_id/
│   └── ...
└── val/
    ├── person1_id/
    ├── person2_id/
    └── ...
```

## 3. วิธีการใช้งาน

### 3.1 การปรับแต่งการตั้งค่า

ปรับแต่งค่าในฟังก์ชัน `main()`:

```python
model_configs = {
    "AdaFace": {"path": "/content/models/adaface_ir101.onnx", "size": (112, 112), "norm": "arcface"},
    "FaceNet": {"path": "/content/models/facenet_vggface2.onnx", "size": (160, 160), "norm": "facenet"},
    "ArcFace": {"path": "/content/models/arcface_r100.onnx", "size": (112, 112), "norm": "arcface"},
}
dataset_path_config = "/path/to/your/dataset"  # เปลี่ยนเป็นพาธไปยังชุดข้อมูลของคุณ
dataset_split_to_use = "val"  # เลือก "val" หรือ "train"
```

### 3.2 การรันระบบ

เรียกใช้ทั้งสคริปต์:

```bash
python face_recognition_test.py
```

หรือในกรณีใช้ Jupyter Notebook:

```python
%run face_recognition_test.py
```

### 3.3 การรันบางส่วนของระบบ

หากต้องการรันเฉพาะบางส่วน สามารถปรับปรุงฟังก์ชัน `main()` โดยเปิด/ปิดการทำงานของแต่ละส่วน:

```python
def main():
    # ... (โค้ดส่วนต้น)
    
    # ส่วน 1: ทดสอบแต่ละโมเดลกับข้อมูลปกติ
    test_clean_data = True  # ตั้งเป็น False เพื่อข้ามการทดสอบส่วนนี้
    
    if test_clean_data:
        print(f"\n--- 1. ทดสอบแต่ละโมเดล (Clean Data) ---")
        # ... (โค้ดส่วนนี้)
    
    # ส่วน 2: ทดสอบ Ensemble Model
    test_ensemble = True  # ตั้งเป็น False เพื่อข้ามการทดสอบส่วนนี้
    
    if test_ensemble and len(models_list) >= 2:
        print(f"\n--- 2. ทดสอบ Ensemble Model ---")
        # ... (โค้ดส่วนนี้)
    
    # ส่วน 3: ทดสอบความทนทาน
    test_robustness = True  # ตั้งเป็น False เพื่อข้ามการทดสอบส่วนนี้
    
    if test_robustness:
        print("\n--- 3. ทดสอบความทนทาน (Augmentation) ---")
        # ... (โค้ดส่วนนี้)
```

## 4. ผลการทดสอบและการแปลความหมาย

### 4.1 ผลการทดสอบของแต่ละโมเดล

จากการทดสอบกับชุดข้อมูล VGGFace2 (200 คู่ภาพ) ได้ผลลัพธ์ดังนี้:

| โมเดล | ความแม่นยำ | F1 Score | ROC AUC | ความเร็ว (EPS) | Threshold |
|-------|------------|----------|---------|---------------|-----------|
| AdaFace | 81.50% | 0.7811 | 0.8621 | 54.14 | 0.2000 |
| FaceNet | 85.50% | 0.8585 | 0.9124 | 81.35 | 0.2000 |
| ArcFace | 67.50% | 0.7280 | 0.8014 | 67.88 | 0.1500 |
| Ensemble | 88.50% | 0.8844 | 0.9292 | 67.79 | 0.2000 |

> **EPS (Embeddings Per Second)** คือจำนวนภาพที่โมเดลสามารถสร้าง embedding ได้ต่อวินาที - ยิ่งสูงยิ่งเร็ว

### 4.2 รูปแบบความผิดพลาด

แต่ละโมเดลมีลักษณะความผิดพลาดที่แตกต่างกัน:

- **AdaFace**: มี False Negative สูง (34) แต่ False Positive ต่ำ (3) - เหมาะกับงานที่ต้องการความปลอดภัยสูง
- **FaceNet**: มีความสมดุลระหว่าง False Positive (17) และ False Negative (12) - เหมาะกับงานทั่วไป
- **ArcFace**: มี False Positive สูง (52) และ False Negative ต่ำ (13) - เหมาะกับงานที่ต้องการไม่พลาดการจับคู่
- **Ensemble**: มีความสมดุลที่ดีที่สุด (FP: 11, FN: 12) - เหมาะกับทุกประเภทงาน

### 4.3 ผลการทดสอบความทนทาน

ผลกระทบของการเปลี่ยนแปลงภาพต่อค่า F1 Score:

| การเปลี่ยนแปลง | AdaFace | FaceNet | ArcFace | Ensemble |
|----------------|---------|---------|---------|----------|
| Clean (ภาพปกติ) | 0.7811 | 0.8585 | 0.7280 | 0.8844 |
| Brightness Low (-60) | 0.7598 | 0.8058 | 0.7234 | 0.8213 |
| Brightness High (+60) | 0.7545 | 0.8557 | 0.7102 | 0.8700 |
| Contrast Low (0.6) | 0.7674 | 0.8585 | 0.7288 | 0.8844 |
| Contrast High (1.4) | 0.7602 | 0.8325 | 0.7073 | 0.8731 |
| Gaussian Blur (5x5) | 0.7362 | 0.8416 | 0.7203 | 0.8442 |
| Gaussian Noise (σ=25) | 0.7553 | 0.7745 | 0.6908 | 0.7980 |

**สรุป**: โมเดล Ensemble มีความทนทานต่อการเปลี่ยนแปลงสูงที่สุด, FaceNet เป็นโมเดลเดี่ยวที่ดีที่สุด, สัญญาณรบกวน Gaussian Noise ส่งผลกระทบมากที่สุด

### 4.4 น้ำหนัก Ensemble ที่เหมาะสม

จากการทดสอบพบว่าน้ำหนักที่เหมาะสมในการรวมโมเดลคือ:
- AdaFace: 25%
- FaceNet: 50%
- ArcFace: 25%

## 5. การประยุกต์ใช้งานจริง

### 5.1 แนวทางการเลือกโมเดลตามวัตถุประสงค์

- **ต้องการความปลอดภัยสูง** (เช่น การเข้าถึงข้อมูลสำคัญ):
  - ใช้ AdaFace หรือ Ensemble กับ Threshold ที่สูงขึ้น (0.3-0.4)

- **ต้องการความสมดุล** (เช่น การยืนยันตัวตนทั่วไป):
  - ใช้ FaceNet หรือ Ensemble กับ Threshold ปกติ (0.2)

- **ต้องการจับคู่ได้มากที่สุด** (เช่น การระบุบุคคลในกลุ่มตัวเลือก):
  - ใช้ ArcFace กับ Threshold ต่ำ (0.1-0.15)

- **ต้องการประมวลผลเร็ว** (เช่น อุปกรณ์ที่มีทรัพยากรจำกัด):
  - ใช้ FaceNet เนื่องจากมีความเร็วสูงสุด (81.35 EPS)

- **ต้องการความทนทานสูง** (เช่น กล้องวงจรปิด):
  - ใช้ Ensemble หรือ FaceNet ที่ปรับ Threshold ตามสภาพแวดล้อม

### 5.2 การปรับค่า Threshold ตามสภาพแวดล้อม

- **สภาพแสงปกติ**: ใช้ค่า Threshold จากการทดสอบ
  - AdaFace, FaceNet, Ensemble: 0.2
  - ArcFace: 0.15

- **สภาพแสงต่ำ**: ลดค่า Threshold ลง 10-15%
  - AdaFace, FaceNet, Ensemble: 0.17-0.18
  - ArcFace: 0.13-0.14

- **สภาพที่มีสัญญาณรบกวน**: ลดค่า Threshold ลง 20%
  - AdaFace, FaceNet, Ensemble: 0.16
  - ArcFace: 0.12

### 5.3 ตัวอย่างการปรับโค้ดสำหรับการใช้งานจริง

```python
# ตัวอย่างการใช้งานโมเดลแบบง่าย
def face_verification(img1_path, img2_path, model_type="ensemble", security_level="medium"):
    # โหลดโมเดล
    models = load_models()
    if model_type.lower() == "adaface":
        model = models["AdaFace"]
        base_threshold = 0.2
    elif model_type.lower() == "facenet":
        model = models["FaceNet"]
        base_threshold = 0.2
    elif model_type.lower() == "arcface":
        model = models["ArcFace"]
        base_threshold = 0.15
    else:  # ensemble
        model = EnsembleFaceRecognition(list(models.values()), weights=[0.25, 0.5, 0.25])
        base_threshold = 0.2
    
    # ปรับ Threshold ตามระดับความปลอดภัย
    if security_level == "high":
        threshold = base_threshold + 0.1  # เพิ่ม threshold เพื่อลด False Positive
    elif security_level == "low":
        threshold = base_threshold - 0.05  # ลด threshold เพื่อลด False Negative
    else:  # medium
        threshold = base_threshold
    
    # ทำการเปรียบเทียบใบหน้า
    similarity, is_same_person = model.compare_faces(img1_path, img2_path, threshold)
    
    return {
        "is_same_person": is_same_person,
        "similarity_score": similarity,
        "threshold_used": threshold,
        "model_used": model_type
    }
```

## 6. การแก้ไขปัญหาที่พบบ่อย

### 6.1 ปัญหาด้านความแม่นยำ

- **ความแม่นยำต่ำเกินไป**:
  - ลองปรับค่า Threshold
  - เปลี่ยนไปใช้โมเดลที่มีประสิทธิภาพสูงกว่า
  - ใช้วิธี Ensemble โดยเพิ่มน้ำหนักให้โมเดลที่แม่นยำกว่า

- **False Positive สูง**:
  - เพิ่มค่า Threshold
  - ใช้โมเดล AdaFace หรือเพิ่มน้ำหนักของ AdaFace ในแบบ Ensemble

- **False Negative สูง**:
  - ลดค่า Threshold
  - ใช้โมเดล ArcFace หรือเพิ่มน้ำหนักของ ArcFace ในแบบ Ensemble

### 6.2 ปัญหาด้านประสิทธิภาพ

- **ประมวลผลช้า**:
  - ใช้ FaceNet ซึ่งเร็วที่สุด
  - ลดขนาดภาพก่อนส่งเข้าโมเดล
  - ใช้ GPU หากมี (ตรวจสอบว่า ONNX Runtime มี CUDAExecutionProvider)

- **หน่วยความจำไม่เพียงพอ**:
  - ลดขนาด batch size
  - ใช้เพียงโมเดลเดียวแทน Ensemble

### 6.3 ปัญหาทั่วไป

- **ไม่สามารถโหลดโมเดล**:
  - ตรวจสอบว่าไฟล์โมเดลอยู่ในตำแหน่งที่ถูกต้อง
  - ตรวจสอบว่าโหลด ONNX Runtime แล้ว
  - ลองเปลี่ยนเป็น CPU Provider หากไม่มี GPU

- **ไม่สามารถโหลดภาพ**:
  - ตรวจสอบว่า OpenCV ติดตั้งถูกต้อง
  - ตรวจสอบเส้นทางไฟล์ภาพ
  - ตรวจสอบรูปแบบไฟล์ภาพว่ารองรับ (jpg, jpeg, png)

## 7. สรุป

ระบบทดสอบโมเดลจดจำใบหน้านี้ช่วยให้คุณสามารถประเมินประสิทธิภาพของโมเดลจดจำใบหน้าต่างๆ และเลือกใช้โมเดลที่เหมาะสมกับงานของคุณ โดยสรุปผลการทดสอบ:

1. **FaceNet** เป็นโมเดลเดี่ยวที่ดีที่สุดสำหรับการใช้งานทั่วไป เนื่องจากมีความสมดุลระหว่างความแม่นยำและความเร็ว
2. **AdaFace** เหมาะกับงานที่ต้องการความปลอดภัยสูง เนื่องจากมี False Positive ต่ำ
3. **ArcFace** เหมาะกับงานที่ต้องการพบคู่ที่เป็นไปได้มากที่สุด แม้จะมีโอกาสจับคู่ผิดพลาดสูง
4. **Ensemble** ให้ความแม่นยำสูงสุดและมีความทนทานต่อการเปลี่ยนแปลงของภาพสูง แต่ต้องใช้ทรัพยากรมากกว่า

การปรับค่า Threshold และการเลือกโมเดลให้เหมาะกับสภาพแวดล้อมและวัตถุประสงค์การใช้งานเป็นกุญแจสำคัญในการเพิ่มประสิทธิภาพของระบบจดจำใบหน้า
