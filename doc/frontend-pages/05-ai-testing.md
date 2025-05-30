# หน้าทดสอบ AI แต่ละบริการ

## ข้อมูลพื้นฐาน
- **URLs:** 
  - `/test/face-recognition`
  - `/test/face-antispoofing`
  - `/test/deepfake-detector`
  - `/test/face-detector`
  - `/test/age-gender-detection`
- **สถานะการเข้าถึง:** สาธารณะ (ไม่ต้องเข้าสู่ระบบ)
- **วัตถุประสงค์:** ให้ผู้ใช้ทดสอบความสามารถของ AI แต่ละตัวก่อนตัดสินใจใช้งานจริง

---

## 1. หน้าทดสอบ Face Recognition (`/test/face-recognition`)

### Components หลัก

#### Header Section
- **ชื่อบริการ:** "ทดสอบ Face Recognition API"
- **คำอธิบาย:** "ทดสอบการจดจำและระบุตัวตนใบหน้าด้วย AI ที่แม่นยำสูง"
- **สถานะ API:** 🟢 Online (XXX ms) / 🔴 Offline / 🟡 Slow

#### Mode Selector (3 โหมด)
```
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│ 📝 Register │ │ ✅ Verify   │ │ 🔍 Identify │
│   Mode      │ │   Mode      │ │   Mode      │
└─────────────┘ └─────────────┘ └─────────────┘
```

##### Mode 1: Register (ลงทะเบียนใบหน้า)
- **Input:**
  - อัปโหลดรูป/ถ่ายภาพผ่านกล้อง
  - กรอก User ID (สำหรับทดสอบ): "test_user_001"
- **Process:** ลงทะเบียนใบหน้าเข้าระบบทดสอบ
- **Output:**
  - Face ID ที่ได้รับ
  - Face Embedding Quality Score
  - ข้อความ: "ลงทะเบียนสำเร็จ"

##### Mode 2: Verify (ตรวจสอบใบหน้า)
- **Input:**
  - อัปโหลดรูป/ถ่ายภาพ
  - กรอก User ID ที่ต้องการตรวจสอบ
- **Process:** เปรียบเทียบใบหน้ากับข้อมูลที่ลงทะเบียนไว้
- **Output:**
  - Match/No Match
  - Confidence Score (0-100%)
  - Distance Score

##### Mode 3: Identify (ระบุตัวตน)
- **Input:**
  - อัปโหลดรูป/ถ่ายภาพ
  - จำนวนผลลัพธ์สูงสุด (1-10)
- **Process:** ค้นหาใบหน้าใน Database ทดสอบ
- **Output:**
  - รายการ User ID ที่อาจตรงกัน
  - Confidence Score แต่ละรายการ
  - Ranked Results

#### การนำทาง
- **← กลับรายการ AI** → `/api-testing`
- **ทดสอบ AI อื่น** → เลือกจาก Quick Links
- **เข้าสู่ระบบ** → `/login`

---

## 2. หน้าทดสอบ Face Antispoofing (`/test/face-antispoofing`)

### Components หลัก

#### Header Section
- **ชื่อบริการ:** "ทดสอบ Face Antispoofing API"
- **คำอธิบาย:** "ตรวจสอบการปลอมแปลงใบหน้า ป้องกันรูปภาพและหน้ากาก"

#### Detection Mode Selector
```
┌─────────────┐ ┌─────────────┐
│ 📷 Passive  │ │ 🎯 Active   │
│ Detection   │ │ Detection   │
└─────────────┘ └─────────────┘
```

##### Passive Detection
- **Input:** อัปโหลดรูปภาพหรือวิดีโอ
- **Process:** วิเคราะห์ภาพเพื่อหาเบาะแสการปลอมแปลง
- **Output:**
  - Live/Fake Status
  - Confidence Score (%)
  - เหตุผลการตัดสิน

##### Active Detection (Liveness Check)
- **Input:** Live Camera + การทำตามคำสั่ง
- **Commands:** "หันซ้าย", "หันขวา", "พยักหน้า", "กระพริบตา"
- **Process:** ตรวจสอบการเคลื่อนไหวแบบเรียลไทม์
- **Output:**
  - Liveness Score (%)
  - ผลแต่ละคำสั่ง
  - Overall Result

---

## 3. หน้าทดสอบ Deepfake Detector (`/test/deepfake-detector`)

### Components หลัก

#### Header Section
- **ชื่อบริการ:** "ทดสอบ Deepfake Detection API"
- **คำอธิบาย:** "ตรวจจับภาพและวิดีโอที่ถูกสร้างด้วย AI"

#### Media Type Selector
```
┌─────────────┐ ┌─────────────┐
│ 🖼️ Image    │ │ 🎥 Video    │
│ Analysis    │ │ Analysis    │
└─────────────┘ └─────────────┘
```

##### Image Analysis
- **Input:** อัปโหลดรูปภาพ (JPG, PNG, max 10MB)
- **Settings:**
  - Detection Level: Basic/Standard/Advanced
  - Analysis Depth: Quick/Thorough
- **Output:**
  - Deepfake Probability (%)
  - Risk Level: Low/Medium/High
  - Suspicious Regions (ถ้ามี)

##### Video Analysis
- **Input:** อัปโหลดวิดีโอ (MP4, AVI, max 100MB)
- **Process:**
  - Progress Bar การวิเคราะห์
  - Frame-by-frame Analysis
- **Output:**
  - Overall Deepfake Score
  - Timeline ของจุดน่าสงสัย
  - Key Frames ที่ตรวจพบ

---

## 4. หน้าทดสอบ Face Detector (`/test/face-detector`)

### Components หลัก

#### Header Section
- **ชื่อบริการ:** "ทดสอบ Face Detection API"
- **คำอธิบาย:** "ตรวจจับและวิเคราะห์ใบหน้าในรูปภาพ"

#### Input Section
- **Image Upload:** Drag & Drop หรือ Browse
- **Camera Capture:** ถ่ายภาพทันที
- **Demo Images:** รูปตัวอย่างให้ทดสอบ

#### Detection Options
- ☑️ **แสดง Bounding Boxes** - กรอบล้อมรอบใบหน้า
- ☑️ **แสดง Facial Landmarks** - จุดสำคัญบนใบหน้า (68 จุด)
- ☑️ **วิเคราะห์อารมณ์** - Happy, Sad, Angry, Neutral
- ☑️ **ประมาณอายุ** - Age range
- ☑️ **ระบุเพศ** - Male/Female

#### Results Display
- **ภาพที่มี Annotations:**
  - Bounding Boxes สีต่างๆ สำหรับแต่ละใบหน้า
  - Landmarks เป็นจุดสี
  - Labels แสดงข้อมูลเพิ่มเติม

- **ตารางข้อมูล:**
  ```
  Face #1: Age 25-30, Female, Happy (85%), Confidence 98.5%
  Face #2: Age 40-45, Male, Neutral (92%), Confidence 96.2%
  ```

---

## 5. หน้าทดสอบ Age & Gender Detection (`/test/age-gender-detection`)

### Components หลัก

#### Header Section
- **ชื่อบริการ:** "ทดสอบ Age & Gender Detection API"
- **คำอธิบาย:** "ประมาณการอายุและเพศจากใบหน้าด้วยความแม่นยำสูง"

#### Input Section
- **Single Face Image:** รูปภาพที่มีใบหน้าคนเดียว
- **Multiple Faces Image:** รูปภาพที่มีหลายคน
- **Live Camera:** ทดสอบแบบเรียลไทม์

#### Analysis Options
- **Age Estimation:**
  - Exact Age (เลขแน่นอน)
  - Age Range (ช่วงอายุ)
  - Age Group (เด็ก, วัยรุ่น, วัยผู้ใหญ่, ผู้สูงอายุ)

- **Gender Classification:**
  - Binary: Male/Female
  - Confidence Score

- **Additional Features:**
  - Face Quality Assessment
  - Ethnicity Estimation (ถ้ามี)

#### Results Display
```
┌─────────────────────────────────────────────┐
│ [Face Image with Bounding Box]             │
├─────────────────────────────────────────────┤
│ Age: 28 years old (±3 years)               │
│ Age Range: 25-31 years                     │
│ Age Group: Young Adult                     │
│ Gender: Female (94.2% confidence)          │
│ Quality Score: 96.8%                       │
└─────────────────────────────────────────────┘
```

---

## Common Components (ทุกหน้าทดสอบ)

### 1. API Response Panel
```json
{
  "status": "success",
  "processing_time": "1.2s",
  "api_version": "v2.1",
  "request_id": "req_12345",
  "results": { ... }
}
```

### 2. Quick Actions
- **ทดสอบอีกครั้ง:** Clear results + เริ่มใหม่
- **ลองรูปอื่น:** เปลี่ยนรูปภาพ
- **บันทึกผลลัพธ์:** Download JSON/PDF
- **แชร์ผลลัพธ์:** Copy Link

### 3. Help & Tips
- 💡 **เคล็ดลับการใช้งาน:**
  - "ใช้รูปที่มีแสงเพียงพอ"
  - "ใบหน้าควรชัดเจนและไม่มีสิ่งบัง"
  - "สำหรับวิดีโอควรมีคุณภาพ HD"

### 4. Limitations & Notice
- ⚠️ **ข้อจำกัด:**
  - ไฟล์สูงสุด: 10MB (รูป), 100MB (วิดีโอ)
  - รองรับ: JPG, PNG, MP4, AVI
  - ผลลัพธ์เก็บไว้ 24 ชั่วโมง
- 🔒 **ความเป็นส่วนตัว:**
  - "ข้อมูลทดสอบไม่ถูกเก็บถาวร"
  - "ใช้เพื่อการทดสอบเท่านั้น"

## การนำทาง (Navigation Paths)

### จากทุกหน้าทดสอบไปได้:
1. **→ `/api-testing`** - กลับรายการ AI ทั้งหมด
2. **→ `/test/other-ai`** - ทดสอบ AI อื่น (Quick Links)
3. **→ `/login`** - เข้าสู่ระบบเพื่อใช้งานจริง
4. **→ `/register`** - สมัครสมาชิก
5. **→ `/`** - กลับหน้าแรก

### หน้าอื่นมาหน้าทดสอบได้:
- **จาก `/api-testing`** → คลิกปุ่ม "ทดสอบ" แต่ละ AI
- **จาก `/`** → คลิกปุ่มทดสอบใน Features Overview
- **จาก `/login`** → คลิก "ทดสอบ AI ก่อน"

## Performance & UX

### Loading States
- **กำลังโหลด AI Model:** Skeleton placeholder
- **กำลังประมวลผล:** Progress bar + เวลาโดยประมาณ
- **กำลังอัปโหลด:** Upload progress

### Error Handling
- **ไฟล์ไม่ถูกต้อง:** "รองรับเฉพาะ JPG, PNG, MP4"
- **ไฟล์ใหญ่เกินไป:** "ขนาดไฟล์ต้องไม่เกิน 10MB"
- **ไม่พบใบหน้า:** "ไม่พบใบหน้าในภาพ กรุณาลองใหม่"
- **API Error:** "บริการไม่พร้อมใช้งาน กรุณาลองใหม่ภายหลัง"

### Mobile Optimization
- Camera Interface แบบ Full Screen
- Touch-friendly Controls
- Optimized File Upload
- Responsive Result Display
