# หน้าทดสอบ AI (API Testing Overview)

## ข้อมูลพื้นฐาน
- **URL:** `/api-testing`
- **สถานะการเข้าถึง:** สาธารณะ (ไม่ต้องเข้าสู่ระบบ)
- **วัตถุประสงค์:** แสดงรายการ AI Services ทั้งหมดและให้ผู้ใช้ทดสอบแต่ละบริการ

## Components หลัก

### 1. Header Section
- **ชื่อหน้า:** "ศูนย์ทดสอบ AI Services"
- **คำอธิบาย:** "ทดสอบความสามารถของ AI ทั้งหมดได้ฟรี ไม่ต้องสมัครสมาชิก"
- **ปุ่มกลับ:** "← กลับหน้าแรก" → ไป `/`

### 2. System Status Overview
- **ข้อมูลสถานะรวม:**
  - จำนวน AI Services: 5 บริการ
  - สถานะระบบ: 🟢 Online / 🟡 บางบริการช้า / 🔴 Maintenance
  - เวลาตอบสนองเฉลี่ย: XX ms
  - ผู้ใช้ทดสอบวันนี้: XXX คน

### 3. AI Services Grid (5 การ์ด)

#### 🤖 Card 1: Face Recognition
- **ชื่อบริการ:** Face Recognition API
- **คำอธิบาย:** จดจำและระบุตัวตนใบหน้าด้วยความแม่นยำสูง
- **สถานะ:** 🟢 Online / 🔴 Offline / 🟡 Slow
- **เวลาตอบสนอง:** XXX ms
- **ปุ่มทดสอบ:** "ทดสอบ Face Recognition" → ไป `/test/face-recognition`
- **ข้อมูลเพิ่มเติม:**
  - Accuracy: 99.5%
  - รองรับ: JPG, PNG, MP4
  - ขนาดไฟล์สูงสุด: 10MB

#### 🛡️ Card 2: Face Antispoofing
- **ชื่อบริการ:** Face Antispoofing API
- **คำอธิบาย:** ตรวจสอบการปลอมแปลงใบหน้า ป้องกันรูปภาพและหน้ากาก
- **สถานะ:** 🟢 Online / 🔴 Offline / 🟡 Slow
- **เวลาตอบสนอง:** XXX ms
- **ปุ่มทดสอบ:** "ทดสอบ Antispoofing" → ไป `/test/face-antispoofing`
- **ข้อมูลเพิ่มเติม:**
  - Detection Rate: 98.2%
  - รองรับ: Live Video, Images
  - เทคนิค: Liveness Detection

#### 🎭 Card 3: Deepfake Detector
- **ชื่อบริการ:** Deepfake Detection API
- **คำอธิบาย:** ตรวจจับภาพและวิดีโอที่ถูกสร้างด้วย AI
- **สถานะ:** 🟢 Online / 🔴 Offline / 🟡 Slow
- **เวลาตอบสนอง:** XXX ms
- **ปุ่มทดสอบ:** "ทดสอบ Deepfake Detector" → ไป `/test/deepfake-detector`
- **ข้อมูลเพิ่มเติม:**
  - Detection Rate: 97.8%
  - รองรับ: Video, Image
  - เวลาประมวลผล: 1-30 วินาที

#### 👁️ Card 4: Face Detector
- **ชื่อบริการ:** Face Detection API
- **คำอธิบาย:** ตรวจจับและวิเคราะห์ใบหน้าในรูปภาพ
- **สถานะ:** 🟢 Online / 🔴 Offline / 🟡 Slow
- **เวลาตอบสนอง:** XXX ms
- **ปุ่มทดสอบ:** "ทดสอบ Face Detector" → ไป `/test/face-detector`
- **ข้อมูลเพิ่มเติม:**
  - Detection Accuracy: 99.1%
  - รองรับ: หลายใบหน้าในภาพเดียว
  - ข้อมูล: Landmarks, Emotions

#### 👥 Card 5: Age & Gender Detection
- **ชื่อบริการ:** Age & Gender Detection API
- **คำอธิบาย:** ประมาณการอายุและเพศจากใบหน้า
- **สถานะ:** 🟢 Online / 🔴 Offline / 🟡 Slow
- **เวลาตอบสนอง:** XXX ms
- **ปุ่มทดสอบ:** "ทดสอบ Age & Gender" → ไป `/test/age-gender-detection`
- **ข้อมูลเพิ่มเติม:**
  - Age Accuracy: ±3 ปี
  - Gender Accuracy: 96.5%
  - ช่วงอายุ: 3-100 ปี

### 4. Quick Actions Panel
- **ปุ่ม "ทดสอบทั้งหมด":** เปิดหน้าต่างใหม่ทดสอบ AI ทั้ง 5 ตัวพร้อมกัน
- **ปุ่ม "ดูสถานะ API":** → ไป `/api-status` (หน้าติดตามสถานะแบบเรียลไทม์)
- **ปุ่ม "เข้าสู่ระบบ":** → ไป `/login` (สำหรับใช้งานจริง)

### 5. Usage Statistics
- **สถิติการใช้งานวันนี้:**
  - Face Recognition: XXX ครั้ง
  - Antispoofing: XXX ครั้ง
  - Deepfake Detection: XXX ครั้ง
  - Face Detection: XXX ครั้ง
  - Age & Gender: XXX ครั้ง

### 6. Information Panel
- **คำแนะนำการใช้งาน:**
  - "ควรใช้ภาพที่มีแสงเพียงพอ"
  - "ใบหน้าควรชัดเจนและไม่มีสิ่งบัง"
  - "วิดีโอควรมีคุณภาพ HD ขึ้นไป"
- **ข้อจำกัด:**
  - ไฟล์สูงสุด: 10MB
  - รองรับ: JPG, PNG, MP4, AVI
  - ผลลัพธ์ถูกเก็บ 24 ชั่วโมง

## การนำทาง (Navigation Paths)

### จากหน้านี้ไปหน้าอื่นได้:
1. **→ `/`** (หน้าแรก)
   - คลิกปุ่ม "← กลับหน้าแรก"
   - คลิก Logo (ถ้ามี)

2. **→ `/test/face-recognition`**
   - คลิก "ทดสอบ Face Recognition"

3. **→ `/test/face-antispoofing`**
   - คลิก "ทดสอบ Antispoofing"

4. **→ `/test/deepfake-detector`**
   - คลิก "ทดสอบ Deepfake Detector"

5. **→ `/test/face-detector`**
   - คลิก "ทดสอบ Face Detector"

6. **→ `/test/age-gender-detection`**
   - คลิก "ทดสอบ Age & Gender"

7. **→ `/api-status`**
   - คลิก "ดูสถานะ API"

8. **→ `/login`**
   - คลิก "เข้าสู่ระบบ"

### หน้าอื่นมาหน้านี้ได้:
- **จาก `/`** → คลิก "ทดสอบ AI" ในเมนูหรือปุ่ม CTA
- **จาก `/test/*`** → คลิกปุ่ม "กลับรายการ AI" หรือ "ทดสอบ AI อื่น"
- **จาก `/login`** → คลิก "ทดสอบ AI ก่อน"

## การออกแบบ UI/UX

### Desktop Layout
```
┌─────────────────────────────────────────────┐
│ Header: Title + Status + Back Button        │
├─────────────────────────────────────────────┤
│ System Overview Panel                      │
├─────────────────────────────────────────────┤
│ ┌─────────┐ ┌─────────┐ ┌─────────┐        │
│ │ AI Card │ │ AI Card │ │ AI Card │        │
│ │    1    │ │    2    │ │    3    │        │
│ └─────────┘ └─────────┘ └─────────┘        │
│ ┌─────────┐ ┌─────────┐                    │
│ │ AI Card │ │ AI Card │                    │
│ │    4    │ │    5    │                    │
│ └─────────┘ └─────────┘                    │
├─────────────────────────────────────────────┤
│ Quick Actions + Statistics                 │
└─────────────────────────────────────────────┘
```

### Mobile Layout
- Header ย่อขนาด
- AI Cards แสดงแบบ 1 คอลัมน์
- สถิติแสดงแบบ Horizontal Scroll

## Real-time Features
- สถานะ API อัปเดตแบบเรียลไทม์ (WebSocket)
- เวลาตอบสนองแสดงผลแบบ Live
- จำนวนผู้ใช้ทดสอบอัปเดตทุก 30 วินาที

## Error Handling
- **เมื่อ API ล่ม:** แสดงสถานะ 🔴 Offline + เวลาที่คาดว่าจะกลับมา
- **เมื่อเครือข่ายขัดข้อง:** แสดงข้อความ "ตรวจสอบการเชื่อมต่อ"
- **เมื่อโหลดช้า:** แสดง Loading Spinner ที่การ์ด

## Analytics Tracking
- ติดตามการคลิกปุ่มทดสอบแต่ละ AI
- วัดเวลาที่ผู้ใช้อยู่ในหน้า
- สถิติ AI ไหนได้รับความนิยมมากที่สุด
