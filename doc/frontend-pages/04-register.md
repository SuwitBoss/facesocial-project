# หน้าสมัครสมาชิก (Register Page)

## ข้อมูลพื้นฐาน
- **URL:** `/register`
- **สถานะการเข้าถึง:** สาธารณะ (ไม่ต้องเข้าสู่ระบบ)
- **วัตถุประสงค์:** ให้ผู้ใช้สมัครสมาชิกใหม่พร้อมระบบตรวจสอบใบหน้าและป้องกันการซ้ำกับ Admin

## ระบบป้องกันการซ้ำกับ Admin
- ระบบจะตรวจสอบ Username, Email และข้อมูลใบหน้าไม่ให้ซ้ำกับ Admin
- มีการตรวจสอบแบบเรียลไทม์ขณะที่ผู้ใช้กรอกข้อมูล
- หากพบการซ้ำจะแสดงข้อผิดพลาดทันที

## Components หลัก

### 1. Header Section
- **ชื่อหน้า:** "สมัครสมาชิก FaceSocial"
- **คำอธิบาย:** "สร้างบัญชีใหม่เพื่อเข้าใช้งานระบบ"
- **ปุ่มกลับ:** "← กลับหน้าแรก" → ไป `/`

### 2. Registration Form (Step 1: ข้อมูลพื้นฐาน)

#### ข้อมูลส่วนตัว
- **Full Name Field**
  - Label: "ชื่อ-นามสกุล"
  - Placeholder: "กรอกชื่อและนามสกุลจริง"
  - Validation: Required, 2-50 characters, Thai/English only

- **Username Field**
  - Label: "ชื่อผู้ใช้"
  - Placeholder: "เช่น john_doe หรือ สมชาย123"
  - Validation: Required, 3-20 characters, Unique
  - **Real-time Check:** ตรวจสอบการซ้ำกับ Admin และ Users อื่น
  - แสดงสถานะ: ✅ ใช้ได้ / ❌ ซ้ำ / ⏳ กำลังตรวจสอบ

- **Email Field**
  - Label: "อีเมล"
  - Placeholder: "example@domain.com"
  - Validation: Required, Valid email format, Unique
  - **Real-time Check:** ตรวจสอบการซ้ำกับ Admin
  - Email Verification: ส่ง OTP ไป Email

- **Password Field**
  - Label: "รหัสผ่าน"
  - Placeholder: "สร้างรหัสผ่านที่ปลอดภัย"
  - Validation: Min 8 characters, Must contain: A-Z, a-z, 0-9, Special chars
  - Show/Hide Password Toggle
  - Password Strength Indicator

- **Confirm Password Field**
  - Label: "ยืนยันรหัสผ่าน"
  - Validation: Must match password

#### ข้อมูลเพิ่มเติม
- **Phone Number** (Optional)
  - Label: "เบอร์โทรศัพท์"
  - Format: +66 XX-XXX-XXXX

- **Date of Birth** (Optional)
  - Date Picker
  - ใช้สำหรับเปรียบเทียบกับ Age Detection

- **Gender** (Optional)
  - Radio Buttons: ชาย / หญิง / ไม่ระบุ
  - ใช้สำหรับเปรียบเทียบกับ Gender Detection

### 3. Face Registration (Step 2: ลงทะเบียนใบหน้า)

#### วิธีการเพิ่มใบหน้า (เลือก 1 วิธี)

##### วิธี 1: 📷 สแกนหน้าผ่านกล้อง
- **Live Camera Interface:**
  - เปิดกล้อง Web/Mobile
  - Face Detection Overlay แบบเรียลไทม์
  - คำแนะนำการถ่าย: "มองตรงไปที่กล้อง"
  - Guidelines: ระยะห่าง, มุม, แสง

- **Capture Process:**
  - ถ่าย 3-5 ภาพต่อเนื่อง
  - มุมต่างๆ: หน้าตรง, เอียงซ้าย, เอียงขวา
  - Quality Check แต่ละภาพ

##### วิธี 2: 📁 อัปโหลดรูปภาพ
- **Upload Interface:**
  - Drag & Drop Area
  - File Browser
  - รองรับ: JPG, PNG (ขนาดสูงสุด 5MB)
  - Preview Gallery

- **Multiple Photos Required:**
  - อัปโหลดได้ 2-10 ภาพ
  - ภาพต้องมีใบหน้าชัดเจน
  - แต่ละภาพต้องผ่าน Face Detection

### 4. Face Verification (Step 3: ตรวจสอบความถูกต้อง)

#### การตรวจสอบภาพ
- **Individual Photo Analysis:**
  - แสดงภาพทั้งหมดที่เพิ่มเข้ามา
  - Face Detection Box บนแต่ละภาพ
  - Face Quality Score (%)
  - Gender/Age Detection Results

#### ปุ่ม "ตรวจสอบ" (ขั้นตอนสำคัญ)
เมื่อกดปุ่มนี้ ระบบจะทำการ:

1. **Face Matching Between Photos:**
   - ตรวจสอบว่าทุกภาพเป็นบุคคลเดียวกัน
   - Similarity Score ต้อง > 85%
   - แสดงผลการเปรียบเทียบแต่ละคู่ภาพ

2. **Anti-duplication Check:**
   - เปรียบเทียบกับใบหน้า Admin ในระบบ
   - เปรียบเทียบกับใบหน้า Users ที่มีอยู่แล้ว
   - ป้องกันบัญชีซ้ำ

3. **Quality Assessment:**
   - ตรวจสอบความชัดเจนของภาพ
   - ตรวจสอบแสงและมุมการถ่าย
   - ตรวจสอบการบิดเบือน

#### ผลการตรวจสอบ
- **✅ ผ่านการตรวจสอบ:**
  - "ใบหน้าทั้งหมดเป็นบุคคลเดียวกัน"
  - "ไม่พบการซ้ำกับบัญชีอื่น"
  - ปุ่ม "ยืนยันสมัครสมาชิก" จะถูกเปิดใช้งาน

- **❌ ไม่ผ่านการตรวจสอบ:**
  - "พบภาพที่เป็นคนละบุคคล"
  - "พบการซ้ำกับบัญชี Admin/User อื่น"
  - "คุณภาพภาพไม่เพียงพอ"
  - ต้องแก้ไขก่อนจึงจะสมัครได้

### 5. Terms and Conditions
- **Privacy Policy Agreement**
  - Checkbox: "ยอมรับนโยบายความเป็นส่วนตัว"
  - Link to full policy → `/privacy`

- **Terms of Service Agreement**
  - Checkbox: "ยอมรับเงื่อนไขการใช้งาน"
  - Link to full terms → `/terms`

- **Face Data Consent**
  - Checkbox: "ยินยอมให้เก็บและใช้ข้อมูลใบหน้าเพื่อการจดจำ"
  - คำอธิบาย: "ข้อมูลใบหน้าจะถูกเข้ารหัสและใช้เฉพาะการจดจำเท่านั้น"

### 6. Registration Completion
- **"ยืนยันสมัครสมาชิก" Button**
  - เปิดใช้งานเมื่อผ่านการตรวจสอบทั้งหมด
  - Loading State: "กำลังสร้างบัญชี..."
  - Success: Redirect ไป `/login` with success message

## การนำทาง (Navigation Paths)

### จากหน้านี้ไปหน้าอื่นได้:

1. **→ `/login`**
   - หลังสมัครสมาชิกสำเร็จ (Auto redirect)
   - คลิกลิงก์ "มีบัญชีแล้ว? เข้าสู่ระบบ"

2. **→ `/`** (หน้าแรก)
   - คลิกปุ่ม "← กลับหน้าแรก"

3. **→ `/privacy`**
   - คลิกลิงก์ "นโยบายความเป็นส่วนตัว"

4. **→ `/terms`**
   - คลิกลิงก์ "เงื่อนไขการใช้งาน"

5. **→ `/help/camera`**
   - คลิก "ช่วยเหลือการใช้กล้อง" (เมื่อมีปัญหากล้อง)

6. **→ `/api-testing`**
   - คลิก "ทดสอบ AI ก่อน" (Optional link)

### หน้าอื่นมาหน้านี้ได้:
- **จาก `/`** → คลิก "สมัครสมาชิก"
- **จาก `/login`** → คลิก "ยังไม่มีบัญชี? สมัครสมาชิก"
- **จาก `/api-testing`** → หลังทดสอบ AI แล้วต้องการสมัคร

## Error Handling

### ข้อผิดพลาดข้อมูลพื้นฐาน:
- **"ชื่อผู้ใช้นี้ถูกใช้แล้ว"** (รวมถึงซ้ำกับ Admin)
- **"อีเมลนี้ถูกใช้แล้ว"** (รวมถึงซ้ำกับ Admin)
- **"รหัสผ่านไม่ปลอดภัยเพียงพอ"**
- **"รหัสผ่านไม่ตรงกัน"**

### ข้อผิดพลาดใบหน้า:
- **"ไม่พบใบหน้าในภาพ"**
- **"ภาพไม่ชัดเจนเพียงพอ"**
- **"พบใบหน้าหลายคนในภาพเดียว"**
- **"ภาพทั้งหมดไม่เป็นบุคคลเดียวกัน"**
- **"ใบหน้านี้มีในระบบแล้ว"** (ซ้ำกับ Admin หรือ User อื่น)

### ข้อผิดพลาดทางเทคนิค:
- **"ไม่สามารถเข้าถึงกล้องได้"**
- **"บริการ AI ไม่พร้อมใช้งาน"**
- **"ไฟล์รูปภาพไม่ถูกต้อง"**
- **"ขนาดไฟล์ใหญ่เกินไป"**

## การออกแบบ UI/UX

### Desktop Layout (3 Steps)
```
Step 1: Basic Info
┌─────────────────────────────────────────────┐
│ Personal Information Form                  │
│ ├── Name, Username, Email                  │
│ ├── Password & Confirmation                │
│ └── Optional: Phone, DOB, Gender           │
└─────────────────────────────────────────────┘

Step 2: Face Registration  
┌─────────────────────────────────────────────┐
│ ┌─────────────┐ ┌─────────────┐            │
│ │📷 Camera    │ │📁 Upload    │ (Tabs)     │
│ │   Scan      │ │   Photos    │            │
│ └─────────────┘ └─────────────┘            │
│ Face Capture/Upload Interface              │
└─────────────────────────────────────────────┘

Step 3: Verification
┌─────────────────────────────────────────────┐
│ Photo Gallery + Analysis Results           │
│ [Check Button] → [Confirmation Button]     │
└─────────────────────────────────────────────┘
```

### Mobile Layout
- Form แบบ Single Column
- Camera Interface แบบ Full Screen
- Step Indicator ด้านบน
- กริดภาพแบบ 2x2

## Security Measures

### ข้อมูลส่วนตัว:
- **Email Verification:** ส่ง OTP ยืนยัน Email
- **Password Hashing:** bcrypt with salt
- **Rate Limiting:** จำกัดการสมัครต่อ IP
- **CAPTCHA:** หลังพยายามสมัครผิดหลายครั้ง

### ข้อมูลใบหน้า:
- **Face Embedding Encryption:** เข้ารหัสข้อมูลใบหน้า
- **Duplicate Prevention:** ตรวจสอบการซ้ำแบบเรียลไทม์
- **Quality Standards:** มาตรฐานคุณภาพภาพ
- **Data Retention:** กำหนดระยะเวลาเก็บข้อมูล

## Performance Requirements
- **Form Validation แบบ Real-time**
- **Camera เริ่มทำงานภายใน 2 วินาที**
- **Face Detection ทำงานภายใน 1 วินาที**
- **การตรวจสอบทั้งหมดเสร็จภายใน 10 วินาที**

## Analytics Tracking
- ติดตามสัดส่วนการใช้ Camera vs Upload
- วัดเวลาเฉลี่ยในแต่ละ Step
- Success Rate ของการสมัครสมาชิก
- จำนวนความพยายามที่ไม่สำเร็จและสาเหตุ
