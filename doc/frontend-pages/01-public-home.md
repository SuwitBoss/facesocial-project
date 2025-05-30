# หน้าแรก (Public Home Page)

## ข้อมูลพื้นฐาน
- **URL:** `/`
- **สถานะการเข้าถึง:** สาธารณะ (ไม่ต้องเข้าสู่ระบบ)
- **วัตถุประสงค์:** แสดงข้อมูลแพลตฟอร์ม FaceSocial และเป็นจุดเริ่มต้นสำหรับผู้ใช้

## Components หลัก

### 1. Header Navigation
- **Logo FaceSocial** - คลิกแล้วกลับหน้าแรก
- **Menu หลัก:**
  - หน้าแรก (ปัจจุบัน)
  - ทดสอบ AI
  - เข้าสู่ระบบ
  - สมัครสมาชิก

### 2. Hero Section
- **ชื่อแพลตฟอร์ม:** "FaceSocial - Social Network with AI Face Recognition"
- **คำอธิบาย:** แพลตฟอร์มโซเชียลเน็ตเวิร์กที่ใช้เทคโนโลยี AI ขั้นสูงในการจดจำใบหน้า
- **รูปภาพหลัก:** ภาพประกอบหรือวิดีโอ Demo การทำงาน
- **CTA Buttons:**
  - "ทดสอบ AI ฟรี" (Primary Button)
  - "เข้าสู่ระบบ" (Secondary Button)

### 3. AI Services Overview
แสดงบริการ AI ทั้งหมด **5 บริการ** ในรูปแบบ Grid:

#### 🤖 Face Recognition
- **คำอธิบาย:** จดจำและระบุตัวตนใบหน้าด้วย AI ที่แม่นยำสูง
- **ปุ่ม:** "ทดสอบ Face Recognition"

#### 🛡️ Face Antispoofing  
- **คำอธิบาย:** ตรวจสอบการปลอมแปลงใบหน้า ป้องกันรูปภาพและหน้ากาก
- **ปุ่ม:** "ทดสอบ Antispoofing"

#### 🎭 Deepfake Detector
- **คำอธิบาย:** ตรวจจับภาพและวิดีโอที่ถูกสร้างด้วย AI (Deepfake)
- **ปุ่ม:** "ทดสอบ Deepfake Detector"

#### 👁️ Face Detector
- **คำอธิบาย:** ตรวจจับและวิเคราะห์ใบหน้าในรูปภาพ
- **ปุ่ม:** "ทดสอบ Face Detector"

#### 👥 Age & Gender Detection
- **คำอธิบาย:** ประมาณการอายุและเพศจากใบหน้า
- **ปุ่ม:** "ทดสอบ Age & Gender"

### 4. Platform Features
- **Auto Face Tagging:** แท็กใบหน้าอัตโนมัติในโพสต์
- **Security Login:** เข้าสู่ระบบด้วยการสแกนใบหน้า
- **Smart Detection:** ตรวจจับภาพปลอมและการหลอกลวง
- **Privacy Protection:** ควบคุมข้อมูลส่วนตัวได้เต็มรูปแบบ

### 5. Call-to-Action Section
- **หัวข้อ:** "เริ่มต้นใช้งาน FaceSocial วันนี้"
- **ปุ่มหลัก:**
  - "ทดสอบ AI ทั้งหมด" → ไป `/api-testing`
  - "สมัครสมาชิกฟรี" → ไป `/register`
  - "เข้าสู่ระบบ" → ไป `/login`

### 6. Footer
- **ข้อมูลติดต่อ:** อีเมล, โทรศัพท์
- **ลิงก์สำคัญ:**
  - นโยบายความเป็นส่วนตัว
  - เงื่อนไขการใช้งาน
  - ช่วยเหลือ
  - ติดต่อเรา

## การนำทาง (Navigation Paths)

### จากหน้านี้ไปหน้าอื่นได้:
1. **→ `/api-testing`** 
   - คลิก "ทดสอบ AI" ในเมนู
   - คลิก "ทดสอบ AI ฟรี" ใน Hero
   - คลิก "ทดสอบ AI ทั้งหมด" ใน CTA
   - คลิกปุ่มทดสอบในแต่ละบริการ AI

2. **→ `/login`**
   - คลิก "เข้าสู่ระบบ" ในเมนู
   - คลิก "เข้าสู่ระบบ" ใน Hero
   - คลิก "เข้าสู่ระบบ" ใน CTA

3. **→ `/register`**
   - คลิก "สมัครสมาชิก" ในเมนู
   - คลิก "สมัครสมาชิกฟรี" ใน CTA

4. **→ `/help`**
   - คลิก "ช่วยเหลือ" ใน Footer

5. **→ `/privacy`**
   - คลิก "นโยบายความเป็นส่วนตัว" ใน Footer

6. **→ `/terms`**
   - คลิก "เงื่อนไขการใช้งาน" ใน Footer

### หน้าอื่นมาหน้านี้ได้:
- จากทุกหน้าสามารถกลับมาหน้าแรกได้โดยคลิก Logo
- จาก `/api-testing` กลับมาได้
- จาก `/login` หรือ `/register` กลับมาได้
- เมื่อ Logout จากระบบจะกลับมาหน้านี้

## การออกแบบ UI/UX

### Desktop Layout
```
┌─────────────────────────────────────────────┐
│ Header: Logo | Menu | Login | Register      │
├─────────────────────────────────────────────┤
│ Hero Section: Title + Description + CTA     │
├─────────────────────────────────────────────┤
│ AI Services Grid (2x3 or 3x2)              │
├─────────────────────────────────────────────┤
│ Platform Features                          │
├─────────────────────────────────────────────┤
│ Final CTA Section                          │
├─────────────────────────────────────────────┤
│ Footer                                     │
└─────────────────────────────────────────────┘
```

### Mobile Layout
- Header แบบ Hamburger Menu
- Hero Section ย่อให้กระชับ
- AI Services Grid แบบ 1 คอลัมน์
- ปุ่ม CTA ขนาดใหญ่เหมาะกับการแตะ

## Responsive Breakpoints
- **Desktop:** 1200px+
- **Tablet:** 768px - 1199px
- **Mobile:** < 768px

## SEO และ Meta Tags
- **Title:** "FaceSocial - AI-Powered Social Network"
- **Description:** "แพลตฟอร์มโซเชียลเน็ตเวิร์กขั้นสูงด้วยเทคโนโลยี AI Face Recognition"
- **Keywords:** "face recognition, AI, social network, deepfake detection"

## Performance Requirements
- การโหลดหน้าแรกต้องไม่เกิน 3 วินาที
- รูปภาพต้องมี Lazy Loading
- CSS และ JS ต้อง Minified
- รองรับ Caching
