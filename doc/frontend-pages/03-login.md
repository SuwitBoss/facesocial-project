# หน้าเข้าสู่ระบบ (Login Page)

## ข้อมูลพื้นฐาน
- **URL:** `/login`
- **สถานะการเข้าถึง:** สาธารณะ (ไม่ต้องเข้าสู่ระบบ)
- **วัตถุประสงค์:** ให้ผู้ใช้เข้าสู่ระบบด้วยวิธีการต่างๆ และระบบตรวจสอบสิทธิ์อัตโนมัติ

## ระบบการตรวจสอบสิทธิ์
ระบบจะตรวจสอบข้อมูลที่ผู้ใช้กรอกและตัดสินใจเส้นทางอัตโนมัติ:
- **ถ้าตรงกับ Admin** → ไป Admin Dashboard
- **ถ้าตรงกับ User** → ไป User Dashboard  
- **ถ้าไม่ตรงกับทั้งสอง** → แสดงข้อผิดพลาด

## Components หลัก

### 1. Header Section
- **ชื่อหน้า:** "เข้าสู่ระบบ FaceSocial"
- **คำอธิบาย:** "เลือกวิธีการเข้าสู่ระบบที่คุณต้องการ"
- **ปุ่มกลับ:** "← กลับหน้าแรก" → ไป `/`

### 2. Login Methods Selector
Tab สำหรับเลือกวิธีการ Login:

#### Tab 1: 🔑 Login ธรรมดา (Password Login)
- **Username/Email Field**
  - Label: "ชื่อผู้ใช้หรือ Email"
  - Placeholder: "กรอกชื่อผู้ใช้หรือ Email"
  - Validation: Required, Email format (ถ้าเป็น Email)

- **Password Field**
  - Label: "รหัสผ่าน"
  - Placeholder: "กรอกรหัสผ่าน"
  - Show/Hide Password Toggle
  - Validation: Required, Min 6 characters

- **Options**
  - Checkbox: "จำฉันไว้ (Remember Me)"
  - Link: "ลืมรหัสผ่าน?" → ไป `/forgot-password`

- **Login Button**
  - Text: "เข้าสู่ระบบ"
  - Loading State: "กำลังตรวจสอบ..."
  - Action: ส่งข้อมูลไปตรวจสอบและ Route อัตโนมัติ

#### Tab 2: 📷 Face Login (เข้าสู่ระบบด้วยใบหน้า)

**ขั้นตอนการ Face Login (3 ขั้นตอนตามลำดับ):**

##### ขั้นที่ 1: Deepfake Detection 🎭
- **Camera Interface:**
  - Live Camera View
  - Face Detection Overlay
  - กรอบสี่เหลี่ยมแสดงตำแหน่งใบหน้า
  - แสดงสถานะ: "กำลังตรวจสอบ Deepfake..."

- **Instructions:**
  - "โปรดมองตรงไปที่กล้อง"
  - "หลีกเลี่ยงการใช้รูปภาพหรือหน้าจอ"
  - "ควรอยู่ในที่มีแสงเพียงพอ"

- **Results:**
  - ✅ **ผ่าน:** "ตรวจสอบ Deepfake สำเร็จ ไปขั้นต่อไป"
  - ❌ **ไม่ผ่าน:** "ตรวจพบภาพที่อาจเป็น Deepfake กรุณาลองใหม่"
  - **Action:** Auto proceed หรือ Retry

##### ขั้นที่ 2: Face Antispoofing 🛡️
- **Liveness Detection:**
  - คำสั่งสุ่ม: "กรุณาพยักหน้า", "หันซ้าย", "หันขวา", "กระพริบตา"
  - Progress Bar แสดงขั้นตอน
  - Real-time Feedback

- **Instructions:**
  - "ทำตามคำสั่งที่แสดงบนหน้าจอ"
  - "เคลื่อนไหวช้าๆ และชัดเจน"
  - "ใบหน้าต้องอยู่ในกรอบตลอดเวลา"

- **Results:**
  - ✅ **ผ่าน:** "ยืนยันเป็นบุคคลจริง ไปขั้นต่อไป"
  - ❌ **ไม่ผ่าน:** "ไม่สามารถยืนยันความมีชีวิตได้ กรุณาลองใหม่"

##### ขั้นที่ 3: Face Recognition 🤖
- **Face Matching:**
  - ถ่ายภาพใบหน้าชัดเจน
  - เปรียบเทียบกับฐานข้อมูล
  - แสดงสถานะ: "กำลังค้นหาใบหน้าในระบบ..."

- **Results:**
  - ✅ **พบข้อมูล:** "ยินดีต้อนรับ [ชื่อผู้ใช้]"
  - ❌ **ไม่พบ:** "ไม่พบข้อมูลใบหน้านี้ในระบบ"
  - **Confidence Score:** แสดง % ความมั่นใจ

### 3. Alternative Options
- **ปุ่ม "ใช้รหัสผ่านแทน":** สลับไป Password Login
- **ปุ่ม "ทดสอบ AI ก่อน":** → ไป `/api-testing`
- **ปุ่ม "สมัครสมาชิก":** → ไป `/register`

### 4. System Status Indicator
- **AI Services Status:**
  - 🟢 "บริการ AI พร้อมใช้งาน"
  - 🟡 "บริการ AI ช้าผิดปกติ" 
  - 🔴 "บริการ AI ไม่พร้อมใช้งาน (ใช้รหัสผ่านแทน)"

## การนำทาง (Navigation Paths)

### เส้นทางจากหน้านี้:

#### เมื่อ Login สำเร็จ:
1. **→ `/admin/dashboard`**
   - เมื่อระบบตรวจสอบแล้วพบว่าเป็น Admin
   - แสดงข้อความ: "ยินดีต้อนรับ Admin"

2. **→ `/dashboard`**
   - เมื่อระบบตรวจสอบแล้วพบว่าเป็น User ทั่วไป
   - แสดงข้อความ: "ยินดีต้อนรับ [ชื่อผู้ใช้]"

#### เส้นทางอื่นๆ:
3. **→ `/`** (หน้าแรก)
   - คลิกปุ่ม "← กลับหน้าแรก"

4. **→ `/register`**
   - คลิก "สมัครสมาชิก"
   - คลิกลิงก์ "ยังไม่มีบัญชี?"

5. **→ `/forgot-password`**
   - คลิก "ลืมรหัสผ่าน?"

6. **→ `/api-testing`**
   - คลิก "ทดสอบ AI ก่อน"

7. **→ `/login/password-login`** (Emergency Fallback)
   - เมื่อ AI Services ล่มหมด
   - บังคับใช้ Password Login เท่านั้น

### หน้าอื่นมาหน้านี้ได้:
- **จาก `/`** → คลิก "เข้าสู่ระบบ" 
- **จาก `/register`** → หลังสมัครเสร็จ
- **จาก `/api-testing`** → คลิก "เข้าสู่ระบบ"
- **จากหน้าที่ต้อง Login** → Redirect กลับมา

## Error Handling

### Password Login Errors:
- **"ชื่อผู้ใช้หรือรหัสผ่านไม่ถูกต้อง"**
- **"บัญชีถูกระงับ กรุณาติดต่อ Admin"**
- **"กรุณากรอกข้อมูลให้ครบถ้วน"**

### Face Login Errors:
- **Deepfake Detection Failed:**
  - "ตรวจพบภาพที่อาจเป็น Deepfake"
  - "กรุณาใช้ใบหน้าจริงและลองใหม่"

- **Antispoofing Failed:**
  - "ไม่สามารถยืนยันความมีชีวิตได้"
  - "กรุณาทำตามคำสั่งและลองใหม่"

- **Face Recognition Failed:**
  - "ไม่พบข้อมูลใบหน้านี้ในระบบ"
  - "คุณภาพภาพไม่เพียงพอ กรุณาลองใหม่"

### Technical Errors:
- **"ไม่สามารถเข้าถึงกล้องได้"** → แสดงวิธีแก้ไข
- **"บริการ AI ไม่พร้อมใช้งาน"** → เปลี่ยนไป Password Login
- **"เชื่อมต่อเซิร์ฟเวอร์ไม่ได้"** → ปุ่ม Retry

## การออกแบบ UI/UX

### Desktop Layout
```
┌─────────────────────────────────────────────┐
│ Header: Title + Back Button                │
├─────────────────────────────────────────────┤
│ ┌─────────────┐ ┌─────────────┐            │
│ │🔑 Password  │ │📷 Face Scan │ (Tabs)     │
│ │   Login     │ │   Login     │            │
│ └─────────────┘ └─────────────┘            │
├─────────────────────────────────────────────┤
│ Login Form / Camera Interface              │
│                                           │
│ (ขึ้นกับ Tab ที่เลือก)                      │
├─────────────────────────────────────────────┤
│ Alternative Options + System Status        │
└─────────────────────────────────────────────┘
```

### Face Login Process
```
Step 1: Deepfake → Step 2: Antispoofing → Step 3: Recognition
  🎭              🛡️                    🤖
 [Camera]         [Liveness]            [Matching]
```

## Security Measures
- **Rate Limiting:** จำกัดความพยายาม Login
- **CAPTCHA:** หลังพยายาม Login ผิด 3 ครั้ง
- **Session Timeout:** Session หมดอายุหลัง 24 ชั่วโมง
- **IP Tracking:** บันทึก IP ของการ Login
- **Device Fingerprinting:** จดจำอุปกรณ์ที่เคย Login

## Performance Requirements
- **Face Login ทั้งสาม Step ต้องเสร็จภายใน 15 วินาที**
- **Password Login ต้องเสร็จภายใน 3 วินาที**
- **Camera ต้องเริ่มทำงานภายใน 2 วินาที**

## Analytics Tracking
- ติดตามสัดส่วนการใช้ Password vs Face Login
- วัดเวลาเฉลี่ยของแต่ละ Step ใน Face Login  
- Success Rate ของ Face Login แต่ละขั้นตอน
