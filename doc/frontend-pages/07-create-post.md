# หน้าสร้างโพสต์ (Create Post Page)

## ภาพรวม
หน้าสร้างโพสต์ที่มีระบบ Auto Face Tagging ขั้นสูง รองรับการอัปโหลดรูปภาพ/วิดีโอ และการติดแท็กใบหน้าอัตโนมัติพร้อมการจัดเก็บ Face Embedding เพื่อความแม่นยำที่ดีขึ้น

## URL และการนำทาง
- **URL หลัก**: `/create-post`
- **เข้าถึงได้จาก**: Dashboard, Navbar, Profile Page
- **สิทธิ์การเข้าถึง**: User และ Admin (ต้อง Login)
- **การ Redirect**: หากไม่ได้ Login จะเปลี่ยนเส้นทางไปหน้า Login

## โครงสร้างหน้า

### 1. Header Section
```
┌─────────────────────────────────────────┐
│  [← กลับ]     สร้างโพสต์     [บันทึกฉบับ] │
└─────────────────────────────────────────┘
```
- ปุ่มกลับ: กลับไปหน้าก่อนหน้า
- หัวข้อหน้า: "สร้างโพสต์ใหม่"
- ปุ่มบันทึกฉบับ: บันทึกเป็น Draft

### 2. Profile Header
```
┌─────────────────────────────────────────┐
│ [Avatar] John Doe                       │
│         @john_doe                       │
│ [🌍 สาธารณะ ▼] [📍 เพิ่มตำแหน่ง]       │
└─────────────────────────────────────────┘
```
- แสดงรูปโปรไฟล์และชื่อผู้ใช้
- ตัวเลือกความเป็นส่วนตัว: สาธารณะ, เพื่อน, เฉพาะตัว
- ตัวเลือกเพิ่มตำแหน่งที่ตั้ง

### 3. Content Input Section
```
┌─────────────────────────────────────────┐
│ คุณกำลังคิดอะไรอยู่?                      │
│                                         │
│ [กล่องข้อความ - ขยายได้]                  │
│                                         │
│ ─────────────────────────────────────── │
│ [📷] [🎥] [😊] [📍] [#] [👤]             │
└─────────────────────────────────────────┘
```
- กล่องข้อความแบบ Rich Text Editor
- ปุ่มเครื่องมือ: รูปภาพ, วิดีโอ, อีโมจิ, ตำแหน่ง, แฮชแท็ก, แท็กคน

### 4. Media Upload Section
```
┌─────────────────────────────────────────┐
│         ลากไฟล์มาวางที่นี่               │
│    หรือ [เลือกไฟล์] จากอุปกรณ์           │
│                                         │
│ รองรับ: JPG, PNG, GIF, MP4, MOV         │
│ ขนาดสูงสุด: 100MB ต่อไฟล์                │
└─────────────────────────────────────────┘
```

### 5. Auto Face Tagging Section (เมื่อมีรูปภาพ)
```
┌─────────────────────────────────────────┐
│ 🤖 ระบบ AI กำลังวิเคราะห์ใบหน้า...      │
│ ┌─────────────────────────────────────┐ │
│ │  [รูปภาพที่อัปโหลด]                 │ │
│ │  📍 พบใบหน้า 3 คน                  │ │
│ │                                     │ │
│ │  ▢ Person 1 (การคาดเดา: 95%)       │ │
│ │  ▢ Person 2 (ไม่รู้จัก)            │ │
│ │  ▢ Person 3 (การคาดเดา: 87%)       │ │
│ └─────────────────────────────────────┘ │
│                                         │
│ [✓ ยอมรับทั้งหมด] [แก้ไขแท็ก]          │
└─────────────────────────────────────────┘
```

### 6. Face Tag Management
```
┌─────────────────────────────────────────┐
│ จัดการแท็กใบหน้า                        │
│                                         │
│ 👤 แท็กที่ยืนยันแล้ว:                   │
│ • @john_smith (95% แม่นยำ)              │
│ • @alice_wong (87% แม่นยำ)              │
│                                         │
│ 🔍 ใบหน้าที่ไม่รู้จัก:                  │
│ • ใบหน้า #1 [🏷️ เพิ่มแท็ก]             │
│                                         │
│ ⚙️ การตั้งค่า:                          │
│ ☑️ แท็กอัตโนมัติสำหรับเพื่อน            │
│ ☑️ แจ้งเตือนเมื่อถูกแท็ก                │
│ ☐ แท็กใบหน้าของฉันอัตโนมัติ            │
└─────────────────────────────────────────┘
```

### 7. Advanced Options
```
┌─────────────────────────────────────────┐
│ ตัวเลือกขั้นสูง                         │
│                                         │
│ ☐ ปิดการแสดงความคิดเห็น                │
│ ☐ เปิดใช้งานการแจ้งเตือน               │
│ ☐ บันทึกเป็น Draft                     │
│ ☐ กำหนดเวลาการโพสต์                   │
│                                         │
│ 🔒 ความปลอดภัย:                        │
│ ☑️ ตรวจสอบ Content Moderation          │
│ ☑️ สแกนหา Inappropriate Content        │
└─────────────────────────────────────────┘
```

### 8. Action Buttons
```
┌─────────────────────────────────────────┐
│              [ยกเลิก] [โพสต์]           │
└─────────────────────────────────────────┘
```

## ระบบ Auto Face Tagging

### 1. ขั้นตอนการทำงาน
1. **การอัปโหลดไฟล์**: ตรวจสอบประเภทและขนาดไฟล์
2. **Face Detection**: ใช้ AI Service ตรวจจับใบหน้าในรูปภาพ
3. **Face Recognition**: เปรียบเทียบกับ Face Database
4. **Embedding Storage**: บันทึก Face Embedding สำหรับการเรียนรู้
5. **Tag Suggestion**: แสดงผลการแนะนำแท็ก
6. **User Confirmation**: ให้ผู้ใช้ยืนยันหรือแก้ไข

### 2. การจัดเก็บ Face Embedding
```json
{
  "post_id": "post_123",
  "faces": [
    {
      "face_id": "face_001",
      "user_id": "user_456",
      "embedding": [0.1, 0.2, ...], // 512-dimension vector
      "confidence": 0.95,
      "bounding_box": {
        "x": 100, "y": 50,
        "width": 80, "height": 100
      },
      "manually_verified": true
    }
  ]
}
```

### 3. การปรับปรุงความแม่นยำ
- **Feedback Loop**: เรียนรู้จากการยืนยันของผู้ใช้
- **Incremental Learning**: ปรับปรุง Model จากข้อมูลใหม่
- **False Positive Handling**: จัดการกรณีระบุผิด
- **Privacy Protection**: ไม่เก็บ Embedding ของคนที่ไม่ยินยอม

## ฟีเจอร์เพิ่มเติม

### 1. Rich Media Support
- **Multi-Image Posts**: อัปโหลดรูปภาพหลายรูป
- **Video Posts**: รองรับวิดีโอพร้อม Thumbnail
- **GIF Support**: รองรับภาพเคลื่อนไหว
- **360° Photos**: รองรับรูปภาพ 360 องศา

### 2. Content Enhancement
- **Filters & Effects**: เอฟเฟกต์และฟิลเตอร์ภาพ
- **Text Overlay**: เพิ่มข้อความลงบนภาพ
- **Stickers**: สติกเกอร์และ Emoji
- **Crop & Rotate**: แก้ไขรูปภาพพื้นฐาน

### 3. Social Features
- **Tag Friends**: แท็กเพื่อนในโพสต์
- **Check-in**: เช็คอินสถานที่
- **Hashtags**: ระบบแฮชแท็กอัตโนมัติ
- **Cross-platform Sharing**: แชร์ไปยัง Social Media อื่น

## การจัดการข้อผิดพลาด

### 1. Upload Errors
```
❌ ไฟล์ขนาดใหญ่เกินไป (สูงสุด 100MB)
❌ ประเภทไฟล์ไม่รองรับ
❌ การเชื่อมต่ออินเทอร์เน็ตขาดหาย
❌ พื้นที่จัดเก็บเต็ม
```

### 2. AI Processing Errors
```
⚠️ ระบบ AI ไม่สามารถวิเคราะห์ได้
⚠️ ใบหน้าเบลอหรือมืดเกินไป
⚠️ รูปภาพคุณภาพต่ำ
⚠️ ไม่พบใบหน้าในรูปภาพ
```

### 3. Content Moderation
```
🚫 เนื้อหาไม่เหมาะสม
🚫 ตรวจพบเนื้อหาที่อาจละเมิด
🚫 รูปภาพมีเนื้อหาสำหรับผู้ใหญ่
```

## การออกแบบ Responsive

### Mobile View (320px - 768px)
- การ์ดแบบเต็มหน้าจอ
- ปุ่มขนาดใหญ่สำหรับสัมผัส
- เมนูแบบ Collapsible
- Upload ผ่าน Camera หรือ Gallery

### Tablet View (768px - 1024px)
- Layout แบบ 2 คอลัมน์
- พื้นที่ Preview ขนาดใหญ่
- Sidebar สำหรับ Options

### Desktop View (1024px+)
- Layout แบบ 3 คอลัมน์
- Multi-panel Interface
- Drag & Drop Upload
- Keyboard Shortcuts

## Performance Optimization

### 1. Image Processing
- **Client-side Resize**: ลดขนาดก่อนอัปโหลด
- **Progressive Upload**: อัปโหลดแบบทีละส่วน
- **Thumbnail Generation**: สร้าง Thumbnail อัตโนมัติ
- **CDN Integration**: ใช้ CDN สำหรับ Media

### 2. AI Processing
- **Batch Processing**: ประมวลผลหลายรูปพร้อมกัน
- **Background Processing**: ประมวลผลใน Background
- **Caching**: แคช Face Recognition Results
- **Queue Management**: จัดการคิวการประมวลผล

## Security Considerations

### 1. File Security
- **Virus Scanning**: สแกนไฟล์หาไวรัส
- **File Type Validation**: ตรวจสอบประเภทไฟล์
- **Content Filtering**: กรองเนื้อหาไม่เหมาะสม
- **Upload Limits**: จำกัดขนาดและจำนวนไฟล์

### 2. Privacy Protection
- **Face Data Encryption**: เข้ารหัสข้อมูลใบหน้า
- **Consent Management**: จัดการความยินยอม
- **Data Retention**: นีติการเก็บข้อมูล
- **Right to Delete**: สิทธิ์ลบข้อมูล

## Accessibility Features

- **Screen Reader Support**: รองรับ Screen Reader
- **Keyboard Navigation**: นำทางด้วยคีย์บอร์ด
- **High Contrast Mode**: โหมดความคมชัดสูง
- **Voice Commands**: คำสั่งเสียง
- **Text Size Adjustment**: ปรับขนาดตัวอักษร

## Integration Points

### Backend APIs
- `POST /api/posts` - สร้างโพสต์ใหม่
- `POST /api/upload` - อัปโหลดไฟล์
- `POST /api/ai/face-detection` - ตรวจจับใบหน้า
- `POST /api/ai/face-recognition` - จดจำใบหน้า
- `GET /api/users/search` - ค้นหาผู้ใช้สำหรับแท็ก

### External Services
- **Cloud Storage**: เก็บไฟล์ Media
- **CDN**: จัดส่งเนื้อหา
- **AI Services**: ประมวลผลภาพ
- **Content Moderation**: ตรวจสอบเนื้อหา

## Future Enhancements

1. **AR Filters**: ฟิลเตอร์ Augmented Reality
2. **Voice Posts**: โพสต์เสียง
3. **Live Streaming**: ถ่ายทอดสด
4. **Collaborative Posts**: โพสต์ร่วมกัน
5. **AI-Generated Content**: เนื้อหาที่ AI สร้าง
6. **3D Content**: เนื้อหา 3 มิติ
7. **VR Integration**: รองรับ Virtual Reality
8. **Blockchain Verification**: การยืนยันด้วย Blockchain
