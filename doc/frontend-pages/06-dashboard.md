# หน้า Dashboard หลัก (Main Dashboard)

## ข้อมูลพื้นฐาน
- **URL:** `/dashboard`
- **สถานะการเข้าถึง:** ต้องเข้าสู่ระบบ (Authentication Required)
- **วัตถุประสงค์:** หน้าหลักหลังจาก Login สำหรับ User ทั่วไป แสดงฟีดและเมนูการใช้งาน
- **หมายเหตุ:** Admin จะไปหน้า `/admin/dashboard` แทน

## Layout หลัก

### Desktop Layout (1200px+)
```
┌─────────────────────────────────────────────────────────────┐
│ Top Header: Logo | Search | Notifications | Profile Menu    │
├──────────┬──────────────────────────────────┬───────────────┤
│ Left     │ Main Feed Area                   │ Right Panel   │
│ Sidebar  │                                  │               │
│ (250px)  │                                  │ (300px)       │
│          │                                  │               │
│ Menu     │ ┌─────────────────────────────┐  │ - Quick AI    │
│ - Home   │ │ Create Post Box             │  │ - Suggestions │
│ - Profile│ └─────────────────────────────┘  │ - Trending    │
│ - AI     │                                  │ - Statistics  │
│ - Chat   │ ┌─────────────────────────────┐  │               │
│ - CCTV   │ │ Post Feed (Infinite Scroll) │  │               │
│ - More   │ │                             │  │               │
│          │ │ - User Posts                │  │               │
│          │ │ - Friends Posts             │  │               │
│          │ │ - AI Analysis Results       │  │               │
│          │ └─────────────────────────────┘  │               │
└──────────┴──────────────────────────────────┴───────────────┘
```

## Components หลัก

### 1. Top Header (ส่วนบน)

#### Left Section
- **FaceSocial Logo** - คลิกกลับ Dashboard
- **Global Search Bar**
  - Placeholder: "ค้นหาเพื่อน, โพสต์, หรือเนื้อหา..."
  - Auto-complete Suggestions
  - Search History

#### Right Section
- **Notifications Icon** 🔔
  - Badge แสดงจำนวนการแจ้งเตือนใหม่
  - Dropdown Preview การแจ้งเตือนล่าสุด
  - คลิกไป `/notifications`

- **Profile Menu Dropdown**
  - รูปโปรไฟล์ + ชื่อ
  - "ดูโปรไฟล์" → `/profile/me`
  - "แก้ไขโปรไฟล์" → `/profile/edit`
  - "การตั้งค่า" → `/settings`
  - "ออกจากระบบ" → Logout

### 2. Left Sidebar Navigation

#### Main Menu
- 🏠 **หน้าแรก** (ปัจจุบัน) - `/dashboard`
- 👤 **โปรไฟล์** - `/profile/me`
- 🤖 **AI Features** - `/ai-features`
- 💬 **แชท** - `/messages`
- 📹 **CCTV** - `/cctv` (ถ้ามีสิทธิ์)
- 🔔 **การแจ้งเตือน** - `/notifications`
- ⚙️ **การตั้งค่า** - `/settings`

#### Quick AI Actions
- 📸 **สแกนหน้า** - เปิด Face Recognition
- 🎭 **ตรวจ Deepfake** - เปิด Deepfake Detector
- 🛡️ **ตรวจความจริง** - เปิด Antispoofing

#### User Info Panel
- รูปโปรไฟล์ขนาดเล็ก
- ชื่อและ Username
- สถานะ Face Verification: ✅ Verified / ❌ Not Verified
- จำนวน: Posts, Followers, Following

### 3. Main Feed Area (กลาง)

#### Create Post Section
```
┌─────────────────────────────────────────────┐
│ 👤 [Profile Pic] "คิดอะไรอยู่ [ชื่อ]?"       │
├─────────────────────────────────────────────┤
│ 📷 Photo/Video | 🎭 AI Check | 📍 Location  │
└─────────────────────────────────────────────┘
```
- **Text Input:** "แชร์ความคิด ภาพ หรือวิดีโอ..."
- **Quick Actions:**
  - 📷 "เพิ่มรูป/วิดีโอ"
  - 🎭 "ตรวจสอบด้วย AI"
  - 📍 "เพิ่มตำแหน่ง"
  - 🏷️ "แท็กเพื่อน"
- **คลิกแล้วไป:** `/create-post`

#### Posts Feed (Infinite Scroll)
แสดงโพสต์จาก:
- โพสต์ของตัวเอง
- โพสต์จากเพื่อนที่ติดตาม
- โพสต์ที่มี Auto Face Tag ของตัวเอง
- โพสต์ที่มี AI Analysis น่าสนใจ

##### Post Card Structure
```
┌─────────────────────────────────────────────┐
│ 👤 [User] John Doe • 2 hours ago    [...] │
├─────────────────────────────────────────────┤
│ "วันนี้ไปเที่ยวกับเพื่อนๆ 😊"                │
├─────────────────────────────────────────────┤
│ [Images/Video with AI Analysis]            │
│ 🏷️ Tagged: @Alice, @Bob (Auto-detected)    │
│ 🤖 AI: 2 faces detected, No deepfake       │
├─────────────────────────────────────────────┤
│ ❤️ 15 👍 8 💬 3 📤 Share                    │
│ 💬 View all 3 comments                     │
└─────────────────────────────────────────────┘
```

**AI Features ในโพสต์:**
- **Auto Face Tagging:** แสดงคนที่ถูกแท็กอัตโนมัติ
- **AI Analysis Badge:** ✅ Safe, ⚠️ Suspicious, ❌ Fake
- **Face Count:** จำนวนใบหน้าที่ตรวจพบ
- **Age/Gender Info:** (ถ้าเปิดใช้งาน)

### 4. Right Panel (ขวา)

#### Quick AI Tools
- **🔍 Face Search**
  - "ค้นหาใบหน้าในโพสต์"
  - Upload Photo → Find Similar
- **🎭 Quick Deepfake Check**
  - Drag & Drop Image/Video
  - Instant Analysis
- **📊 My AI Usage**
  - สถิติการใช้ AI วันนี้
  - เครดิตที่เหลือ (ถ้ามี)

#### Friend Suggestions
- **"คนที่คุณอาจรู้จัก"**
- แสดงรูปโปรไฟล์ + ชื่อ
- "เหตุผล: มีเพื่อนร่วมกัน 5 คน"
- ปุ่ม "ติดตาม" / "ปิด"

#### Trending & Highlights
- **Trending Hashtags**
  - #วันนี้ (123 posts)
  - #ai_check (89 posts)
  - #face_social (67 posts)

- **AI Highlights**
  - "ตรวจพบ Deepfake 5 คลิปวันนี้"
  - "มีการแท็กหน้าคุณ 3 ครั้ง"
  - "คลิกดูรายละเอียด"

#### System Status
- **API Status:**
  - 🟢 All AI Services Online
  - 🟡 Some Services Slow
  - 🔴 Maintenance Mode
- **Response Time:** Average 1.2s

## การนำทาง (Navigation Paths)

### จากหน้านี้ไปหน้าอื่นได้:

#### Navigation Menu
1. **→ `/profile/me`** - ดูโปรไฟล์ตัวเอง
2. **→ `/ai-features`** - ศูนย์รวม AI Tools
3. **→ `/messages`** - ระบบแชท
4. **→ `/cctv`** - CCTV Monitoring (ถ้าห้ามีสิทธิ์)
5. **→ `/notifications`** - การแจ้งเตือนทั้งหมด
6. **→ `/settings`** - การตั้งค่าระบบ

#### Content Actions
7. **→ `/create-post`** - สร้างโพสต์ใหม่
8. **→ `/post/{postId}`** - ดูโพสต์แต่ละโพสต์
9. **→ `/profile/{userId}`** - ดูโปรไฟล์เพื่อน
10. **→ `/search?q=...`** - ผลการค้นหา

#### AI Actions
11. **→ `/ai/face-search`** - ค้นหาด้วยใบหน้า
12. **→ `/ai/deepfake-check`** - ตรวจสอบ Deepfake
13. **→ `/ai/face-verification`** - ยืนยันตัวตน

### หน้าอื่นมาหน้านี้ได้:
- **จาก `/login`** → หลัง Login สำเร็จ (User ทั่วไป)
- **จาก `/create-post`** → หลังโพสต์เสร็จ
- **จาก `/profile/*`** → คลิก "หน้าแรก"
- **จากหน้าอื่นๆ** → คลิก Logo หรือ "หน้าแรก"

## Responsive Design

### Tablet (768px - 1199px)
- Sidebar แบบ Collapsible
- Right Panel ซ้อนใต้ Main Feed
- แสดงเป็น 2 คอลัมน์

### Mobile (< 768px)
- ซ่อน Sidebar → แสดงเป็น Bottom Navigation
- Right Panel ซ่อน → แสดงบางส่วนใน Slide Menu
- Main Feed แสดงเต็มหน้าจอ

#### Bottom Navigation (Mobile)
```
┌─────┬─────┬─────┬─────┬─────┐
│ 🏠  │ 🔍  │ ➕  │ 💬  │ 👤  │
│Home │Find │Post │Chat │Me   │
└─────┴─────┴─────┴─────┴─────┘
```

## Real-time Features

### Live Updates
- **New Posts:** Auto-load ใหม่ทุก 30 วินาที
- **Notifications:** Real-time badge update
- **Chat Messages:** Instant notification
- **AI Processing:** Live progress bars

### WebSocket Events
- `new_post` - โพสต์ใหม่จากเพื่อน
- `face_tagged` - ถูกแท็กในรูป
- `ai_completed` - AI analysis เสร็จ
- `message_received` - ข้อความใหม่

## Performance Optimization

### Loading Strategy
- **Initial Load:** แสดง 10 โพสต์แรก
- **Infinite Scroll:** โหลดเพิ่มทีละ 5 โพสต์
- **Image Lazy Loading:** โหลดรูปเมื่อเลื่อนใกล้
- **Video Autoplay:** เมื่ออยู่ในมุมมอง

### Caching
- **Profile Data:** Cache 1 ชั่วโมง
- **AI Results:** Cache 24 ชั่วโมง
- **Static Assets:** CDN + Browser Cache

## Security Features

### Privacy Controls
- **Post Visibility:** แต่ละโพสต์ตั้งค่าได้
- **Face Tag Approval:** อนุมัติก่อนแท็ก
- **AI Opt-out:** ปิดการวิเคราะห์ AI ได้
- **Profile Privacy:** ซ่อนจากคนที่ไม่รู้จัก

### Activity Monitoring
- **Login Sessions:** ติดตามการเข้าใช้
- **Suspicious Activity:** แจ้งเตือนการใช้งานผิดปกติ
- **Face Verification:** ยืนยันตัวตนสำหรับการกระทำสำคัญ

## Analytics Tracking
- เวลาที่ใช้ในหน้า Dashboard
- Post Engagement Rate
- AI Features Usage
- Search Query Analytics
- User Journey Tracking
