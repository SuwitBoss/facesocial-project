# Database Schema Design สำหรับ FaceSocial AI Services

## 1. ภาพรวมโครงสร้างฐานข้อมูล

### Primary Database: PostgreSQL
```sql
-- หลักการออกแบบ:
-- 1. แยกข้อมูล AI แต่ละบริการเป็น schema ต่างหาก
-- 2. ใช้ UUID เป็น primary key ทั้งหมด
-- 3. เก็บ audit trail ครบถ้วน
-- 4. ออกแบบให้รองรับ GDPR compliance
-- 5. Optimized สำหรับ high-performance queries
```

### Vector Database: Milvus
```yaml
# สำหรับ Face Embeddings และ Similarity Search
collections:
  - face_embeddings_512d    # ArcFace embeddings
  - face_embeddings_256d    # AdaFace embeddings
  - deepfake_features      # Deepfake detection features
```

### Cache Layer: Redis
```yaml
# สำหรับ caching และ real-time data
keys:
  - "face_cache:{user_id}"     # Face recognition cache
  - "session:{session_id}"     # Active sessions
  - "rate_limit:{ip}"          # Rate limiting
```

## 2. Core Schema Structure

### 2.1 Users and Authentication
```sql
-- Schema: core
CREATE SCHEMA IF NOT EXISTS core;

-- ตาราง users (เชื่อมต่อกับ main user service)
CREATE TABLE core.users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL UNIQUE, -- Reference to main user service
    face_id UUID, -- Primary face for this user
    privacy_settings JSONB DEFAULT '{}',
    consent_given BOOLEAN DEFAULT FALSE,
    consent_timestamp TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    deleted_at TIMESTAMP WITH TIME ZONE
);

-- Index สำหรับ performance
CREATE INDEX idx_users_user_id ON core.users(user_id);
CREATE INDEX idx_users_face_id ON core.users(face_id);
CREATE INDEX idx_users_created_at ON core.users(created_at);
```

### 2.2 Face Recognition Schema
```sql
-- Schema: face_recognition
CREATE SCHEMA IF NOT EXISTS face_recognition;

-- ตาราง face_data - ข้อมูลใบหน้าที่ลงทะเบียน
CREATE TABLE face_recognition.face_data (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES core.users(user_id),
    face_id VARCHAR(64) NOT NULL UNIQUE, -- External face ID for Milvus
    
    -- Face metadata
    image_hash VARCHAR(64), -- SHA-256 of original image
    face_quality_score DECIMAL(5,4), -- 0.0000-1.0000
    face_landmarks JSONB, -- Facial landmark points
    face_attributes JSONB, -- Additional face attributes
    
    -- Embedding information
    embedding_version VARCHAR(20) NOT NULL, -- Model version used
    embedding_dimension INTEGER NOT NULL, -- 256 or 512
    milvus_collection VARCHAR(50) NOT NULL, -- Collection name in Milvus
    
    -- Registration context
    registration_source VARCHAR(50), -- 'signup', 'mobile_app', 'web', etc.
    registration_device_info JSONB,
    ip_address INET,
    
    -- Status and lifecycle
    status VARCHAR(20) DEFAULT 'active', -- 'active', 'suspended', 'deleted'
    verification_count INTEGER DEFAULT 0,
    last_used_at TIMESTAMP WITH TIME ZONE,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    deleted_at TIMESTAMP WITH TIME ZONE
);

-- ตาราง face_recognition_sessions - เซสชันการตรวจสอบใบหน้า
CREATE TABLE face_recognition.recognition_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id VARCHAR(64) NOT NULL,
    user_id UUID REFERENCES core.users(user_id),
    
    -- Recognition request
    operation_type VARCHAR(20) NOT NULL, -- 'register', 'verify', 'identify'
    input_image_hash VARCHAR(64),
    
    -- Results
    matched_face_id UUID REFERENCES face_recognition.face_data(id),
    confidence_score DECIMAL(5,4),
    similarity_score DECIMAL(5,4),
    recognition_result VARCHAR(20), -- 'success', 'failed', 'rejected'
    
    -- Processing details
    model_version VARCHAR(20),
    processing_time_ms INTEGER,
    preprocessing_steps JSONB,
    
    -- Context
    request_source VARCHAR(50), -- 'login', 'post_tagging', 'video_call', etc.
    device_info JSONB,
    ip_address INET,
    user_agent TEXT,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ตาราง face_groups - การจัดกลุ่มใบหน้า
CREATE TABLE face_recognition.face_groups (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    group_name VARCHAR(100) NOT NULL,
    user_id UUID NOT NULL REFERENCES core.users(user_id),
    description TEXT,
    face_count INTEGER DEFAULT 0,
    privacy_level VARCHAR(20) DEFAULT 'private', -- 'public', 'friends', 'private'
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ตาราง face_group_members - สมาชิกในกลุ่มใบหน้า
CREATE TABLE face_recognition.face_group_members (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    group_id UUID NOT NULL REFERENCES face_recognition.face_groups(id),
    face_id UUID NOT NULL REFERENCES face_recognition.face_data(id),
    added_by UUID REFERENCES core.users(user_id),
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(group_id, face_id)
);

-- Indexes สำหรับ face_recognition
CREATE INDEX idx_face_data_user_id ON face_recognition.face_data(user_id);
CREATE INDEX idx_face_data_face_id ON face_recognition.face_data(face_id);
CREATE INDEX idx_face_data_status ON face_recognition.face_data(status);
CREATE INDEX idx_face_data_created_at ON face_recognition.face_data(created_at);

CREATE INDEX idx_recognition_sessions_user_id ON face_recognition.recognition_sessions(user_id);
CREATE INDEX idx_recognition_sessions_session_id ON face_recognition.recognition_sessions(session_id);
CREATE INDEX idx_recognition_sessions_operation ON face_recognition.recognition_sessions(operation_type);
CREATE INDEX idx_recognition_sessions_created_at ON face_recognition.recognition_sessions(created_at);
```

### 2.3 Face Detection Schema
```sql
-- Schema: face_detection
CREATE SCHEMA IF NOT EXISTS face_detection;

-- ตาราง detection_requests - คำขอตรวจจับใบหน้า
CREATE TABLE face_detection.detection_requests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    request_id VARCHAR(64) NOT NULL UNIQUE,
    user_id UUID REFERENCES core.users(user_id),
    
    -- Input data
    image_hash VARCHAR(64) NOT NULL,
    image_url TEXT,
    image_metadata JSONB, -- resolution, format, etc.
    
    -- Detection parameters
    min_face_size INTEGER DEFAULT 40,
    max_faces INTEGER DEFAULT 100,
    detection_confidence DECIMAL(5,4) DEFAULT 0.7,
    return_landmarks BOOLEAN DEFAULT TRUE,
    return_attributes BOOLEAN DEFAULT FALSE,
    
    -- Results
    faces_detected INTEGER DEFAULT 0,
    processing_time_ms INTEGER,
    model_version VARCHAR(20),
    
    -- Context
    request_source VARCHAR(50), -- 'post_upload', 'profile_photo', 'auto_tag', etc.
    ip_address INET,
    
    -- Status
    status VARCHAR(20) DEFAULT 'processing', -- 'processing', 'completed', 'failed'
    error_message TEXT,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE
);

-- ตาราง detected_faces - ใบหน้าที่ตรวจพบ
CREATE TABLE face_detection.detected_faces (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    detection_request_id UUID NOT NULL REFERENCES face_detection.detection_requests(id),
    face_index INTEGER NOT NULL, -- ลำดับใบหน้าในภาพ (0, 1, 2, ...)
    
    -- Bounding box
    bbox_x INTEGER NOT NULL,
    bbox_y INTEGER NOT NULL,
    bbox_width INTEGER NOT NULL,
    bbox_height INTEGER NOT NULL,
    
    -- Detection confidence
    detection_confidence DECIMAL(5,4) NOT NULL,
    face_quality_score DECIMAL(5,4),
    
    -- Facial landmarks (optional)
    landmarks JSONB, -- 5-point or 68-point landmarks
    
    -- Face attributes (optional)
    estimated_age DECIMAL(5,2),
    estimated_gender VARCHAR(10),
    gender_confidence DECIMAL(5,4),
    emotions JSONB, -- happiness, sadness, anger, etc.
    
    -- Recognition results (if performed)
    recognized_user_id UUID REFERENCES core.users(user_id),
    recognition_confidence DECIMAL(5,4),
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes สำหรับ face_detection
CREATE INDEX idx_detection_requests_user_id ON face_detection.detection_requests(user_id);
CREATE INDEX idx_detection_requests_status ON face_detection.detection_requests(status);
CREATE INDEX idx_detection_requests_created_at ON face_detection.detection_requests(created_at);

CREATE INDEX idx_detected_faces_request_id ON face_detection.detected_faces(detection_request_id);
CREATE INDEX idx_detected_faces_recognized_user ON face_detection.detected_faces(recognized_user_id);
```

### 2.4 Antispoofing Schema
```sql
-- Schema: antispoofing
CREATE SCHEMA IF NOT EXISTS antispoofing;

-- ตาราง antispoofing_checks - การตรวจสอบ antispoofing
CREATE TABLE antispoofing.antispoofing_checks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id VARCHAR(64) NOT NULL,
    user_id UUID REFERENCES core.users(user_id),
    
    -- Check details
    check_type VARCHAR(20) NOT NULL, -- 'passive', 'active_liveness'
    input_type VARCHAR(20) NOT NULL, -- 'image', 'video', 'live_stream'
    input_hash VARCHAR(64),
    
    -- Active liveness details (if applicable)
    liveness_challenges JSONB, -- ["turn_left", "turn_right", "blink", "nod"]
    challenge_responses JSONB, -- Response data for each challenge
    
    -- Results
    is_live BOOLEAN,
    confidence_score DECIMAL(5,4),
    spoof_type VARCHAR(30), -- 'photo', 'video', 'mask', 'screen', null if live
    risk_level VARCHAR(20), -- 'low', 'medium', 'high'
    
    -- Processing details
    model_version VARCHAR(20),
    processing_time_ms INTEGER,
    analysis_details JSONB,
    
    -- Context
    request_source VARCHAR(50), -- 'login', 'payment', 'verification', etc.
    device_info JSONB,
    ip_address INET,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ตาราง liveness_sessions - เซสชัน liveness check แบบ active
CREATE TABLE antispoofing.liveness_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_token VARCHAR(128) NOT NULL UNIQUE,
    user_id UUID REFERENCES core.users(user_id),
    
    -- Session configuration
    challenges_required JSONB NOT NULL, -- List of required challenges
    max_attempts INTEGER DEFAULT 3,
    timeout_seconds INTEGER DEFAULT 30,
    
    -- Session progress
    current_challenge VARCHAR(20),
    challenges_completed JSONB DEFAULT '[]',
    attempts_count INTEGER DEFAULT 0,
    
    -- Results
    session_status VARCHAR(20) DEFAULT 'active', -- 'active', 'completed', 'failed', 'expired'
    overall_result BOOLEAN,
    failure_reason VARCHAR(100),
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE
);

-- Indexes สำหรับ antispoofing
CREATE INDEX idx_antispoofing_checks_user_id ON antispoofing.antispoofing_checks(user_id);
CREATE INDEX idx_antispoofing_checks_session_id ON antispoofing.antispoofing_checks(session_id);
CREATE INDEX idx_antispoofing_checks_is_live ON antispoofing.antispoofing_checks(is_live);
CREATE INDEX idx_antispoofing_checks_created_at ON antispoofing.antispoofing_checks(created_at);

CREATE INDEX idx_liveness_sessions_token ON antispoofing.liveness_sessions(session_token);
CREATE INDEX idx_liveness_sessions_user_id ON antispoofing.liveness_sessions(user_id);
CREATE INDEX idx_liveness_sessions_status ON antispoofing.liveness_sessions(session_status);
```

### 2.5 Deepfake Detection Schema
```sql
-- Schema: deepfake_detection
CREATE SCHEMA IF NOT EXISTS deepfake_detection;

-- ตาราง deepfake_analyses - การวิเคราะห์ deepfake
CREATE TABLE deepfake_detection.deepfake_analyses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    analysis_id VARCHAR(64) NOT NULL UNIQUE,
    user_id UUID REFERENCES core.users(user_id),
    
    -- Input data
    content_type VARCHAR(20) NOT NULL, -- 'image', 'video'
    content_hash VARCHAR(64) NOT NULL,
    content_url TEXT,
    content_metadata JSONB,
    
    -- Analysis parameters
    detection_level VARCHAR(20) DEFAULT 'standard', -- 'basic', 'standard', 'advanced'
    analysis_scope JSONB, -- Which aspects to analyze
    
    -- Results - Image
    is_deepfake BOOLEAN,
    confidence_score DECIMAL(5,4),
    manipulation_score DECIMAL(5,4),
    
    -- Results - Video (frame-by-frame)
    total_frames INTEGER,
    analyzed_frames INTEGER,
    deepfake_frame_count INTEGER,
    consistency_score DECIMAL(5,4),
    
    -- Detailed analysis
    manipulation_types JSONB, -- ["face_swap", "attribute_manipulation", "expression_transfer"]
    inconsistency_areas JSONB, -- Regions with detected inconsistencies
    temporal_inconsistencies JSONB, -- For videos
    
    -- Processing details
    model_version VARCHAR(20),
    processing_time_ms INTEGER,
    processing_method VARCHAR(30), -- 'sync', 'async', 'batch'
    
    -- Job management (for async processing)
    job_id VARCHAR(64),
    job_status VARCHAR(20) DEFAULT 'pending', -- 'pending', 'processing', 'completed', 'failed'
    progress_percentage DECIMAL(5,2) DEFAULT 0,
    estimated_completion TIMESTAMP WITH TIME ZONE,
    
    -- Context
    request_source VARCHAR(50), -- 'content_upload', 'content_moderation', etc.
    priority INTEGER DEFAULT 5, -- 1-10, higher = more urgent
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE
);

-- ตาราง frame_analyses - วิเคราะห์รายเฟรมสำหรับวิดีโอ
CREATE TABLE deepfake_detection.frame_analyses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    analysis_id UUID NOT NULL REFERENCES deepfake_detection.deepfake_analyses(id),
    frame_number INTEGER NOT NULL,
    timestamp_ms INTEGER NOT NULL,
    
    -- Frame analysis results
    is_deepfake BOOLEAN,
    confidence_score DECIMAL(5,4),
    manipulation_types JSONB,
    inconsistent_regions JSONB,
    
    -- Frame metadata
    frame_hash VARCHAR(64),
    face_count INTEGER DEFAULT 0,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(analysis_id, frame_number)
);

-- Indexes สำหรับ deepfake_detection
CREATE INDEX idx_deepfake_analyses_user_id ON deepfake_detection.deepfake_analyses(user_id);
CREATE INDEX idx_deepfake_analyses_analysis_id ON deepfake_detection.deepfake_analyses(analysis_id);
CREATE INDEX idx_deepfake_analyses_job_status ON deepfake_detection.deepfake_analyses(job_status);
CREATE INDEX idx_deepfake_analyses_created_at ON deepfake_detection.deepfake_analyses(created_at);

CREATE INDEX idx_frame_analyses_analysis_id ON deepfake_detection.frame_analyses(analysis_id);
CREATE INDEX idx_frame_analyses_timestamp ON deepfake_detection.frame_analyses(timestamp_ms);
```

### 2.6 Age & Gender Detection Schema
```sql
-- Schema: demographics
CREATE SCHEMA IF NOT EXISTS demographics;

-- ตาราง demographic_analyses - การวิเคราะห์ demographics
CREATE TABLE demographics.demographic_analyses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    request_id VARCHAR(64) NOT NULL UNIQUE,
    user_id UUID REFERENCES core.users(user_id),
    
    -- Input
    image_hash VARCHAR(64) NOT NULL,
    face_detection_id UUID, -- Reference to face_detection if available
    
    -- Processing details
    faces_analyzed INTEGER DEFAULT 0,
    model_version VARCHAR(20),
    processing_time_ms INTEGER,
    
    -- Context
    request_source VARCHAR(50), -- 'content_analysis', 'user_profiling', etc.
    batch_id VARCHAR(64), -- For batch processing
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ตาราง face_demographics - ผลการวิเคราะห์รายใบหน้า
CREATE TABLE demographics.face_demographics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    analysis_id UUID NOT NULL REFERENCES demographics.demographic_analyses(id),
    face_index INTEGER NOT NULL,
    
    -- Face location (if not from face_detection)
    bbox_x INTEGER,
    bbox_y INTEGER,
    bbox_width INTEGER,
    bbox_height INTEGER,
    
    -- Age estimation
    estimated_age DECIMAL(5,2) NOT NULL,
    age_confidence DECIMAL(5,4),
    age_range_min INTEGER,
    age_range_max INTEGER,
    age_group VARCHAR(20), -- 'child', 'teenager', 'young_adult', 'adult', 'senior'
    
    -- Gender estimation
    estimated_gender VARCHAR(10) NOT NULL, -- 'male', 'female'
    gender_confidence DECIMAL(5,4),
    
    -- Additional attributes (optional)
    emotion_primary VARCHAR(20), -- 'happy', 'sad', 'angry', 'neutral', etc.
    emotion_confidence DECIMAL(5,4),
    face_quality_score DECIMAL(5,4),
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(analysis_id, face_index)
);

-- Indexes สำหรับ demographics
CREATE INDEX idx_demographic_analyses_user_id ON demographics.demographic_analyses(user_id);
CREATE INDEX idx_demographic_analyses_request_id ON demographics.demographic_analyses(request_id);
CREATE INDEX idx_demographic_analyses_created_at ON demographics.demographic_analyses(created_at);

CREATE INDEX idx_face_demographics_analysis_id ON demographics.face_demographics(analysis_id);
CREATE INDEX idx_face_demographics_age_group ON demographics.face_demographics(age_group);
CREATE INDEX idx_face_demographics_gender ON demographics.face_demographics(estimated_gender);
```

## 3. Audit and Analytics Schema

### 3.1 Audit Trail
```sql
-- Schema: audit
CREATE SCHEMA IF NOT EXISTS audit;

-- ตาราง ai_operations_log - บันทึกการใช้งาน AI
CREATE TABLE audit.ai_operations_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    operation_id VARCHAR(64) NOT NULL,
    user_id UUID REFERENCES core.users(user_id),
    
    -- Operation details
    service_name VARCHAR(50) NOT NULL, -- 'face_recognition', 'deepfake_detection', etc.
    operation_type VARCHAR(50) NOT NULL, -- 'register', 'verify', 'detect', 'analyze'
    endpoint VARCHAR(100),
    
    -- Request/Response details
    request_size_bytes INTEGER,
    response_size_bytes INTEGER,
    processing_time_ms INTEGER,
    
    -- Results summary
    operation_result VARCHAR(20), -- 'success', 'error', 'partial'
    confidence_score DECIMAL(5,4),
    error_code VARCHAR(50),
    error_message TEXT,
    
    -- Context
    ip_address INET,
    user_agent TEXT,
    api_key_id VARCHAR(50),
    request_source VARCHAR(50),
    
    -- Billing/Usage
    cost_units INTEGER DEFAULT 1, -- For billing calculations
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ตาราง performance_metrics - เมทริกส์ performance
CREATE TABLE audit.performance_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    metric_date DATE NOT NULL,
    service_name VARCHAR(50) NOT NULL,
    
    -- Volume metrics
    total_requests INTEGER DEFAULT 0,
    successful_requests INTEGER DEFAULT 0,
    failed_requests INTEGER DEFAULT 0,
    
    -- Performance metrics
    avg_response_time_ms DECIMAL(10,2),
    p95_response_time_ms DECIMAL(10,2),
    p99_response_time_ms DECIMAL(10,2),
    
    -- Accuracy metrics (if applicable)
    avg_confidence_score DECIMAL(5,4),
    high_confidence_count INTEGER DEFAULT 0, -- > 0.9
    medium_confidence_count INTEGER DEFAULT 0, -- 0.7-0.9
    low_confidence_count INTEGER DEFAULT 0, -- < 0.7
    
    -- Error analysis
    error_rate DECIMAL(5,4),
    timeout_count INTEGER DEFAULT 0,
    model_error_count INTEGER DEFAULT 0,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(metric_date, service_name)
);

-- Indexes สำหรับ audit
CREATE INDEX idx_ai_operations_log_user_id ON audit.ai_operations_log(user_id);
CREATE INDEX idx_ai_operations_log_service ON audit.ai_operations_log(service_name);
CREATE INDEX idx_ai_operations_log_operation ON audit.ai_operations_log(operation_type);
CREATE INDEX idx_ai_operations_log_created_at ON audit.ai_operations_log(created_at);

CREATE INDEX idx_performance_metrics_date ON audit.performance_metrics(metric_date);
CREATE INDEX idx_performance_metrics_service ON audit.performance_metrics(service_name);
```

## 4. Configuration and Settings Schema

### 4.1 System Configuration
```sql
-- Schema: config
CREATE SCHEMA IF NOT EXISTS config;

-- ตาราง ai_models - การจัดการโมเดล AI
CREATE TABLE config.ai_models (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    service_name VARCHAR(50) NOT NULL,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(20) NOT NULL,
    
    -- Model details
    model_type VARCHAR(50), -- 'face_detection', 'face_recognition', etc.
    model_path TEXT NOT NULL,
    model_size_mb DECIMAL(10,2),
    input_dimensions JSONB,
    output_dimensions JSONB,
    
    -- Performance characteristics
    avg_inference_time_ms DECIMAL(10,2),
    accuracy_score DECIMAL(5,4),
    supported_formats JSONB,
    
    -- Status
    status VARCHAR(20) DEFAULT 'active', -- 'active', 'deprecated', 'testing'
    is_default BOOLEAN DEFAULT FALSE,
    
    -- Metadata
    description TEXT,
    training_date DATE,
    deployment_date DATE,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(service_name, model_name, model_version)
);

-- ตาราง service_settings - การตั้งค่าแต่ละ service
CREATE TABLE config.service_settings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    service_name VARCHAR(50) NOT NULL,
    setting_key VARCHAR(100) NOT NULL,
    setting_value JSONB NOT NULL,
    setting_type VARCHAR(20) NOT NULL, -- 'string', 'number', 'boolean', 'object'
    
    -- Metadata
    description TEXT,
    is_user_configurable BOOLEAN DEFAULT FALSE,
    requires_restart BOOLEAN DEFAULT FALSE,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(service_name, setting_key)
);

-- Indexes สำหรับ config
CREATE INDEX idx_ai_models_service ON config.ai_models(service_name);
CREATE INDEX idx_ai_models_status ON config.ai_models(status);
CREATE INDEX idx_service_settings_service ON config.service_settings(service_name);
```

## 5. Milvus Collections Schema

### 5.1 Face Embeddings Collections
```python
# Milvus collection schemas
face_embeddings_schema = {
    "collection_name": "face_embeddings_512d",
    "description": "Face embeddings from ArcFace model",
    "fields": [
        {
            "name": "id",
            "type": "INT64",
            "is_primary": True,
            "auto_id": True
        },
        {
            "name": "face_id", 
            "type": "VARCHAR",
            "max_length": 64,
            "is_primary": False
        },
        {
            "name": "user_id",
            "type": "VARCHAR", 
            "max_length": 36,
            "is_primary": False
        },
        {
            "name": "embedding",
            "type": "FLOAT_VECTOR",
            "dimension": 512
        },
        {
            "name": "created_at",
            "type": "INT64"  # Unix timestamp
        }
    ],
    "index_params": {
        "metric_type": "COSINE",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 1024}
    }
}

deepfake_features_schema = {
    "collection_name": "deepfake_features",
    "description": "Features for deepfake detection",
    "fields": [
        {
            "name": "id",
            "type": "INT64", 
            "is_primary": True,
            "auto_id": True
        },
        {
            "name": "content_hash",
            "type": "VARCHAR",
            "max_length": 64
        },
        {
            "name": "features",
            "type": "FLOAT_VECTOR",
            "dimension": 1024
        },
        {
            "name": "is_deepfake",
            "type": "BOOL"
        }
    ]
}
```

## 6. Redis Key Patterns

### 6.1 Caching Patterns
```yaml
# Face recognition cache
"face_cache:{user_id}":
  type: hash
  ttl: 3600  # 1 hour
  fields:
    - face_id
    - embedding_hash
    - last_updated

# Session data
"session:{session_id}":
  type: hash  
  ttl: 1800  # 30 minutes
  fields:
    - user_id
    - operation_type
    - created_at
    - attempts

# Rate limiting
"rate_limit:{service}:{user_id}":
  type: counter
  ttl: 60    # 1 minute
  
# Processing queues
"queue:deepfake_video":
  type: list
  description: "Async video processing queue"
  
"queue:batch_demographics":
  type: list  
  description: "Batch demographic analysis"
```

## 7. Database Maintenance

### 7.1 Partitioning Strategy
```sql
-- Partition large tables by date
CREATE TABLE audit.ai_operations_log_y2024m01 
PARTITION OF audit.ai_operations_log 
FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

-- Auto-partition creation
CREATE OR REPLACE FUNCTION create_monthly_partitions()
RETURNS void AS $$
DECLARE
    start_date date;
    end_date date;
    table_name text;
BEGIN
    start_date := date_trunc('month', CURRENT_DATE);
    end_date := start_date + interval '1 month';
    table_name := 'ai_operations_log_y' || to_char(start_date, 'YYYY') || 'm' || to_char(start_date, 'MM');
    
    EXECUTE format('CREATE TABLE IF NOT EXISTS audit.%I PARTITION OF audit.ai_operations_log FOR VALUES FROM (%L) TO (%L)',
                   table_name, start_date, end_date);
END;
$$ LANGUAGE plpgsql;
```

### 7.2 Data Retention Policies
```sql
-- Clean up old data automatically
CREATE OR REPLACE FUNCTION cleanup_old_data()
RETURNS void AS $$
BEGIN
    -- Remove logs older than 1 year
    DELETE FROM audit.ai_operations_log 
    WHERE created_at < NOW() - INTERVAL '1 year';
    
    -- Remove completed deepfake analyses older than 6 months
    DELETE FROM deepfake_detection.deepfake_analyses 
    WHERE job_status = 'completed' 
    AND completed_at < NOW() - INTERVAL '6 months';
    
    -- Remove expired liveness sessions
    DELETE FROM antispoofing.liveness_sessions 
    WHERE expires_at < NOW();
END;
$$ LANGUAGE plpgsql;

-- Schedule cleanup job
SELECT cron.schedule('cleanup_old_data', '0 2 * * 0', 'SELECT cleanup_old_data()');
```
