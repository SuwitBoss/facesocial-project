-- FaceSocial AI Database Initialization Script
-- This script creates the core database schema for the FaceSocial AI services
-- Updated: December 2024

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Enable vector extension for face embeddings (requires pgvector)
-- Note: Install pgvector: apt-get install postgresql-15-pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Enable trigram extension for text search
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- ==============================================
-- CORE TABLES
-- ==============================================

-- User Management
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT true,
    is_verified BOOLEAN DEFAULT false
);

-- Face Records
CREATE TABLE IF NOT EXISTS faces (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    face_hash VARCHAR(64) UNIQUE NOT NULL, -- SHA-256 hash for deduplication
    face_embedding vector(512), -- Face recognition embedding (512-dim)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_primary BOOLEAN DEFAULT false,
    quality_score FLOAT DEFAULT 0.0,
    confidence_score FLOAT DEFAULT 0.0
);

-- Image Analysis Sessions
CREATE TABLE IF NOT EXISTS analysis_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_key VARCHAR(64) UNIQUE NOT NULL,
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    image_path VARCHAR(500),
    image_hash VARCHAR(64),
    image_size INTEGER,
    mime_type VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    status VARCHAR(20) DEFAULT 'pending', -- pending, processing, completed, failed
    total_faces_detected INTEGER DEFAULT 0,
    processing_time_ms INTEGER DEFAULT 0
);

-- Face Detection Results
CREATE TABLE IF NOT EXISTS face_detections (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID REFERENCES analysis_sessions(id) ON DELETE CASCADE,
    face_id UUID REFERENCES faces(id) ON DELETE SET NULL,
    detection_method VARCHAR(20) NOT NULL, -- mediapipe, yolo, mtcnn
    bounding_box JSONB NOT NULL, -- {x, y, width, height}
    landmarks JSONB, -- facial landmarks coordinates
    confidence_score FLOAT NOT NULL,
    quality_metrics JSONB, -- blur, lighting, angle, etc.
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Face Recognition Results
CREATE TABLE IF NOT EXISTS face_recognitions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    detection_id UUID REFERENCES face_detections(id) ON DELETE CASCADE,
    matched_face_id UUID REFERENCES faces(id) ON DELETE SET NULL,
    similarity_score FLOAT NOT NULL,
    recognition_method VARCHAR(20) DEFAULT 'ensemble', -- adaface, facenet, arcface, ensemble
    embedding vector(512),
    is_verified_match BOOLEAN DEFAULT false,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Anti-Spoofing Analysis
CREATE TABLE IF NOT EXISTS antispoof_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    detection_id UUID REFERENCES face_detections(id) ON DELETE CASCADE,
    is_real BOOLEAN NOT NULL,
    confidence_score FLOAT NOT NULL,
    spoof_type VARCHAR(20), -- photo, video, mask, 3d_model
    analysis_details JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Gender and Age Analysis
CREATE TABLE IF NOT EXISTS gender_age_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    detection_id UUID REFERENCES face_detections(id) ON DELETE CASCADE,
    predicted_gender VARCHAR(10) NOT NULL, -- male, female
    gender_confidence FLOAT NOT NULL,
    predicted_age INTEGER NOT NULL,
    age_range_min INTEGER,
    age_range_max INTEGER,
    age_confidence FLOAT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Deepfake Detection
CREATE TABLE IF NOT EXISTS deepfake_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    detection_id UUID REFERENCES face_detections(id) ON DELETE CASCADE,
    is_deepfake BOOLEAN NOT NULL,
    confidence_score FLOAT NOT NULL,
    manipulation_type VARCHAR(30), -- face_swap, face_reenactment, speech_driven, full_synthetic
    detection_method VARCHAR(20) DEFAULT 'vit_transformer',
    analysis_details JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ==============================================
-- PERFORMANCE & MONITORING TABLES
-- ==============================================

-- Service Performance Metrics
CREATE TABLE IF NOT EXISTS service_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    service_name VARCHAR(50) NOT NULL,
    endpoint VARCHAR(100) NOT NULL,
    request_count INTEGER DEFAULT 1,
    avg_response_time_ms FLOAT NOT NULL,
    min_response_time_ms INTEGER NOT NULL,
    max_response_time_ms INTEGER NOT NULL,
    error_count INTEGER DEFAULT 0,
    memory_usage_mb FLOAT,
    gpu_memory_usage_mb FLOAT,
    cpu_usage_percent FLOAT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- System Health Logs
CREATE TABLE IF NOT EXISTS health_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    service_name VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL, -- healthy, unhealthy, degraded
    response_time_ms INTEGER,
    error_message TEXT,
    gpu_memory_total_mb FLOAT,
    gpu_memory_used_mb FLOAT,
    gpu_utilization_percent FLOAT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- API Usage Analytics
CREATE TABLE IF NOT EXISTS api_usage (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    endpoint VARCHAR(100) NOT NULL,
    method VARCHAR(10) NOT NULL,
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    ip_address INET,
    user_agent TEXT,
    request_size_bytes INTEGER,
    response_size_bytes INTEGER,
    response_status INTEGER NOT NULL,
    processing_time_ms INTEGER NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);-- ==============================================
-- INDEXES FOR PERFORMANCE
-- ==============================================

-- Face embedding similarity search (requires pgvector)
-- Note: This index will be created only if pgvector extension is available
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector') THEN
        CREATE INDEX IF NOT EXISTS idx_faces_embedding ON faces USING ivfflat (face_embedding vector_cosine_ops) WITH (lists = 100);
    END IF;
END $$;

-- User lookups
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_active ON users(is_active) WHERE is_active = true;

-- Session and detection lookups
CREATE INDEX IF NOT EXISTS idx_analysis_sessions_user_id ON analysis_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_analysis_sessions_status ON analysis_sessions(status);
CREATE INDEX IF NOT EXISTS idx_analysis_sessions_created_at ON analysis_sessions(created_at);

CREATE INDEX IF NOT EXISTS idx_face_detections_session_id ON face_detections(session_id);
CREATE INDEX IF NOT EXISTS idx_face_detections_method ON face_detections(detection_method);
CREATE INDEX IF NOT EXISTS idx_face_detections_confidence ON face_detections(confidence_score);

-- Recognition and analysis lookups
CREATE INDEX IF NOT EXISTS idx_face_recognitions_detection_id ON face_recognitions(detection_id);
CREATE INDEX IF NOT EXISTS idx_face_recognitions_matched_face_id ON face_recognitions(matched_face_id);
CREATE INDEX IF NOT EXISTS idx_face_recognitions_similarity ON face_recognitions(similarity_score);

CREATE INDEX IF NOT EXISTS idx_antispoof_detection_id ON antispoof_results(detection_id);
CREATE INDEX IF NOT EXISTS idx_gender_age_detection_id ON gender_age_results(detection_id);
CREATE INDEX IF NOT EXISTS idx_deepfake_detection_id ON deepfake_results(detection_id);

-- Performance monitoring indexes
CREATE INDEX IF NOT EXISTS idx_service_metrics_service_timestamp ON service_metrics(service_name, timestamp);
CREATE INDEX IF NOT EXISTS idx_health_logs_service_timestamp ON health_logs(service_name, timestamp);
CREATE INDEX IF NOT EXISTS idx_api_usage_endpoint_timestamp ON api_usage(endpoint, timestamp);

-- ==============================================
-- VIEWS FOR ANALYTICS
-- ==============================================

-- Comprehensive face analysis view
CREATE OR REPLACE VIEW face_analysis_summary AS
SELECT 
    s.id as session_id,
    s.session_key,
    s.user_id,
    s.created_at,
    s.status,
    s.total_faces_detected,
    fd.id as detection_id,
    fd.detection_method,
    fd.confidence_score as detection_confidence,
    fd.bounding_box,
    fr.similarity_score as recognition_score,
    fr.matched_face_id,
    ar.is_real as is_real_face,
    ar.confidence_score as antispoof_confidence,
    gar.predicted_gender,
    gar.predicted_age,
    dr.is_deepfake,
    dr.confidence_score as deepfake_confidence
FROM analysis_sessions s
LEFT JOIN face_detections fd ON s.id = fd.session_id
LEFT JOIN face_recognitions fr ON fd.id = fr.detection_id
LEFT JOIN antispoof_results ar ON fd.id = ar.detection_id
LEFT JOIN gender_age_results gar ON fd.id = gar.detection_id
LEFT JOIN deepfake_results dr ON fd.id = dr.detection_id;

-- Service performance summary
CREATE OR REPLACE VIEW service_performance_summary AS
SELECT 
    service_name,
    DATE_TRUNC('hour', timestamp) as hour,
    COUNT(*) as total_requests,
    AVG(avg_response_time_ms) as avg_response_time,
    MIN(min_response_time_ms) as min_response_time,
    MAX(max_response_time_ms) as max_response_time,
    SUM(error_count) as total_errors,
    AVG(memory_usage_mb) as avg_memory_usage,
    AVG(gpu_memory_usage_mb) as avg_gpu_memory_usage
FROM service_metrics 
GROUP BY service_name, DATE_TRUNC('hour', timestamp)
ORDER BY hour DESC;

-- ==============================================
-- FUNCTIONS AND TRIGGERS
-- ==============================================

-- Update timestamp function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for updated_at
DROP TRIGGER IF EXISTS update_users_updated_at ON users;
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

DROP TRIGGER IF EXISTS update_faces_updated_at ON faces;
CREATE TRIGGER update_faces_updated_at BEFORE UPDATE ON faces
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Face similarity search function (only if pgvector is available)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector') THEN
        EXECUTE 'CREATE OR REPLACE FUNCTION find_similar_faces(
            query_embedding vector(512),
            similarity_threshold float DEFAULT 0.8,
            max_results int DEFAULT 10
        )
        RETURNS TABLE(
            face_id UUID,
            user_id UUID,
            similarity_score float
        ) AS $func$
        BEGIN
            RETURN QUERY
            SELECT 
                f.id as face_id,
                f.user_id,
                1 - (f.face_embedding <=> query_embedding) as similarity_score
            FROM faces f
            WHERE f.face_embedding IS NOT NULL
            AND 1 - (f.face_embedding <=> query_embedding) >= similarity_threshold
            ORDER BY f.face_embedding <=> query_embedding
            LIMIT max_results;
        END;
        $func$ LANGUAGE plpgsql;';
    END IF;
END $$;

-- ==============================================
-- SYSTEM CONFIGURATION
-- ==============================================

-- Insert system configuration
CREATE TABLE IF NOT EXISTS system_config (
    key VARCHAR(50) PRIMARY KEY,
    value TEXT NOT NULL,
    description TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO system_config (key, value, description) VALUES
('max_faces_per_image', '10', 'Maximum number of faces to detect per image'),
('min_face_size', '20', 'Minimum face size in pixels'),
('recognition_threshold', '0.8', 'Face recognition similarity threshold'),
('antispoof_threshold', '0.5', 'Anti-spoofing confidence threshold'),
('deepfake_threshold', '0.5', 'Deepfake detection confidence threshold'),
('gpu_memory_limit_mb', '6144', 'Total GPU memory limit in MB'),
('batch_processing_enabled', 'true', 'Enable batch processing for multiple images')
ON CONFLICT (key) DO NOTHING;

-- ==============================================
-- SAMPLE DATA (Development Only)
-- ==============================================

-- Insert sample admin user
INSERT INTO users (username, email, password_hash, first_name, last_name, is_verified)
VALUES (
    'admin',
    'admin@facesocial.ai',
    '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LeVMstdMWT0FRuRKy', -- password: admin123
    'System',
    'Administrator',
    true
) ON CONFLICT (username) DO NOTHING;

-- Create admin schema complete message
INSERT INTO health_logs (service_name, status, error_message)
VALUES ('database', 'healthy', 'Database schema initialized successfully');

COMMIT;
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Batch processing jobs
CREATE TABLE IF NOT EXISTS batch_jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_type VARCHAR(100) NOT NULL, -- 'face_detection', 'face_recognition', 'deepfake_analysis'
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    status VARCHAR(50) DEFAULT 'pending', -- 'pending', 'processing', 'completed', 'failed'
    total_items INTEGER DEFAULT 0,
    processed_items INTEGER DEFAULT 0,
    failed_items INTEGER DEFAULT 0,
    input_data JSONB NOT NULL DEFAULT '{}',
    results JSONB DEFAULT '{}',
    error_message TEXT,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- API usage tracking
CREATE TABLE IF NOT EXISTS api_usage_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    service_name VARCHAR(100) NOT NULL,
    endpoint VARCHAR(200) NOT NULL,
    method VARCHAR(10) NOT NULL,
    status_code INTEGER,
    processing_time_ms INTEGER,
    input_size_bytes BIGINT,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_face_embeddings_user_id ON face_embeddings(user_id);
CREATE INDEX IF NOT EXISTS idx_face_embeddings_model_name ON face_embeddings(model_name);
CREATE INDEX IF NOT EXISTS idx_face_embeddings_is_primary ON face_embeddings(is_primary);

CREATE INDEX IF NOT EXISTS idx_face_detections_image_hash ON face_detections(image_hash);
CREATE INDEX IF NOT EXISTS idx_face_detections_model ON face_detections(detection_model);
CREATE INDEX IF NOT EXISTS idx_face_detections_created_at ON face_detections(created_at);

CREATE INDEX IF NOT EXISTS idx_face_recognition_sessions_user_id ON face_recognition_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_face_recognition_sessions_type ON face_recognition_sessions(session_type);
CREATE INDEX IF NOT EXISTS idx_face_recognition_sessions_successful ON face_recognition_sessions(is_successful);

CREATE INDEX IF NOT EXISTS idx_antispoofing_session_id ON antispoofing_results(session_id);
CREATE INDEX IF NOT EXISTS idx_antispoofing_is_real ON antispoofing_results(is_real);

CREATE INDEX IF NOT EXISTS idx_deepfake_content_id ON deepfake_detections(content_id);
CREATE INDEX IF NOT EXISTS idx_deepfake_is_deepfake ON deepfake_detections(is_deepfake);
CREATE INDEX IF NOT EXISTS idx_deepfake_file_type ON deepfake_detections(file_type);

CREATE INDEX IF NOT EXISTS idx_demographics_detection_id ON demographics_analysis(detection_id);

CREATE INDEX IF NOT EXISTS idx_batch_jobs_user_id ON batch_jobs(user_id);
CREATE INDEX IF NOT EXISTS idx_batch_jobs_status ON batch_jobs(status);
CREATE INDEX IF NOT EXISTS idx_batch_jobs_type ON batch_jobs(job_type);

CREATE INDEX IF NOT EXISTS idx_api_usage_user_id ON api_usage_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_api_usage_service ON api_usage_logs(service_name);
CREATE INDEX IF NOT EXISTS idx_api_usage_created_at ON api_usage_logs(created_at);

-- Create triggers for updated_at timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_face_embeddings_updated_at BEFORE UPDATE ON face_embeddings
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Insert sample data for testing
INSERT INTO users (id, email, username, full_name) VALUES
    ('550e8400-e29b-41d4-a716-446655440000', 'admin@facesocial.com', 'admin', 'System Administrator'),
    ('550e8400-e29b-41d4-a716-446655440001', 'testuser@facesocial.com', 'testuser', 'Test User'),
    ('550e8400-e29b-41d4-a716-446655440002', 'demo@facesocial.com', 'demo', 'Demo User')
ON CONFLICT (email) DO NOTHING;

-- Create views for analytics
CREATE OR REPLACE VIEW face_recognition_stats AS
SELECT 
    DATE(created_at) as date,
    session_type,
    COUNT(*) as total_sessions,
    COUNT(*) FILTER (WHERE is_successful = true) as successful_sessions,
    AVG(similarity_score) as avg_similarity_score,
    AVG(processing_time_ms) as avg_processing_time
FROM face_recognition_sessions
GROUP BY DATE(created_at), session_type
ORDER BY date DESC;

CREATE OR REPLACE VIEW daily_api_usage AS
SELECT 
    DATE(created_at) as date,
    service_name,
    COUNT(*) as request_count,
    AVG(processing_time_ms) as avg_response_time,
    COUNT(*) FILTER (WHERE status_code >= 400) as error_count
FROM api_usage_logs
GROUP BY DATE(created_at), service_name
ORDER BY date DESC;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA ai_services TO facesocial;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA ai_services TO facesocial;
GRANT USAGE ON SCHEMA ai_services TO facesocial;
