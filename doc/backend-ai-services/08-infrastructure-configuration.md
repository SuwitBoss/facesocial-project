# API Gateway และ Infrastructure Configuration

## 1. API Gateway Architecture

### 1.1 NGINX API Gateway Configuration

```nginx
# /etc/nginx/nginx.conf
user nginx;
worker_processes auto;
error_log /var/log/nginx/error.log notice;
pid /var/run/nginx.pid;

events {
    worker_connections 4096;
    use epoll;
    multi_accept on;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;
    
    # Logging
    log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                   '$status $body_bytes_sent "$http_referer" '
                   '"$http_user_agent" "$http_x_forwarded_for" '
                   'rt=$request_time uct="$upstream_connect_time" '
                   'uht="$upstream_header_time" urt="$upstream_response_time"';
                   
    access_log /var/log/nginx/access.log main;
    
    # Performance
    sendfile on;
    tcp_nopush on;
    tcp_nodelay on;
    keepalive_timeout 65;
    types_hash_max_size 2048;
    client_max_body_size 100M;
    
    # Gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types text/plain text/css application/json application/javascript 
               text/xml application/xml application/xml+rss text/javascript;
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=upload_limit:10m rate=2r/s;
    
    # Upstream definitions
    upstream face_recognition_service {
        least_conn;
        server face-recognition-service-1:8000 max_fails=3 fail_timeout=30s;
        server face-recognition-service-2:8000 max_fails=3 fail_timeout=30s;
        server face-recognition-service-3:8000 max_fails=3 fail_timeout=30s;
        keepalive 32;
    }
    
    upstream antispoofing_service {
        least_conn;
        server antispoofing-service-1:8000 max_fails=3 fail_timeout=30s;
        server antispoofing-service-2:8000 max_fails=3 fail_timeout=30s;
        keepalive 32;
    }
    
    upstream deepfake_detection_service {
        least_conn;
        server deepfake-detection-service-1:8000 max_fails=3 fail_timeout=30s;
        server deepfake-detection-service-2:8000 max_fails=3 fail_timeout=30s;
        keepalive 32;
    }
    
    upstream face_detection_service {
        least_conn;
        server face-detection-service-1:8000 max_fails=3 fail_timeout=30s;
        server face-detection-service-2:8000 max_fails=3 fail_timeout=30s;
        server face-detection-service-3:8000 max_fails=3 fail_timeout=30s;
        keepalive 32;
    }
    
    upstream age_gender_service {
        least_conn;
        server age-gender-service-1:8000 max_fails=3 fail_timeout=30s;
        server age-gender-service-2:8000 max_fails=3 fail_timeout=30s;
        keepalive 32;
    }
    
    # Main server block
    server {
        listen 80;
        listen 443 ssl http2;
        server_name api.facesocial.com;
        
        # SSL Configuration
        ssl_certificate /etc/ssl/certs/facesocial.crt;
        ssl_certificate_key /etc/ssl/private/facesocial.key;
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
        ssl_prefer_server_ciphers off;
        ssl_session_cache shared:SSL:10m;
        ssl_session_timeout 10m;
        
        # Redirect HTTP to HTTPS
        if ($scheme != "https") {
            return 301 https://$server_name$request_uri;
        }
        
        # CORS headers
        add_header 'Access-Control-Allow-Origin' '*' always;
        add_header 'Access-Control-Allow-Methods' 'GET, POST, PUT, DELETE, OPTIONS' always;
        add_header 'Access-Control-Allow-Headers' 'DNT,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Range,Authorization' always;
        
        # Handle preflight requests
        if ($request_method = 'OPTIONS') {
            add_header 'Access-Control-Max-Age' 1728000;
            add_header 'Content-Type' 'text/plain; charset=utf-8';
            add_header 'Content-Length' 0;
            return 204;
        }
        
        # Health check endpoint
        location /health {
            access_log off;
            return 200 "healthy\n";
            add_header Content-Type text/plain;
        }
        
        # Face Recognition Service
        location /api/v1/face-recognition/ {
            limit_req zone=api_limit burst=20 nodelay;
            
            proxy_pass http://face_recognition_service/api/v1/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            proxy_connect_timeout 5s;
            proxy_send_timeout 60s;
            proxy_read_timeout 60s;
            
            proxy_buffering off;
            proxy_request_buffering off;
        }
        
        # Face Antispoofing Service
        location /api/v1/antispoofing/ {
            limit_req zone=api_limit burst=15 nodelay;
            
            proxy_pass http://antispoofing_service/api/v1/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            proxy_connect_timeout 5s;
            proxy_send_timeout 90s;
            proxy_read_timeout 90s;
        }
        
        # Deepfake Detection Service
        location /api/v1/deepfake-detection/ {
            limit_req zone=upload_limit burst=5 nodelay;
            
            proxy_pass http://deepfake_detection_service/api/v1/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            proxy_connect_timeout 10s;
            proxy_send_timeout 300s;
            proxy_read_timeout 300s;
            
            # Large file uploads for video analysis
            client_max_body_size 500M;
            proxy_request_buffering off;
        }
        
        # Face Detection Service
        location /api/v1/face-detection/ {
            limit_req zone=api_limit burst=30 nodelay;
            
            proxy_pass http://face_detection_service/api/v1/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            proxy_connect_timeout 5s;
            proxy_send_timeout 30s;
            proxy_read_timeout 30s;
        }
        
        # Age & Gender Detection Service
        location /api/v1/demographics/ {
            limit_req zone=api_limit burst=25 nodelay;
            
            proxy_pass http://age_gender_service/api/v1/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            proxy_connect_timeout 5s;
            proxy_send_timeout 20s;
            proxy_read_timeout 20s;
        }
        
        # Batch processing endpoints (higher limits)
        location ~* /api/v1/.*/batch {
            limit_req zone=upload_limit burst=3 nodelay;
            client_max_body_size 200M;
            
            proxy_pass http://$1_service/api/v1/$2;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            proxy_connect_timeout 10s;
            proxy_send_timeout 600s;
            proxy_read_timeout 600s;
            proxy_request_buffering off;
        }
        
        # Metrics and monitoring (restrict access)
        location /metrics {
            allow 10.0.0.0/8;
            allow 172.16.0.0/12;
            allow 192.168.0.0/16;
            deny all;
            
            proxy_pass http://prometheus-exporter:9090/metrics;
        }
        
        # Default error pages
        error_page 404 /404.html;
        error_page 500 502 503 504 /50x.html;
        location = /50x.html {
            root /usr/share/nginx/html;
        }
    }
}
```

### 1.2 JWT Authentication Middleware

```lua
-- /etc/nginx/lua/auth.lua
local jwt = require "resty.jwt"
local cjson = require "cjson"

-- JWT secret key (should be loaded from environment)
local jwt_secret = os.getenv("JWT_SECRET") or "your-secret-key"

function verify_jwt_token()
    local auth_header = ngx.var.http_authorization
    
    if not auth_header then
        ngx.status = 401
        ngx.say(cjson.encode({
            success = false,
            error = {
                code = "MISSING_TOKEN",
                message = "Authorization header required"
            }
        }))
        ngx.exit(401)
    end
    
    local token = string.match(auth_header, "Bearer%s+(.+)")
    if not token then
        ngx.status = 401
        ngx.say(cjson.encode({
            success = false,
            error = {
                code = "INVALID_TOKEN_FORMAT",
                message = "Invalid authorization format"
            }
        }))
        ngx.exit(401)
    end
    
    -- Verify JWT token
    local jwt_obj = jwt:verify(jwt_secret, token)
    if not jwt_obj.valid then
        ngx.status = 401
        ngx.say(cjson.encode({
            success = false,
            error = {
                code = "INVALID_TOKEN",
                message = "Invalid or expired token"
            }
        }))
        ngx.exit(401)
    end
    
    -- Set user context
    ngx.var.user_id = jwt_obj.payload.user_id
    ngx.var.user_role = jwt_obj.payload.role
end

-- Call verification
verify_jwt_token()
```

---

## 2. Docker Compose Configuration

### 2.1 Main Docker Compose
```yaml
# docker-compose.yml
version: '3.8'

services:
  # API Gateway
  api-gateway:
    image: nginx:alpine
    container_name: facesocial-gateway
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./nginx/lua:/etc/nginx/lua
      - ./ssl:/etc/ssl
    depends_on:
      - face-recognition-service
      - antispoofing-service
      - deepfake-detection-service
      - face-detection-service
      - age-gender-service
    networks:
      - facesocial-network
    restart: unless-stopped
    
  # Face Recognition Service
  face-recognition-service:
    build: 
      context: ./services/face-recognition
      dockerfile: docker/Dockerfile
    container_name: face-recognition-service
    ports:
      - "8001:8000"
    environment:
      - DATABASE_URL=postgresql://user:password@postgres:5432/facesocial
      - MILVUS_HOST=milvus
      - MILVUS_PORT=19530
      - REDIS_URL=redis://redis:6379
      - MODEL_PATH=/app/models
    volumes:
      - ./models/face-recognition:/app/models
      - face-recognition-logs:/app/logs
    depends_on:
      - postgres
      - milvus
      - redis
    networks:
      - facesocial-network
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 2G
    
  # Antispoofing Service
  antispoofing-service:
    build:
      context: ./services/antispoofing
      dockerfile: docker/Dockerfile
    container_name: antispoofing-service
    ports:
      - "8002:8000"
    environment:
      - DATABASE_URL=postgresql://user:password@postgres:5432/facesocial
      - REDIS_URL=redis://redis:6379
      - MODEL_PATH=/app/models
    volumes:
      - ./models/antispoofing:/app/models
      - antispoofing-logs:/app/logs
    depends_on:
      - postgres
      - redis
    networks:
      - facesocial-network
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 3G
        reservations:
          memory: 1.5G
    
  # Deepfake Detection Service
  deepfake-detection-service:
    build:
      context: ./services/deepfake-detection
      dockerfile: docker/Dockerfile
    container_name: deepfake-detection-service
    ports:
      - "8003:8000"
    environment:
      - DATABASE_URL=postgresql://user:password@postgres:5432/facesocial
      - REDIS_URL=redis://redis:6379
      - CELERY_BROKER_URL=redis://redis:6379
      - MODEL_PATH=/app/models
    volumes:
      - ./models/deepfake-detection:/app/models
      - deepfake-logs:/app/logs
      - deepfake-temp:/app/temp
    depends_on:
      - postgres
      - redis
      - celery-worker
    networks:
      - facesocial-network
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 6G
        reservations:
          memory: 3G
    
  # Face Detection Service
  face-detection-service:
    build:
      context: ./services/face-detection
      dockerfile: docker/Dockerfile
    container_name: face-detection-service
    ports:
      - "8004:8000"
    environment:
      - DATABASE_URL=postgresql://user:password@postgres:5432/facesocial
      - REDIS_URL=redis://redis:6379
      - MODEL_PATH=/app/models
    volumes:
      - ./models/face-detection:/app/models
      - face-detection-logs:/app/logs
    depends_on:
      - postgres
      - redis
    networks:
      - facesocial-network
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 2G
    
  # Age & Gender Detection Service
  age-gender-service:
    build:
      context: ./services/age-gender
      dockerfile: docker/Dockerfile
    container_name: age-gender-service
    ports:
      - "8005:8000"
    environment:
      - DATABASE_URL=postgresql://user:password@postgres:5432/facesocial
      - REDIS_URL=redis://redis:6379
      - MODEL_PATH=/app/models
    volumes:
      - ./models/age-gender:/app/models
      - age-gender-logs:/app/logs
    depends_on:
      - postgres
      - redis
    networks:
      - facesocial-network
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 3G
        reservations:
          memory: 1.5G
    
  # PostgreSQL Database
  postgres:
    image: postgres:15-alpine
    container_name: facesocial-postgres
    environment:
      POSTGRES_DB: facesocial
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
    volumes:
      - postgres-data:/var/lib/postgresql/data
      - ./database/init:/docker-entrypoint-initdb.d
    ports:
      - "5432:5432"
    networks:
      - facesocial-network
    restart: unless-stopped
    
  # Milvus Vector Database
  milvus:
    image: milvusdb/milvus:v2.3.0
    container_name: facesocial-milvus
    command: ["milvus", "run", "standalone"]
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    volumes:
      - milvus-data:/var/lib/milvus
    ports:
      - "19530:19530"
    depends_on:
      - etcd
      - minio
    networks:
      - facesocial-network
    restart: unless-stopped
    
  # Etcd for Milvus
  etcd:
    image: quay.io/coreos/etcd:v3.5.0
    container_name: facesocial-etcd
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
      - ETCD_QUOTA_BACKEND_BYTES=4294967296
      - ETCD_SNAPSHOT_COUNT=50000
    volumes:
      - etcd-data:/etcd
    command: etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379 --data-dir /etcd
    networks:
      - facesocial-network
    restart: unless-stopped
    
  # MinIO for Milvus
  minio:
    image: minio/minio:RELEASE.2023-03-20T20-16-18Z
    container_name: facesocial-minio
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    volumes:
      - minio-data:/data
    command: minio server /data
    ports:
      - "9001:9000"
    networks:
      - facesocial-network
    restart: unless-stopped
    
  # Redis Cache
  redis:
    image: redis:7-alpine
    container_name: facesocial-redis
    command: redis-server --appendonly yes --requirepass redis-password
    volumes:
      - redis-data:/data
    ports:
      - "6379:6379"
    networks:
      - facesocial-network
    restart: unless-stopped
    
  # Celery Worker for async tasks
  celery-worker:
    build:
      context: ./services/deepfake-detection
      dockerfile: docker/Dockerfile
    container_name: facesocial-celery-worker
    command: celery -A app.tasks worker --loglevel=info --concurrency=4
    environment:
      - DATABASE_URL=postgresql://user:password@postgres:5432/facesocial
      - REDIS_URL=redis://redis:6379
      - CELERY_BROKER_URL=redis://redis:6379
    volumes:
      - ./models/deepfake-detection:/app/models
      - celery-logs:/app/logs
    depends_on:
      - redis
      - postgres
    networks:
      - facesocial-network
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 2G
    
  # Celery Beat for scheduled tasks
  celery-beat:
    build:
      context: ./services/deepfake-detection
      dockerfile: docker/Dockerfile
    container_name: facesocial-celery-beat
    command: celery -A app.tasks beat --loglevel=info
    environment:
      - DATABASE_URL=postgresql://user:password@postgres:5432/facesocial
      - REDIS_URL=redis://redis:6379
      - CELERY_BROKER_URL=redis://redis:6379
    volumes:
      - celery-beat-data:/app/celerybeat-schedule
    depends_on:
      - redis
      - postgres
    networks:
      - facesocial-network
    restart: unless-stopped

networks:
  facesocial-network:
    driver: bridge

volumes:
  postgres-data:
  milvus-data:
  etcd-data:
  minio-data:
  redis-data:
  face-recognition-logs:
  antispoofing-logs:
  deepfake-logs:
  face-detection-logs:
  age-gender-logs:
  celery-logs:
  celery-beat-data:
  deepfake-temp:
```

---

## 3. Kubernetes Configuration

### 3.1 Namespace and ConfigMap
```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: facesocial-ai
---
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: facesocial-config
  namespace: facesocial-ai
data:
  DATABASE_URL: "postgresql://user:password@postgres-service:5432/facesocial"
  REDIS_URL: "redis://redis-service:6379"
  MILVUS_HOST: "milvus-service"
  MILVUS_PORT: "19530"
  LOG_LEVEL: "INFO"
  MODEL_CACHE_SIZE: "1000"
```

### 3.2 Services Deployment
```yaml
# k8s/face-recognition-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: face-recognition-service
  namespace: facesocial-ai
  labels:
    app: face-recognition-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: face-recognition-service
  template:
    metadata:
      labels:
        app: face-recognition-service
    spec:
      containers:
      - name: face-recognition
        image: facesocial/face-recognition:latest
        ports:
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: facesocial-config
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        volumeMounts:
        - name: model-storage
          mountPath: /app/models
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: face-recognition-models-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: face-recognition-service
  namespace: facesocial-ai
spec:
  selector:
    app: face-recognition-service
  ports:
  - protocol: TCP
    port: 8000
    targetPort: 8000
  type: ClusterIP
```

### 3.3 Database Services
```yaml
# k8s/postgres-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres
  namespace: facesocial-ai
spec:
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:15-alpine
        env:
        - name: POSTGRES_DB
          value: "facesocial"
        - name: POSTGRES_USER
          value: "user"
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: postgres-secret
              key: password
        ports:
        - containerPort: 5432
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
      volumes:
      - name: postgres-storage
        persistentVolumeClaim:
          claimName: postgres-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: postgres-service
  namespace: facesocial-ai
spec:
  selector:
    app: postgres
  ports:
  - protocol: TCP
    port: 5432
    targetPort: 5432
  type: ClusterIP
```

### 3.4 Ingress Controller
```yaml
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: facesocial-ai-ingress
  namespace: facesocial-ai
  annotations:
    kubernetes.io/ingress.class: "nginx"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/rate-limit-window: "1m"
    nginx.ingress.kubernetes.io/proxy-body-size: "100m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "300"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - api.facesocial.com
    secretName: facesocial-tls
  rules:
  - host: api.facesocial.com
    http:
      paths:
      - path: /api/v1/face-recognition
        pathType: Prefix
        backend:
          service:
            name: face-recognition-service
            port:
              number: 8000
      - path: /api/v1/antispoofing
        pathType: Prefix
        backend:
          service:
            name: antispoofing-service
            port:
              number: 8000
      - path: /api/v1/deepfake-detection
        pathType: Prefix
        backend:
          service:
            name: deepfake-detection-service
            port:
              number: 8000
      - path: /api/v1/face-detection
        pathType: Prefix
        backend:
          service:
            name: face-detection-service
            port:
              number: 8000
      - path: /api/v1/demographics
        pathType: Prefix
        backend:
          service:
            name: age-gender-service
            port:
              number: 8000
```

---

## 4. Monitoring และ Logging

### 4.1 Prometheus Configuration
```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

scrape_configs:
  - job_name: 'face-recognition-service'
    static_configs:
      - targets: ['face-recognition-service:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s
    
  - job_name: 'antispoofing-service'
    static_configs:
      - targets: ['antispoofing-service:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s
    
  - job_name: 'deepfake-detection-service'
    static_configs:
      - targets: ['deepfake-detection-service:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s
    
  - job_name: 'face-detection-service'
    static_configs:
      - targets: ['face-detection-service:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s
    
  - job_name: 'age-gender-service'
    static_configs:
      - targets: ['age-gender-service:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### 4.2 Grafana Dashboards
```json
{
  "dashboard": {
    "title": "FaceSocial AI Services Dashboard",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{service}} - {{method}}"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total{status=~\"5..\"}[5m])",
            "legendFormat": "{{service}} - Errors"
          }
        ]
      },
      {
        "title": "AI Model Performance",
        "type": "graph",
        "targets": [
          {
            "expr": "ai_model_inference_duration_seconds",
            "legendFormat": "{{model_name}} - Inference Time"
          }
        ]
      }
    ]
  }
}
```

---

## 5. Deployment Scripts

### 5.1 Production Deployment Script
```bash
#!/bin/bash
# deploy.sh

set -e

# Configuration
ENVIRONMENT=${1:-production}
NAMESPACE="facesocial-ai"
DOCKER_REGISTRY="your-registry.com"

echo "🚀 Deploying FaceSocial AI Services to $ENVIRONMENT"

# Build and push Docker images
echo "📦 Building Docker images..."
services=("face-recognition" "antispoofing" "deepfake-detection" "face-detection" "age-gender")

for service in "${services[@]}"; do
    echo "Building $service..."
    docker build -t "$DOCKER_REGISTRY/facesocial-$service:latest" ./services/$service
    docker push "$DOCKER_REGISTRY/facesocial-$service:latest"
done

# Apply Kubernetes configurations
echo "🔧 Applying Kubernetes configurations..."
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secrets.yaml

# Deploy databases first
echo "🗄️ Deploying databases..."
kubectl apply -f k8s/postgres-deployment.yaml
kubectl apply -f k8s/redis-deployment.yaml
kubectl apply -f k8s/milvus-deployment.yaml

# Wait for databases to be ready
echo "⏳ Waiting for databases to be ready..."
kubectl wait --for=condition=ready pod -l app=postgres -n $NAMESPACE --timeout=300s
kubectl wait --for=condition=ready pod -l app=redis -n $NAMESPACE --timeout=300s
kubectl wait --for=condition=ready pod -l app=milvus -n $NAMESPACE --timeout=300s

# Deploy AI services
echo "🤖 Deploying AI services..."
for service in "${services[@]}"; do
    kubectl apply -f k8s/$service-deployment.yaml
done

# Deploy ingress
echo "🌐 Deploying ingress..."
kubectl apply -f k8s/ingress.yaml

# Wait for services to be ready
echo "⏳ Waiting for AI services to be ready..."
for service in "${services[@]}"; do
    kubectl wait --for=condition=ready pod -l app=$service-service -n $NAMESPACE --timeout=600s
done

echo "✅ Deployment completed successfully!"

# Run health checks
echo "🔍 Running health checks..."
kubectl get pods -n $NAMESPACE
kubectl get services -n $NAMESPACE
kubectl get ingress -n $NAMESPACE

echo "🎉 FaceSocial AI Services deployed successfully!"
```

### 5.2 Database Migration Script
```bash
#!/bin/bash
# migrate.sh

set -e

echo "🗄️ Running database migrations..."

# PostgreSQL migrations
echo "Running PostgreSQL migrations..."
kubectl exec -it deployment/postgres -n facesocial-ai -- psql -U user -d facesocial -c "
CREATE SCHEMA IF NOT EXISTS core;
CREATE SCHEMA IF NOT EXISTS face_recognition;
CREATE SCHEMA IF NOT EXISTS face_detection;
CREATE SCHEMA IF NOT EXISTS antispoofing;
CREATE SCHEMA IF NOT EXISTS deepfake_detection;
CREATE SCHEMA IF NOT EXISTS demographics;
CREATE SCHEMA IF NOT EXISTS audit;
CREATE SCHEMA IF NOT EXISTS config;
"

# Run service-specific migrations
services=("face-recognition" "antispoofing" "deepfake-detection" "face-detection" "age-gender")

for service in "${services[@]}"; do
    echo "Running migrations for $service..."
    kubectl exec -it deployment/$service-service -n facesocial-ai -- python -m alembic upgrade head
done

echo "✅ Database migrations completed!"
```

---

## 6. Security Configuration

### 6.1 Network Policies
```yaml
# k8s/network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: facesocial-ai-network-policy
  namespace: facesocial-ai
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
  - from:
    - podSelector:
        matchLabels:
          app: prometheus
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: postgres
    ports:
    - protocol: TCP
      port: 5432
  - to:
    - podSelector:
        matchLabels:
          app: redis
    ports:
    - protocol: TCP
      port: 6379
  - to:
    - podSelector:
        matchLabels:
          app: milvus
    ports:
    - protocol: TCP
      port: 19530
```

### 6.2 RBAC Configuration
```yaml
# k8s/rbac.yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: facesocial-ai-sa
  namespace: facesocial-ai
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: facesocial-ai
  name: facesocial-ai-role
rules:
- apiGroups: [""]
  resources: ["pods", "services"]
  verbs: ["get", "list"]
- apiGroups: ["apps"]
  resources: ["deployments"]
  verbs: ["get", "list"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: facesocial-ai-rolebinding
  namespace: facesocial-ai
subjects:
- kind: ServiceAccount
  name: facesocial-ai-sa
  namespace: facesocial-ai
roleRef:
  kind: Role
  name: facesocial-ai-role
  apiGroup: rbac.authorization.k8s.io
```

---

## 7. Backup และ Disaster Recovery

### 7.1 Database Backup Strategy
```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d_%H%M%S)

# PostgreSQL backup
echo "📦 Creating PostgreSQL backup..."
kubectl exec deployment/postgres -n facesocial-ai -- pg_dump -U user facesocial > "$BACKUP_DIR/postgres_$DATE.sql"

# Milvus backup
echo "📦 Creating Milvus backup..."
kubectl exec deployment/milvus -n facesocial-ai -- tar -czf - /var/lib/milvus > "$BACKUP_DIR/milvus_$DATE.tar.gz"

# Redis backup
echo "📦 Creating Redis backup..."
kubectl exec deployment/redis -n facesocial-ai -- redis-cli BGSAVE
kubectl exec deployment/redis -n facesocial-ai -- tar -czf - /data > "$BACKUP_DIR/redis_$DATE.tar.gz"

echo "✅ Backup completed: $DATE"
```

Infrastructure Configuration นี้ให้ความครอบคลุมสำหรับการ deploy และจัดการ Backend AI Services ทั้ง 5 บริการ พร้อม monitoring, logging, security และ backup strategies ที่จำเป็นสำหรับ production environment
