# Monitoring และ Analytics Dashboard

## 1. System Monitoring Architecture

### 1.1 Monitoring Stack Overview
```
┌─────────────────────────────────────────────────────────────────┐
│                    FaceSocial AI Monitoring                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐            │
│  │  Prometheus  │ │   Grafana    │ │ AlertManager │            │
│  │   Metrics    │ │  Dashboard   │ │  Alerting    │            │
│  │  Collection  │ │ Visualization│ │   System     │            │
│  └──────────────┘ └──────────────┘ └──────────────┘            │
│         │                  │                  │                 │
│         ▼                  ▼                  ▼                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  AI Services                            │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────┐   │   │
│  │  │Face Recog.  │ │Antispoofing │ │Deepfake Detect. │   │   │
│  │  └─────────────┘ └─────────────┘ └─────────────────┘   │   │
│  │  ┌─────────────┐ ┌─────────────┐                       │   │
│  │  │Face Detect. │ │Age & Gender │                       │   │
│  │  └─────────────┘ └─────────────┘                       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐            │
│  │    Jaeger    │ │     ELK      │ │   Business   │            │
│  │   Tracing    │ │   Logging    │ │  Analytics   │            │
│  └──────────────┘ └──────────────┘ └──────────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Metrics Collection Strategy

#### Service-Level Metrics
```python
# app/utils/metrics.py
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
import time
from functools import wraps

# Registry for each service
REGISTRY = CollectorRegistry()

# Request metrics
REQUEST_COUNT = Counter(
    'ai_service_requests_total',
    'Total number of requests',
    ['service', 'endpoint', 'method', 'status'],
    registry=REGISTRY
)

REQUEST_DURATION = Histogram(
    'ai_service_request_duration_seconds',
    'Request duration in seconds',
    ['service', 'endpoint'],
    registry=REGISTRY
)

# AI-specific metrics
MODEL_INFERENCE_DURATION = Histogram(
    'ai_model_inference_duration_seconds',
    'Model inference duration in seconds',
    ['service', 'model_name', 'model_version'],
    registry=REGISTRY
)

FACE_DETECTION_COUNT = Counter(
    'faces_detected_total',
    'Total number of faces detected',
    ['service', 'confidence_level'],
    registry=REGISTRY
)

ACCURACY_SCORE = Gauge(
    'ai_model_accuracy_score',
    'Current model accuracy score',
    ['service', 'model_name', 'metric_type'],
    registry=REGISTRY
)

ERROR_COUNT = Counter(
    'ai_service_errors_total',
    'Total number of errors',
    ['service', 'error_type', 'error_code'],
    registry=REGISTRY
)

# Resource metrics
MEMORY_USAGE = Gauge(
    'ai_service_memory_usage_bytes',
    'Memory usage in bytes',
    ['service'],
    registry=REGISTRY
)

GPU_UTILIZATION = Gauge(
    'ai_service_gpu_utilization_percent',
    'GPU utilization percentage',
    ['service', 'gpu_id'],
    registry=REGISTRY
)

def track_requests(service_name: str):
    """Decorator to track API requests"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            endpoint = func.__name__
            status = "success"
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                status = "error"
                ERROR_COUNT.labels(
                    service=service_name,
                    error_type=type(e).__name__,
                    error_code=getattr(e, 'code', 'unknown')
                ).inc()
                raise
            finally:
                duration = time.time() - start_time
                REQUEST_DURATION.labels(
                    service=service_name,
                    endpoint=endpoint
                ).observe(duration)
                REQUEST_COUNT.labels(
                    service=service_name,
                    endpoint=endpoint,
                    method="POST",
                    status=status
                ).inc()
                
        return wrapper
    return decorator

def track_model_inference(service_name: str, model_name: str):
    """Decorator to track model inference time"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                MODEL_INFERENCE_DURATION.labels(
                    service=service_name,
                    model_name=model_name,
                    model_version="v1.0"
                ).observe(duration)
                
        return wrapper
    return decorator

class MetricsCollector:
    def __init__(self, service_name: str):
        self.service_name = service_name
        
    def record_face_detected(self, confidence: float):
        """Record face detection event"""
        confidence_level = "high" if confidence > 0.8 else "medium" if confidence > 0.5 else "low"
        FACE_DETECTION_COUNT.labels(
            service=self.service_name,
            confidence_level=confidence_level
        ).inc()
        
    def update_accuracy_score(self, model_name: str, accuracy: float, metric_type: str):
        """Update model accuracy score"""
        ACCURACY_SCORE.labels(
            service=self.service_name,
            model_name=model_name,
            metric_type=metric_type
        ).set(accuracy)
        
    def record_memory_usage(self, memory_bytes: int):
        """Record memory usage"""
        MEMORY_USAGE.labels(service=self.service_name).set(memory_bytes)
        
    def record_gpu_utilization(self, gpu_id: int, utilization: float):
        """Record GPU utilization"""
        GPU_UTILIZATION.labels(
            service=self.service_name,
            gpu_id=str(gpu_id)
        ).set(utilization)
```

---

## 2. Grafana Dashboards

### 2.1 Main AI Services Dashboard
```json
{
  "dashboard": {
    "id": null,
    "title": "FaceSocial AI Services Overview",
    "tags": ["facesocial", "ai", "services"],
    "timezone": "browser",
    "refresh": "30s",
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "panels": [
      {
        "id": 1,
        "title": "Service Health Status",
        "type": "stat",
        "gridPos": {"h": 8, "w": 24, "x": 0, "y": 0},
        "targets": [
          {
            "expr": "up{job=~\".*-service\"}",
            "legendFormat": "{{instance}}",
            "refId": "A"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "mappings": [
              {"options": {"0": {"text": "DOWN"}}, "type": "value"},
              {"options": {"1": {"text": "UP"}}, "type": "value"}
            ],
            "thresholds": {
              "steps": [
                {"color": "red", "value": 0},
                {"color": "green", "value": 1}
              ]
            }
          }
        }
      },
      {
        "id": 2,
        "title": "Request Rate (req/sec)",
        "type": "graph",
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
        "targets": [
          {
            "expr": "sum(rate(ai_service_requests_total[5m])) by (service)",
            "legendFormat": "{{service}}",
            "refId": "A"
          }
        ],
        "yAxes": [
          {
            "label": "Requests/sec",
            "min": 0
          }
        ],
        "legend": {
          "displayMode": "table",
          "placement": "bottom"
        }
      },
      {
        "id": 3,
        "title": "Response Time (95th percentile)",
        "type": "graph",
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
        "targets": [
          {
            "expr": "histogram_quantile(0.95, sum(rate(ai_service_request_duration_seconds_bucket[5m])) by (service, le))",
            "legendFormat": "{{service}}",
            "refId": "A"
          }
        ],
        "yAxes": [
          {
            "label": "Seconds",
            "min": 0
          }
        ]
      },
      {
        "id": 4,
        "title": "Error Rate (%)",
        "type": "graph",
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 16},
        "targets": [
          {
            "expr": "sum(rate(ai_service_requests_total{status=\"error\"}[5m])) by (service) / sum(rate(ai_service_requests_total[5m])) by (service) * 100",
            "legendFormat": "{{service}}",
            "refId": "A"
          }
        ],
        "yAxes": [
          {
            "label": "Percent",
            "min": 0,
            "max": 100
          }
        ],
        "alert": {
          "conditions": [
            {
              "query": {"params": ["A", "5m", "now"]},
              "reducer": {"params": [], "type": "avg"},
              "evaluator": {"params": [5], "type": "gt"}
            }
          ],
          "executionErrorState": "alerting",
          "for": "5m",
          "frequency": "10s",
          "handler": 1,
          "name": "High Error Rate",
          "noDataState": "no_data"
        }
      },
      {
        "id": 5,
        "title": "Model Inference Time",
        "type": "graph",
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 16},
        "targets": [
          {
            "expr": "histogram_quantile(0.95, sum(rate(ai_model_inference_duration_seconds_bucket[5m])) by (service, model_name, le))",
            "legendFormat": "{{service}} - {{model_name}}",
            "refId": "A"
          }
        ]
      },
      {
        "id": 6,
        "title": "Faces Detected per Minute",
        "type": "graph",
        "gridPos": {"h": 8, "w": 24, "x": 0, "y": 24},
        "targets": [
          {
            "expr": "sum(rate(faces_detected_total[1m])) by (service, confidence_level) * 60",
            "legendFormat": "{{service}} - {{confidence_level}} confidence",
            "refId": "A"
          }
        ]
      }
    ]
  }
}
```

### 2.2 Face Recognition Service Dashboard
```json
{
  "dashboard": {
    "title": "Face Recognition Service Detailed Metrics",
    "panels": [
      {
        "id": 1,
        "title": "Recognition Accuracy",
        "type": "stat",
        "targets": [
          {
            "expr": "ai_model_accuracy_score{service=\"face-recognition\", metric_type=\"recognition_accuracy\"}",
            "refId": "A"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "percent",
            "thresholds": {
              "steps": [
                {"color": "red", "value": 0},
                {"color": "yellow", "value": 95},
                {"color": "green", "value": 99}
              ]
            }
          }
        }
      },
      {
        "id": 2,
        "title": "Face Registration vs Verification",
        "type": "graph",
        "targets": [
          {
            "expr": "sum(rate(ai_service_requests_total{service=\"face-recognition\", endpoint=\"register\"}[5m]))",
            "legendFormat": "Registrations",
            "refId": "A"
          },
          {
            "expr": "sum(rate(ai_service_requests_total{service=\"face-recognition\", endpoint=\"verify\"}[5m]))",
            "legendFormat": "Verifications",
            "refId": "B"
          },
          {
            "expr": "sum(rate(ai_service_requests_total{service=\"face-recognition\", endpoint=\"identify\"}[5m]))",
            "legendFormat": "Identifications",
            "refId": "C"
          }
        ]
      },
      {
        "id": 3,
        "title": "Face Database Size",
        "type": "stat",
        "targets": [
          {
            "expr": "face_database_total_faces{service=\"face-recognition\"}",
            "refId": "A"
          }
        ]
      },
      {
        "id": 4,
        "title": "Vector Search Performance",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, sum(rate(vector_search_duration_seconds_bucket[5m])) by (le))",
            "legendFormat": "95th percentile",
            "refId": "A"
          },
          {
            "expr": "histogram_quantile(0.50, sum(rate(vector_search_duration_seconds_bucket[5m])) by (le))",
            "legendFormat": "50th percentile",
            "refId": "B"
          }
        ]
      }
    ]
  }
}
```

### 2.3 Resource Utilization Dashboard
```json
{
  "dashboard": {
    "title": "AI Services Resource Utilization",
    "panels": [
      {
        "id": 1,
        "title": "Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "ai_service_memory_usage_bytes / 1024 / 1024 / 1024",
            "legendFormat": "{{service}}",
            "refId": "A"
          }
        ],
        "yAxes": [
          {
            "label": "GB",
            "min": 0
          }
        ]
      },
      {
        "id": 2,
        "title": "GPU Utilization",
        "type": "graph",
        "targets": [
          {
            "expr": "ai_service_gpu_utilization_percent",
            "legendFormat": "{{service}} - GPU {{gpu_id}}",
            "refId": "A"
          }
        ],
        "yAxes": [
          {
            "label": "Percent",
            "min": 0,
            "max": 100
          }
        ]
      },
      {
        "id": 3,
        "title": "CPU Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(container_cpu_usage_seconds_total{container=~\".*-service\"}[5m]) * 100",
            "legendFormat": "{{container}}",
            "refId": "A"
          }
        ]
      }
    ]
  }
}
```

---

## 3. Business Analytics Dashboard

### 3.1 AI Usage Analytics
```python
# app/analytics/business_metrics.py
from typing import Dict, Any, List
from datetime import datetime, timedelta
import pandas as pd
from sqlalchemy import select, func, and_

class BusinessAnalytics:
    def __init__(self, db_manager):
        self.db_manager = db_manager
        
    async def get_ai_usage_summary(
        self, 
        date_from: datetime, 
        date_to: datetime
    ) -> Dict[str, Any]:
        """Get comprehensive AI usage analytics"""
        
        async with self.db_manager.get_session() as db:
            # Service usage statistics
            service_stats = await self._get_service_usage_stats(db, date_from, date_to)
            
            # User engagement metrics
            user_metrics = await self._get_user_engagement_metrics(db, date_from, date_to)
            
            # Performance trends
            performance_trends = await self._get_performance_trends(db, date_from, date_to)
            
            # Error analysis
            error_analysis = await self._get_error_analysis(db, date_from, date_to)
            
            # Demographics insights
            demographics = await self._get_demographics_insights(db, date_from, date_to)
            
            return {
                "summary": {
                    "total_requests": service_stats["total_requests"],
                    "active_users": user_metrics["active_users"],
                    "avg_response_time": performance_trends["avg_response_time"],
                    "success_rate": (1 - error_analysis["error_rate"]) * 100
                },
                "service_usage": service_stats,
                "user_engagement": user_metrics,
                "performance": performance_trends,
                "errors": error_analysis,
                "demographics": demographics,
                "recommendations": await self._generate_recommendations()
            }
            
    async def _get_service_usage_stats(
        self, db, date_from: datetime, date_to: datetime
    ) -> Dict[str, Any]:
        """Get service usage statistics"""
        
        # Face Recognition usage
        face_recog_stats = await self._query_service_stats(
            db, "face_recognition", date_from, date_to
        )
        
        # Face Detection usage
        face_detect_stats = await self._query_service_stats(
            db, "face_detection", date_from, date_to
        )
        
        # Antispoofing usage
        antispoofing_stats = await self._query_service_stats(
            db, "antispoofing", date_from, date_to
        )
        
        # Deepfake Detection usage
        deepfake_stats = await self._query_service_stats(
            db, "deepfake_detection", date_from, date_to
        )
        
        # Age & Gender usage
        demographics_stats = await self._query_service_stats(
            db, "demographics", date_from, date_to
        )
        
        total_requests = (
            face_recog_stats["requests"] + 
            face_detect_stats["requests"] + 
            antispoofing_stats["requests"] + 
            deepfake_stats["requests"] + 
            demographics_stats["requests"]
        )
        
        return {
            "total_requests": total_requests,
            "by_service": {
                "face_recognition": face_recog_stats,
                "face_detection": face_detect_stats,
                "antispoofing": antispoofing_stats,
                "deepfake_detection": deepfake_stats,
                "age_gender": demographics_stats
            },
            "most_used_service": max(
                [
                    ("face_recognition", face_recog_stats["requests"]),
                    ("face_detection", face_detect_stats["requests"]),
                    ("antispoofing", antispoofing_stats["requests"]),
                    ("deepfake_detection", deepfake_stats["requests"]),
                    ("age_gender", demographics_stats["requests"])
                ],
                key=lambda x: x[1]
            )[0]
        }
        
    async def _get_user_engagement_metrics(
        self, db, date_from: datetime, date_to: datetime
    ) -> Dict[str, Any]:
        """Get user engagement metrics"""
        
        # Daily active users
        dau_query = select(func.count(func.distinct("user_id"))).where(
            and_(
                "created_at >= %s" % date_from,
                "created_at <= %s" % date_to
            )
        )
        
        # User retention analysis
        retention_data = await self._calculate_user_retention(db, date_from, date_to)
        
        # Feature adoption rates
        feature_adoption = await self._calculate_feature_adoption(db, date_from, date_to)
        
        return {
            "active_users": 0,  # Placeholder for actual query result
            "retention": retention_data,
            "feature_adoption": feature_adoption,
            "engagement_trends": await self._get_engagement_trends(db, date_from, date_to)
        }
        
    async def generate_insights_report(
        self, 
        date_from: datetime, 
        date_to: datetime
    ) -> Dict[str, Any]:
        """Generate comprehensive insights report"""
        
        analytics = await self.get_ai_usage_summary(date_from, date_to)
        
        insights = {
            "key_findings": [
                f"Face Recognition is the most used service with {analytics['service_usage']['by_service']['face_recognition']['requests']} requests",
                f"Average response time is {analytics['performance']['avg_response_time']:.2f}ms",
                f"Success rate is {analytics['summary']['success_rate']:.1f}%",
                f"Peak usage occurs during {analytics['user_engagement']['engagement_trends']['peak_hours']}"
            ],
            "recommendations": [
                "Consider scaling Face Recognition service during peak hours",
                "Optimize model loading for better response times",
                "Implement caching for frequently accessed face embeddings",
                "Monitor and alert on error rates exceeding 2%"
            ],
            "trends": {
                "usage_growth": analytics["service_usage"]["growth_rate"],
                "performance_improvement": analytics["performance"]["improvement_rate"],
                "error_reduction": analytics["errors"]["error_reduction"]
            }
        }
        
        return insights
```

### 3.2 Real-time Analytics Dashboard
```javascript
// Frontend Dashboard Component
class AIAnalyticsDashboard {
    constructor() {
        this.ws = new WebSocket('wss://api.facesocial.com/analytics/realtime');
        this.charts = {};
        this.setupCharts();
        this.connectWebSocket();
    }
    
    setupCharts() {
        // Request rate chart
        this.charts.requestRate = new Chart(document.getElementById('requestRateChart'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Face Recognition',
                        data: [],
                        borderColor: 'rgb(75, 192, 192)',
                        tension: 0.1
                    },
                    {
                        label: 'Face Detection',
                        data: [],
                        borderColor: 'rgb(255, 99, 132)',
                        tension: 0.1
                    },
                    {
                        label: 'Antispoofing',
                        data: [],
                        borderColor: 'rgb(54, 162, 235)',
                        tension: 0.1
                    }
                ]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Requests/sec'
                        }
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'Real-time Request Rate'
                    }
                }
            }
        });
        
        // Response time chart
        this.charts.responseTime = new Chart(document.getElementById('responseTimeChart'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: '95th Percentile',
                        data: [],
                        borderColor: 'rgb(255, 206, 86)',
                        tension: 0.1
                    },
                    {
                        label: 'Average',
                        data: [],
                        borderColor: 'rgb(75, 192, 192)',
                        tension: 0.1
                    }
                ]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Response Time (ms)'
                        }
                    }
                }
            }
        });
        
        // Error rate gauge
        this.charts.errorRate = new Chart(document.getElementById('errorRateChart'), {
            type: 'doughnut',
            data: {
                labels: ['Success', 'Error'],
                datasets: [{
                    data: [99, 1],
                    backgroundColor: ['rgb(75, 192, 192)', 'rgb(255, 99, 132)']
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: 'Success Rate'
                    }
                }
            }
        });
    }
    
    connectWebSocket() {
        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.updateCharts(data);
            this.updateKPIs(data);
        };
        
        this.ws.onclose = () => {
            // Reconnect after 5 seconds
            setTimeout(() => this.connectWebSocket(), 5000);
        };
    }
    
    updateCharts(data) {
        const timestamp = new Date().toLocaleTimeString();
        
        // Update request rate chart
        const requestRateChart = this.charts.requestRate;
        requestRateChart.data.labels.push(timestamp);
        
        requestRateChart.data.datasets[0].data.push(data.metrics.face_recognition.request_rate);
        requestRateChart.data.datasets[1].data.push(data.metrics.face_detection.request_rate);
        requestRateChart.data.datasets[2].data.push(data.metrics.antispoofing.request_rate);
        
        // Keep only last 20 data points
        if (requestRateChart.data.labels.length > 20) {
            requestRateChart.data.labels.shift();
            requestRateChart.data.datasets.forEach(dataset => dataset.data.shift());
        }
        
        requestRateChart.update('none');
        
        // Update response time chart
        const responseTimeChart = this.charts.responseTime;
        responseTimeChart.data.labels.push(timestamp);
        responseTimeChart.data.datasets[0].data.push(data.metrics.overall.response_time_95th);
        responseTimeChart.data.datasets[1].data.push(data.metrics.overall.response_time_avg);
        
        if (responseTimeChart.data.labels.length > 20) {
            responseTimeChart.data.labels.shift();
            responseTimeChart.data.datasets.forEach(dataset => dataset.data.shift());
        }
        
        responseTimeChart.update('none');
        
        // Update error rate
        const errorRate = data.metrics.overall.error_rate * 100;
        const successRate = 100 - errorRate;
        this.charts.errorRate.data.datasets[0].data = [successRate, errorRate];
        this.charts.errorRate.update();
    }
    
    updateKPIs(data) {
        // Update KPI cards
        document.getElementById('totalRequests').textContent = 
            data.metrics.overall.total_requests.toLocaleString();
        document.getElementById('avgResponseTime').textContent = 
            `${data.metrics.overall.response_time_avg}ms`;
        document.getElementById('successRate').textContent = 
            `${((1 - data.metrics.overall.error_rate) * 100).toFixed(1)}%`;
        document.getElementById('activeUsers').textContent = 
            data.metrics.overall.active_users.toLocaleString();
            
        // Update service statuses
        Object.keys(data.metrics).forEach(service => {
            if (service !== 'overall') {
                const statusElement = document.getElementById(`${service}Status`);
                const isHealthy = data.metrics[service].error_rate < 0.05;
                statusElement.className = isHealthy ? 'status-healthy' : 'status-error';
                statusElement.textContent = isHealthy ? 'Healthy' : 'Issues';
            }
        });
    }
}

// Initialize dashboard when page loads
document.addEventListener('DOMContentLoaded', () => {
    new AIAnalyticsDashboard();
});
```

---

## 4. Alert Configuration

### 4.1 Prometheus Alert Rules
```yaml
# monitoring/alert_rules.yml
groups:
- name: ai_services_alerts
  rules:
  # High error rate alert
  - alert: HighErrorRate
    expr: |
      (
        sum(rate(ai_service_requests_total{status="error"}[5m])) by (service) /
        sum(rate(ai_service_requests_total[5m])) by (service)
      ) * 100 > 5
    for: 2m
    labels:
      severity: warning
      service: "{{ $labels.service }}"
    annotations:
      summary: "High error rate detected for {{ $labels.service }}"
      description: "Error rate is {{ $value | humanizePercentage }} for service {{ $labels.service }}"
      
  # Slow response time alert
  - alert: SlowResponseTime
    expr: |
      histogram_quantile(0.95, 
        sum(rate(ai_service_request_duration_seconds_bucket[5m])) by (service, le)
      ) > 2
    for: 5m
    labels:
      severity: warning
      service: "{{ $labels.service }}"
    annotations:
      summary: "Slow response time for {{ $labels.service }}"
      description: "95th percentile response time is {{ $value }}s for {{ $labels.service }}"
      
  # Service down alert
  - alert: ServiceDown
    expr: up{job=~".*-service"} == 0
    for: 1m
    labels:
      severity: critical
      service: "{{ $labels.job }}"
    annotations:
      summary: "Service {{ $labels.job }} is down"
      description: "Service {{ $labels.job }} has been down for more than 1 minute"
      
  # High memory usage alert
  - alert: HighMemoryUsage
    expr: |
      ai_service_memory_usage_bytes / 1024 / 1024 / 1024 > 3
    for: 5m
    labels:
      severity: warning
      service: "{{ $labels.service }}"
    annotations:
      summary: "High memory usage for {{ $labels.service }}"
      description: "Memory usage is {{ $value | humanizeBytes }} for {{ $labels.service }}"
      
  # Model accuracy degradation alert
  - alert: ModelAccuracyDegraded
    expr: |
      ai_model_accuracy_score < 0.95
    for: 10m
    labels:
      severity: critical
      service: "{{ $labels.service }}"
      model: "{{ $labels.model_name }}"
    annotations:
      summary: "Model accuracy degraded for {{ $labels.model_name }}"
      description: "Accuracy is {{ $value | humanizePercentage }} for model {{ $labels.model_name }} in service {{ $labels.service }}"
      
  # Face detection rate anomaly
  - alert: FaceDetectionAnomaly
    expr: |
      (
        rate(faces_detected_total[5m]) < 
        rate(faces_detected_total[1h] offset 1d) * 0.5
      ) and rate(faces_detected_total[5m]) > 0
    for: 10m
    labels:
      severity: warning
      service: "face-detection"
    annotations:
      summary: "Face detection rate anomaly"
      description: "Current face detection rate is significantly lower than usual"
```

### 4.2 AlertManager Configuration
```yaml
# monitoring/alertmanager.yml
global:
  smtp_smarthost: 'localhost:587'
  smtp_from: 'alerts@facesocial.com'

route:
  group_by: ['alertname', 'service']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'
  
  routes:
  - match:
      severity: critical
    receiver: 'critical-alerts'
    repeat_interval: 5m
    
  - match:
      severity: warning
    receiver: 'warning-alerts'
    repeat_interval: 30m

receivers:
- name: 'web.hook'
  webhook_configs:
  - url: 'http://localhost:5001/'
    
- name: 'critical-alerts'
  email_configs:
  - to: 'ops-team@facesocial.com'
    subject: 'CRITICAL: {{ .GroupLabels.service }} Service Alert'
    body: |
      {{ range .Alerts }}
      Alert: {{ .Annotations.summary }}
      Description: {{ .Annotations.description }}
      Service: {{ .Labels.service }}
      Time: {{ .StartsAt }}
      {{ end }}
  slack_configs:
  - api_url: 'YOUR_SLACK_WEBHOOK_URL'
    channel: '#critical-alerts'
    title: 'Critical Alert: {{ .GroupLabels.service }}'
    text: |
      {{ range .Alerts }}
      {{ .Annotations.summary }}
      {{ .Annotations.description }}
      {{ end }}
    
- name: 'warning-alerts'
  email_configs:
  - to: 'dev-team@facesocial.com'
    subject: 'WARNING: {{ .GroupLabels.service }} Service Alert'
    body: |
      {{ range .Alerts }}
      Alert: {{ .Annotations.summary }}
      Description: {{ .Annotations.description }}
      {{ end }}
```

---

## 5. Performance Analytics

### 5.1 Custom Performance Metrics
```python
# app/analytics/performance_analyzer.py
import numpy as np
from typing import Dict, List, Any
from datetime import datetime, timedelta

class PerformanceAnalyzer:
    def __init__(self, metrics_collector):
        self.metrics = metrics_collector
        
    async def analyze_service_performance(
        self, 
        service_name: str,
        time_window_hours: int = 24
    ) -> Dict[str, Any]:
        """Analyze performance metrics for a specific service"""
        
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=time_window_hours)
        
        # Get performance data
        response_times = await self._get_response_times(service_name, start_time, end_time)
        error_rates = await self._get_error_rates(service_name, start_time, end_time)
        throughput = await self._get_throughput(service_name, start_time, end_time)
        resource_usage = await self._get_resource_usage(service_name, start_time, end_time)
        
        # Calculate performance scores
        performance_score = self._calculate_performance_score(
            response_times, error_rates, throughput, resource_usage
        )
        
        # Identify trends and anomalies
        trends = self._analyze_trends(response_times, error_rates, throughput)
        anomalies = self._detect_anomalies(response_times, error_rates)
        
        # Generate recommendations
        recommendations = self._generate_performance_recommendations(
            performance_score, trends, anomalies
        )
        
        return {
            "service": service_name,
            "time_window": f"{time_window_hours}h",
            "performance_score": performance_score,
            "metrics": {
                "avg_response_time": np.mean(response_times),
                "p95_response_time": np.percentile(response_times, 95),
                "avg_error_rate": np.mean(error_rates),
                "avg_throughput": np.mean(throughput),
                "peak_throughput": np.max(throughput)
            },
            "trends": trends,
            "anomalies": anomalies,
            "recommendations": recommendations,
            "status": self._get_service_status(performance_score)
        }
        
    def _calculate_performance_score(
        self, 
        response_times: List[float],
        error_rates: List[float],
        throughput: List[float],
        resource_usage: List[float]
    ) -> float:
        """Calculate overall performance score (0-100)"""
        
        # Response time score (lower is better)
        avg_response_time = np.mean(response_times)
        response_score = max(0, 100 - (avg_response_time / 2.0) * 100)  # Target: <2s
        
        # Error rate score (lower is better)
        avg_error_rate = np.mean(error_rates)
        error_score = max(0, 100 - (avg_error_rate * 100) * 20)  # Target: <5%
        
        # Throughput score (higher is better, but with diminishing returns)
        avg_throughput = np.mean(throughput)
        throughput_score = min(100, (avg_throughput / 100) * 100)  # Target: 100 req/s
        
        # Resource efficiency score
        avg_resource_usage = np.mean(resource_usage)
        resource_score = max(0, 100 - (avg_resource_usage - 0.7) * 200)  # Target: <70%
        
        # Weighted average
        weights = {"response": 0.3, "error": 0.3, "throughput": 0.2, "resource": 0.2}
        
        overall_score = (
            weights["response"] * response_score +
            weights["error"] * error_score +
            weights["throughput"] * throughput_score +
            weights["resource"] * resource_score
        )
        
        return round(overall_score, 2)
        
    def _analyze_trends(
        self, 
        response_times: List[float],
        error_rates: List[float],
        throughput: List[float]
    ) -> Dict[str, str]:
        """Analyze performance trends"""
        
        def calculate_trend(values: List[float]) -> str:
            if len(values) < 2:
                return "insufficient_data"
            
            # Simple linear regression slope
            x = np.arange(len(values))
            slope = np.polyfit(x, values, 1)[0]
            
            if abs(slope) < 0.01:
                return "stable"
            elif slope > 0:
                return "increasing"
            else:
                return "decreasing"
        
        return {
            "response_time": calculate_trend(response_times),
            "error_rate": calculate_trend(error_rates),
            "throughput": calculate_trend(throughput)
        }
        
    def _detect_anomalies(
        self, 
        response_times: List[float],
        error_rates: List[float]
    ) -> List[Dict[str, Any]]:
        """Detect performance anomalies"""
        
        anomalies = []
        
        # Response time anomalies (using z-score)
        if len(response_times) > 10:
            rt_mean = np.mean(response_times)
            rt_std = np.std(response_times)
            
            for i, rt in enumerate(response_times):
                z_score = abs(rt - rt_mean) / rt_std if rt_std > 0 else 0
                if z_score > 2.5:  # 2.5 sigma threshold
                    anomalies.append({
                        "type": "response_time_spike",
                        "timestamp": i,
                        "value": rt,
                        "z_score": z_score
                    })
        
        # Error rate spikes
        error_threshold = 0.1  # 10%
        for i, error_rate in enumerate(error_rates):
            if error_rate > error_threshold:
                anomalies.append({
                    "type": "error_rate_spike",
                    "timestamp": i,
                    "value": error_rate
                })
        
        return anomalies
        
    def _generate_performance_recommendations(
        self,
        performance_score: float,
        trends: Dict[str, str],
        anomalies: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate performance improvement recommendations"""
        
        recommendations = []
        
        if performance_score < 70:
            recommendations.append("Overall performance needs attention")
            
        if trends["response_time"] == "increasing":
            recommendations.append("Response time is trending upward - consider scaling or optimization")
            
        if trends["error_rate"] == "increasing":
            recommendations.append("Error rate is increasing - investigate recent changes")
            
        if trends["throughput"] == "decreasing":
            recommendations.append("Throughput is declining - check for bottlenecks")
            
        if len(anomalies) > 5:
            recommendations.append("Multiple performance anomalies detected - investigate system stability")
            
        # Model-specific recommendations
        response_time_anomalies = [a for a in anomalies if a["type"] == "response_time_spike"]
        if len(response_time_anomalies) > 3:
            recommendations.append("Consider model optimization or caching for inference speed")
            
        return recommendations
```

---

## 6. Real-time Monitoring WebSocket Service

### 6.1 Real-time Metrics Broadcaster
```python
# app/monitoring/realtime_broadcaster.py
import asyncio
import json
from typing import Dict, Set
from fastapi import WebSocket
from datetime import datetime

class RealTimeMetricsBroadcaster:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.metrics_cache: Dict[str, any] = {}
        
    async def connect(self, websocket: WebSocket):
        """Accept new WebSocket connection"""
        await websocket.accept()
        self.active_connections.add(websocket)
        
        # Send initial metrics
        await self.send_metrics_to_client(websocket)
        
    async def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        self.active_connections.discard(websocket)
        
    async def broadcast_metrics(self, metrics: Dict[str, any]):
        """Broadcast metrics to all connected clients"""
        self.metrics_cache = metrics
        
        if self.active_connections:
            message = json.dumps({
                "timestamp": datetime.utcnow().isoformat(),
                "metrics": metrics
            })
            
            # Remove disconnected clients
            disconnected = set()
            
            for connection in self.active_connections:
                try:
                    await connection.send_text(message)
                except Exception:
                    disconnected.add(connection)
                    
            # Clean up disconnected clients
            self.active_connections -= disconnected
            
    async def send_metrics_to_client(self, websocket: WebSocket):
        """Send current metrics to specific client"""
        if self.metrics_cache:
            message = json.dumps({
                "timestamp": datetime.utcnow().isoformat(),
                "metrics": self.metrics_cache
            })
            
            try:
                await websocket.send_text(message)
            except Exception:
                pass
                
    async def start_metrics_collection(self):
        """Start background task for metrics collection"""
        while True:
            try:
                # Collect current metrics from all services
                current_metrics = await self.collect_current_metrics()
                
                # Broadcast to all connected clients
                await self.broadcast_metrics(current_metrics)
                
                # Wait 5 seconds before next collection
                await asyncio.sleep(5)
                
            except Exception as e:
                print(f"Error in metrics collection: {e}")
                await asyncio.sleep(5)
                
    async def collect_current_metrics(self) -> Dict[str, any]:
        """Collect current metrics from Prometheus"""
        # This would integrate with your Prometheus client
        # For now, returning mock data structure
        
        return {
            "overall": {
                "total_requests": 12543,
                "active_users": 234,
                "response_time_avg": 156.7,
                "response_time_95th": 487.2,
                "error_rate": 0.023,
                "success_rate": 97.7
            },
            "face_recognition": {
                "request_rate": 45.2,
                "response_time": 234.5,
                "error_rate": 0.012,
                "accuracy": 99.6,
                "active_models": 3
            },
            "face_detection": {
                "request_rate": 67.8,
                "response_time": 123.4,
                "error_rate": 0.008,
                "faces_detected_per_min": 145,
                "avg_faces_per_image": 2.1
            },
            "antispoofing": {
                "request_rate": 23.1,
                "response_time": 678.9,
                "error_rate": 0.034,
                "liveness_accuracy": 98.2,
                "spoof_detection_rate": 96.7
            },
            "deepfake_detection": {
                "request_rate": 12.4,
                "response_time": 1234.5,
                "error_rate": 0.056,
                "deepfake_detection_rate": 94.3,
                "processing_queue_size": 5
            },
            "age_gender": {
                "request_rate": 34.7,
                "response_time": 167.8,
                "error_rate": 0.019,
                "age_accuracy": 89.4,
                "gender_accuracy": 96.1
            }
        }

# Global broadcaster instance
metrics_broadcaster = RealTimeMetricsBroadcaster()
```

### 6.2 WebSocket Endpoint
```python
# app/api/v1/routes/monitoring.py
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from ...monitoring.realtime_broadcaster import metrics_broadcaster

router = APIRouter()

@router.websocket("/analytics/realtime")
async def websocket_analytics(websocket: WebSocket):
    """Real-time analytics WebSocket endpoint"""
    await metrics_broadcaster.connect(websocket)
    
    try:
        while True:
            # Keep connection alive and handle client messages
            await websocket.receive_text()
            
    except WebSocketDisconnect:
        await metrics_broadcaster.disconnect(websocket)
```

Monitoring และ Analytics Dashboard นี้ให้ระบบการติดตามและวิเคราะห์ที่ครอบคลุมสำหรับ Backend AI Services ทั้ง 5 บริการ รวมถึง real-time monitoring, alerting, performance analysis และ business insights ที่จำเป็นสำหรับการจัดการระบบ AI ในระดับ production
