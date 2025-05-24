from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
import os
import asyncio
from datetime import datetime
import logging
import json
import redis.asyncio as redis
from typing import Dict, List, Optional
import uuid

from services.stream_processor import StreamProcessor
from services.alert_manager import AlertManager
from models.monitoring import MonitoringSession, MonitoringStatus, DetectionResult
from utils.auth import verify_token

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="CCTV Face Recognition Service",
    description="Real-time face recognition for CCTV and video streams",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
monitoring_sessions: Dict[str, MonitoringSession] = {}
stream_processors: Dict[str, StreamProcessor] = {}
redis_client = None
alert_manager = None
active_websockets: List[WebSocket] = []

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global redis_client, alert_manager
    try:
        redis_client = await redis.Redis(
            host=os.getenv('REDIS_HOST', 'redis'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            decode_responses=True
        )
        await redis_client.ping()
        logger.info("✅ Redis connected")
        alert_manager = AlertManager(redis_client)
        logger.info("✅ Alert Manager initialized")
    except Exception as e:
        logger.error(f"❌ Startup error: {str(e)}")

@app.on_event("shutdown")
async def shutdown_event():
    for session_id in list(monitoring_sessions.keys()):
        await stop_monitoring_internal(session_id)
    if redis_client:
        await redis_client.close()

@app.get("/")
def read_root():
    return {
        "service": "CCTV Face Recognition Service",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "active_sessions": len(monitoring_sessions),
        "endpoints": [
            "GET /health",
            "POST /cctv/start-monitoring",
            "GET /cctv/monitoring/{monitoring_id}",
            "DELETE /cctv/monitoring/{monitoring_id}",
            "GET /cctv/monitoring/{monitoring_id}/results",
            "WS /ws/{monitoring_id}"
        ]
    }

@app.get("/health")
def health_check():
    return {
        "status": "OK",
        "service": "CCTV Face Recognition Service",
        "timestamp": datetime.now().isoformat(),
        "redis_connected": redis_client is not None,
        "active_streams": len(monitoring_sessions),
        "active_websockets": len(active_websockets)
    }

@app.post("/cctv/start-monitoring")
async def start_monitoring(
    stream_url: str = Form(...),
    stream_type: str = Form(default="rtsp"),
    detection_interval: int = Form(default=1000),
    min_detection_confidence: float = Form(default=0.7),
    notify_on_match: bool = Form(default=True),
    save_detections: bool = Form(default=True),
    alert_threshold: float = Form(default=0.8),
    mock_mode: bool = Form(default=False)  # เพิ่ม mock_mode
):
    try:
        monitoring_id = str(uuid.uuid4())
        session = MonitoringSession(
            id=monitoring_id,
            stream_url=stream_url,
            stream_type=stream_type,
            status=MonitoringStatus.STARTING,
            created_at=datetime.utcnow(),
            config={
                "detection_interval": detection_interval,
                "min_detection_confidence": min_detection_confidence,
                "notify_on_match": notify_on_match,
                "save_detections": save_detections,
                "alert_threshold": alert_threshold,
                "mock_mode": mock_mode
            }
        )
        monitoring_sessions[monitoring_id] = session
        processor = StreamProcessor(
            monitoring_id=monitoring_id,
            stream_url=stream_url,
            stream_type=stream_type,
            redis_client=redis_client,
            alert_manager=alert_manager,
            config=session.config
        )
        stream_processors[monitoring_id] = processor
        asyncio.create_task(processor.start_processing())
        session.status = MonitoringStatus.ACTIVE
        await redis_client.setex(
            f"monitoring:{monitoring_id}",
            3600,
            json.dumps(session.to_dict())
        )
        return {
            "monitoring_id": monitoring_id,
            "status": "active",
            "message": "Monitoring started successfully",
            "stream_url": stream_url,
            "config": session.config
        }
    except Exception as e:
        logger.error(f"Failed to start monitoring: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cctv/monitoring/{monitoring_id}")
async def get_monitoring_status(monitoring_id: str):
    if monitoring_id not in monitoring_sessions:
        cached = await redis_client.get(f"monitoring:{monitoring_id}")
        if cached:
            return json.loads(cached)
        raise HTTPException(status_code=404, detail="Monitoring session not found")
    session = monitoring_sessions[monitoring_id]
    processor = stream_processors.get(monitoring_id)
    return {
        "monitoring_id": monitoring_id,
        "status": session.status.value,
        "created_at": session.created_at.isoformat(),
        "stream_url": session.stream_url,
        "config": session.config,
        "statistics": processor.get_statistics() if processor else {},
        "last_detection": session.last_detection_at.isoformat() if session.last_detection_at else None
    }

@app.delete("/cctv/monitoring/{monitoring_id}")
async def stop_monitoring(monitoring_id: str):
    if monitoring_id not in monitoring_sessions:
        raise HTTPException(status_code=404, detail="Monitoring session not found")
    await stop_monitoring_internal(monitoring_id)
    return {
        "monitoring_id": monitoring_id,
        "status": "stopped",
        "message": "Monitoring stopped successfully"
    }

async def stop_monitoring_internal(monitoring_id: str):
    try:
        if monitoring_id in stream_processors:
            processor = stream_processors[monitoring_id]
            await processor.stop_processing()
            del stream_processors[monitoring_id]
        if monitoring_id in monitoring_sessions:
            session = monitoring_sessions[monitoring_id]
            session.status = MonitoringStatus.STOPPED
            session.stopped_at = datetime.utcnow()
            await redis_client.setex(
                f"monitoring:{monitoring_id}",
                300,
                json.dumps(session.to_dict())
            )
            del monitoring_sessions[monitoring_id]
    except Exception as e:
        logger.error(f"Error stopping monitoring {monitoring_id}: {str(e)}")

@app.get("/cctv/monitoring/{monitoring_id}/results")
async def get_monitoring_results(
    monitoring_id: str,
    from_time: Optional[str] = None,
    to_time: Optional[str] = None,
    page: int = 1,
    limit: int = 20
):
    try:
        results_key = f"results:{monitoring_id}"
        results = await redis_client.lrange(results_key, 0, -1)
        detections = []
        for result_json in results:
            detection = json.loads(result_json)
            detections.append(detection)
        detections.sort(key=lambda x: x['timestamp'], reverse=True)
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit
        paginated_results = detections[start_idx:end_idx]
        return {
            "monitoring_id": monitoring_id,
            "total_detections": len(detections),
            "page": page,
            "limit": limit,
            "results": paginated_results,
            "has_more": end_idx < len(detections)
        }
    except Exception as e:
        logger.error(f"Failed to get results: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/{monitoring_id}")
async def websocket_endpoint(websocket: WebSocket, monitoring_id: str):
    await websocket.accept()
    active_websockets.append(websocket)
    try:
        pubsub = redis_client.pubsub()
        await pubsub.subscribe(f"detections:{monitoring_id}")
        await websocket.send_json({
            "type": "connected",
            "monitoring_id": monitoring_id,
            "timestamp": datetime.utcnow().isoformat()
        })
        async for message in pubsub.listen():
            if message['type'] == 'message':
                data = json.loads(message['data'])
                await websocket.send_json(data)
    except WebSocketDisconnect:
        active_websockets.remove(websocket)
        await pubsub.unsubscribe(f"detections:{monitoring_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        if websocket in active_websockets:
            active_websockets.remove(websocket)

@app.get("/cctv/active-sessions")
async def get_active_sessions():
    sessions = []
    for session_id, session in monitoring_sessions.items():
        processor = stream_processors.get(session_id)
        sessions.append({
            "monitoring_id": session_id,
            "stream_url": session.stream_url,
            "status": session.status.value,
            "created_at": session.created_at.isoformat(),
            "statistics": processor.get_statistics() if processor else {}
        })
    return {
        "total_sessions": len(sessions),
        "sessions": sessions
    }

@app.get("/cctv/dashboard")
async def dashboard():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>CCTV Monitoring Dashboard</title>
        <style>
            body { font-family: Arial; margin: 20px; background: #f0f0f0; }
            .container { max-width: 1200px; margin: auto; background: white; padding: 20px; border-radius: 10px; }
            .session { border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px; }
            .active { border-color: #4CAF50; background: #f1f8f4; }
            .detection { background: #fffbf0; padding: 10px; margin: 5px 0; border-radius: 3px; }
            button { padding: 10px 15px; margin: 5px; cursor: pointer; border: none; border-radius: 5px; }
            .start { background: #4CAF50; color: white; }
            .stop { background: #f44336; color: white; }
            #log { height: 300px; overflow-y: auto; background: #f5f5f5; padding: 10px; margin-top: 20px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎥 CCTV Face Recognition Dashboard</h1>
            <div>
                <h3>Start New Monitoring</h3>
                <input type="text" id="streamUrl" placeholder="Stream URL (rtsp://...)" style="width: 300px; padding: 8px;">
                <select id="streamType" style="padding: 8px;">
                    <option value="rtsp">RTSP</option>
                    <option value="http">HTTP</option>
                    <option value="webcam">Webcam</option>
                </select>
                <button class="start" onclick="startMonitoring()">Start Monitoring</button>
            </div>
            <div id="sessions">
                <h3>Active Sessions</h3>
                <div id="sessionList"></div>
            </div>
            <div>
                <h3>Real-time Detections</h3>
                <div id="log"></div>
            </div>
        </div>
        <script>
            let activeSessions = {};
            let activeWebSockets = {};
            async function startMonitoring() {
                const streamUrl = document.getElementById('streamUrl').value;
                const streamType = document.getElementById('streamType').value;
                if (!streamUrl) {
                    alert('Please enter stream URL');
                    return;
                }
                const formData = new FormData();
                formData.append('stream_url', streamUrl);
                formData.append('stream_type', streamType);
                try {
                    const response = await fetch('/cctv/start-monitoring', {
                        method: 'POST',
                        body: formData
                    });
                    const data = await response.json();
                    if (data.monitoring_id) {
                        log(`✅ Started monitoring: ${data.monitoring_id}`);
                        connectWebSocket(data.monitoring_id);
                        refreshSessions();
                    }
                } catch (error) {
                    log(`❌ Error: ${error.message}`);
                }
            }
            function connectWebSocket(monitoringId) {
                const ws = new WebSocket(`ws://${location.hostname}:${location.port || 8003}/ws/${monitoringId}`);
                ws.onopen = () => {
                    log(`🔌 WebSocket connected for ${monitoringId}`);
                };
                ws.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    if (data.type === 'detection') {
                        log(`👤 Detection: ${data.person_name} (${(data.confidence * 100).toFixed(1)}%)`);
                    }
                };
                ws.onclose = () => {
                    log(`🔌 WebSocket disconnected for ${monitoringId}`);
                };
                activeWebSockets[monitoringId] = ws;
            }
            async function stopMonitoring(monitoringId) {
                try {
                    await fetch(`/cctv/monitoring/${monitoringId}`, {
                        method: 'DELETE'
                    });
                    if (activeWebSockets[monitoringId]) {
                        activeWebSockets[monitoringId].close();
                        delete activeWebSockets[monitoringId];
                    }
                    log(`⏹️ Stopped monitoring: ${monitoringId}`);
                    refreshSessions();
                } catch (error) {
                    log(`❌ Error: ${error.message}`);
                }
            }
            async function refreshSessions() {
                try {
                    const response = await fetch('/cctv/active-sessions');
                    const data = await response.json();
                    const sessionList = document.getElementById('sessionList');
                    sessionList.innerHTML = '';
                    data.sessions.forEach(session => {
                        const div = document.createElement('div');
                        div.className = 'session active';
                        div.innerHTML = `
                            <strong>Session:</strong> ${session.monitoring_id}<br>
                            <strong>Stream:</strong> ${session.stream_url}<br>
                            <strong>Status:</strong> ${session.status}<br>
                            <strong>Detections:</strong> ${(session.statistics && session.statistics.total_detections) ? session.statistics.total_detections : 0}<br>
                            <button class="stop" onclick="stopMonitoring('${session.monitoring_id}')">Stop</button>
                        `;
                        sessionList.appendChild(div);
                    });
                } catch (error) {
                    console.error('Error refreshing sessions:', error);
                }
            }
            function log(message) {
                const logEl = document.getElementById('log');
                const time = new Date().toLocaleTimeString();
                logEl.innerHTML = `[${time}] ${message}<br>` + logEl.innerHTML;
            }
            setInterval(refreshSessions, 5000);
            refreshSessions();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8003))
    uvicorn.run(app, host="0.0.0.0", port=port)
