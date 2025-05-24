# Test CCTV Face Recognition Service

Write-Host "🎥 Testing CCTV Face Recognition Service" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Base URL
$baseUrl = "http://localhost:8003"

# 1. Check health
Write-Host "`n1️⃣ Checking service health..." -ForegroundColor Yellow
$health = Invoke-RestMethod -Uri "$baseUrl/health" -Method Get
Write-Host "✅ Health Status: $($health.status)" -ForegroundColor Green
Write-Host "   Redis Connected: $($health.redis_connected)"
Write-Host "   Active Streams: $($health.active_streams)"

# 2. Start monitoring a test stream
Write-Host "`n2️⃣ Starting CCTV monitoring..." -ForegroundColor Yellow

# Test with sample RTSP stream (you can replace with actual stream)
$formData = @{
    stream_url = "rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mp4"  # Public test stream
    stream_type = "rtsp"
    detection_interval = 2000  # 2 seconds
    min_detection_confidence = 0.7
    notify_on_match = "true"
    save_detections = "true"
    alert_threshold = 0.8
}

try {
    $response = Invoke-RestMethod -Uri "$baseUrl/cctv/start-monitoring" -Method Post -Form $formData
    Write-Host "✅ Monitoring started successfully!" -ForegroundColor Green
    Write-Host "   Monitoring ID: $($response.monitoring_id)" -ForegroundColor Cyan
    Write-Host "   Status: $($response.status)"
    $monitoringId = $response.monitoring_id
} catch {
    Write-Host "❌ Failed to start monitoring: $_" -ForegroundColor Red
    exit 1
}

# 3. Check monitoring status
Write-Host "`n3️⃣ Checking monitoring status..." -ForegroundColor Yellow
Start-Sleep -Seconds 2
$status = Invoke-RestMethod -Uri "$baseUrl/cctv/monitoring/$monitoringId" -Method Get
Write-Host "✅ Monitoring Status:" -ForegroundColor Green
Write-Host "   Status: $($status.status)"
Write-Host "   Stream URL: $($status.stream_url)"
Write-Host "   Created: $($status.created_at)"

# 4. Get active sessions
Write-Host "`n4️⃣ Getting active sessions..." -ForegroundColor Yellow
$sessions = Invoke-RestMethod -Uri "$baseUrl/cctv/active-sessions" -Method Get
Write-Host "✅ Active Sessions: $($sessions.total_sessions)" -ForegroundColor Green
foreach ($session in $sessions.sessions) {
    Write-Host "   - ID: $($session.monitoring_id.Substring(0,8))... | Stream: $($session.stream_url)"
}

# 5. Wait for some detections
Write-Host "`n5️⃣ Waiting for face detections (10 seconds)..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# 6. Get detection results
Write-Host "`n6️⃣ Getting detection results..." -ForegroundColor Yellow
$results = Invoke-RestMethod -Uri "$baseUrl/cctv/monitoring/$monitoringId/results" -Method Get
Write-Host "✅ Detection Results:" -ForegroundColor Green
Write-Host "   Total Detections: $($results.total_detections)"
if ($results.results.Count -gt 0) {
    foreach ($detection in $results.results[0..2]) {  # Show first 3
        Write-Host "   - Time: $($detection.timestamp) | Known: $($detection.is_known_person) | Confidence: $($detection.confidence)"
    }
}

# 7. Stop monitoring
Write-Host "`n7️⃣ Stopping monitoring..." -ForegroundColor Yellow
$stop = Invoke-RestMethod -Uri "$baseUrl/cctv/monitoring/$monitoringId" -Method Delete
Write-Host "✅ Monitoring stopped: $($stop.message)" -ForegroundColor Green

# 8. Test with webcam (optional)
Write-Host "`n8️⃣ Test with webcam? (y/n)" -ForegroundColor Yellow
$useWebcam = Read-Host
if ($useWebcam -eq 'y') {
    Write-Host "Starting webcam monitoring..." -ForegroundColor Cyan
    
    $webcamData = @{
        stream_url = "0"  # Default webcam
        stream_type = "webcam"
        detection_interval = 1000
        min_detection_confidence = 0.6
        notify_on_match = "true"
        save_detections = "true"
        alert_threshold = 0.75
    }
    
    try {
        $webcamResponse = Invoke-RestMethod -Uri "$baseUrl/cctv/start-monitoring" -Method Post -Form $webcamData
        Write-Host "✅ Webcam monitoring started!" -ForegroundColor Green
        Write-Host "   Monitoring ID: $($webcamResponse.monitoring_id)"
        Write-Host "   Press Enter to stop webcam monitoring..."
        Read-Host
        
        # Stop webcam monitoring
        Invoke-RestMethod -Uri "$baseUrl/cctv/monitoring/$($webcamResponse.monitoring_id)" -Method Delete
        Write-Host "✅ Webcam monitoring stopped" -ForegroundColor Green
    } catch {
        Write-Host "❌ Webcam monitoring failed: $_" -ForegroundColor Red
    }
}

Write-Host "`n✨ CCTV Service test completed!" -ForegroundColor Green
Write-Host "📊 Dashboard available at: http://localhost:8003/cctv/dashboard" -ForegroundColor Cyan
