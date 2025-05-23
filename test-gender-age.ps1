param(
    [string]$ImagePath = "face-test-new.jpg"
)

Write-Host "🧪 Testing Gender & Age Detection Service" -ForegroundColor Cyan

# Test 1: Health Check
Write-Host "`n1. Testing Health Check..." -ForegroundColor Yellow
try {
    $health = Invoke-RestMethod -Uri "http://localhost:8011/health"
    Write-Host "✅ Health Check: $($health.status)" -ForegroundColor Green
    Write-Host "   Model Status: $($health.models_status.gender_age_detector)" -ForegroundColor Gray
} catch {
    Write-Host "❌ Health Check Failed: $($_.Exception.Message)" -ForegroundColor Red
}

# Test 2: Health Check via Kong
Write-Host "`n2. Testing Health Check via Kong..." -ForegroundColor Yellow
try {
    $healthKong = Invoke-RestMethod -Uri "http://localhost:8002/gender-age/health"
    Write-Host "✅ Kong Health Check: $($healthKong.status)" -ForegroundColor Green
} catch {
    Write-Host "❌ Kong Health Check Failed: $($_.Exception.Message)" -ForegroundColor Red
}

# Test 3: Service Info
Write-Host "`n3. Testing Service Info..." -ForegroundColor Yellow
try {
    $info = Invoke-RestMethod -Uri "http://localhost:8001/"
    Write-Host "✅ Service: $($info.service)" -ForegroundColor Green
    Write-Host "   Version: $($info.version)" -ForegroundColor Gray
    Write-Host "   Models Loaded: $($info.models_loaded.gender_age_detector)" -ForegroundColor Gray
} catch {
    Write-Host "❌ Service Info Failed: $($_.Exception.Message)" -ForegroundColor Red
}

# Test 4: Stats Endpoint
Write-Host "`n4. Testing Stats Endpoint..." -ForegroundColor Yellow
try {
    $stats = Invoke-RestMethod -Uri "http://localhost:8011/demographics/stats"
    Write-Host "✅ Stats Retrieved" -ForegroundColor Green
    Write-Host "   Model Type: $($stats.model_info.model_type)" -ForegroundColor Gray
    Write-Host "   Input Size: $($stats.model_info.input_size)" -ForegroundColor Gray
    Write-Host "   Max Batch Size: $($stats.max_batch_size)" -ForegroundColor Gray
} catch {
    Write-Host "❌ Stats Failed: $($_.Exception.Message)" -ForegroundColor Red
}

# Test 4.1: Stats Endpoint via Kong
Write-Host "`n4.1. Testing Stats Endpoint via Kong..." -ForegroundColor Yellow
try {
    $statsKong = Invoke-RestMethod -Uri "http://localhost:8002/gender-age/demographics/stats"
    Write-Host "✅ Kong Stats Retrieved" -ForegroundColor Green
    Write-Host "   Model Type: $($statsKong.model_info.model_type)" -ForegroundColor Gray
    Write-Host "   Input Size: $($statsKong.model_info.input_size)" -ForegroundColor Gray
    Write-Host "   Max Batch Size: $($statsKong.max_batch_size)" -ForegroundColor Gray
} catch {
    Write-Host "❌ Kong Stats Failed: $($_.Exception.Message)" -ForegroundColor Red
}

# Test 5: Demographics Analysis (if image file exists)
$testImage = $ImagePath
if (Test-Path $testImage) {
    Write-Host "`n5. Testing Demographics Analysis (Direct)..." -ForegroundColor Yellow
    try {
        $uri = "http://localhost:8001/demographics/analyze"
        $form = @{
            file = Get-Item $testImage
            return_face_info = "true"
        }
        $response = pwsh -Command "Invoke-RestMethod -Uri '$uri' -Method Post -Form @{'file'=(Get-Item '$testImage');'return_face_info'='true'}"
        Write-Host "[DEBUG] Full response: $($response | ConvertTo-Json -Depth 5)" -ForegroundColor DarkGray
        Write-Host "✅ Demographics Analysis Success" -ForegroundColor Green
        if ($response.success) {
            $analysis = $response.analysis
            Write-Host "   Gender: $($analysis.gender.prediction) (Confidence: $([math]::Round($analysis.gender.confidence * 100, 1))%)" -ForegroundColor Gray
            Write-Host "   Age: $($analysis.age.prediction) years (Range: $($analysis.age.range.min)-$($analysis.age.range.max))" -ForegroundColor Gray
            Write-Host "   [RESULT] Gender: $($analysis.gender.prediction), Age: $($analysis.age.prediction) (Value: $($analysis.age.value)), Range: $($analysis.age.range.min)-$($analysis.age.range.max)" -ForegroundColor Magenta
        }
    } catch {
        Write-Host "❌ Demographics Analysis Failed: $($_.Exception.Message)" -ForegroundColor Red
    }
    Write-Host "`n6. Testing Demographics Analysis (via Kong)..." -ForegroundColor Yellow
    try {
        $uri = "http://localhost:8002/gender-age/analyze"
        $response = pwsh -Command "Invoke-RestMethod -Uri '$uri' -Method Post -Form @{'file'=(Get-Item '$testImage');'return_face_info'='true'}"
        Write-Host "[DEBUG] Full response: $($response | ConvertTo-Json -Depth 5)" -ForegroundColor DarkGray
        Write-Host "✅ Kong Demographics Analysis Success" -ForegroundColor Green
        if ($response.analysis) {
            $analysis = $response.analysis
            Write-Host "   Gender: $($analysis.gender.prediction) (Confidence: $([math]::Round($analysis.gender.confidence * 100, 1))%)" -ForegroundColor Gray
            Write-Host "   Age: $($analysis.age.prediction) years" -ForegroundColor Gray
            Write-Host "   [RESULT] Gender: $($analysis.gender.prediction), Age: $($analysis.age.prediction) (Value: $($analysis.age.value)), Range: $($analysis.age.range.min)-$($analysis.age.range.max)" -ForegroundColor Magenta
        }
    } catch {
        Write-Host "❌ Kong Demographics Analysis Failed: $($_.Exception.Message)" -ForegroundColor Red
    }
} else {
    Write-Host "`n5-6. Skipping image analysis tests (no test image found: $testImage)" -ForegroundColor Yellow
}

Write-Host "`n🎯 Gender & Age Detection Service Test Complete!" -ForegroundColor Cyan
Write-Host "Run with: .\test-gender-age.ps1" -ForegroundColor Gray
