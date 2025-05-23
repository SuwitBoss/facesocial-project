# This script will clean up unnecessary files from the workspace.
# It will NOT delete any source code, Dockerfiles, requirements.txt, or model files.
# It will only delete test images, test result files, and loose .json/.txt files that are not part of a service.

$deleteFiles = @(
    'direct-chat.json',
    'group-chat.json',
    'login2.json',
    'result-with-faces.jpg',
    'result.json',
    'test-face.png',
    'test-gender-age.ps1',
    'test-image.txt',
    'test.txt',
    'user.json',
    'user2.json',
    'user3.json',
    'face-test-new.jpg',
    'image.png'
)

foreach ($file in $deleteFiles) {
    if (Test-Path $file) {
        Write-Host "Deleting $file" -ForegroundColor Yellow
        Remove-Item $file -Force
    }
}

Write-Host "Cleanup complete!" -ForegroundColor Green
