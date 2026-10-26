# Clinical Shorthand-to-FHIR Compiler - Web Workstation Launcher
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host "  Clinical Shorthand-to-FHIR Compiler: Enterprise Workstation" -ForegroundColor Cyan
Write-Host "  Runtime: umc301 Conda Environment" -ForegroundColor Cyan
Write-Host "  Server: http://localhost:8000" -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Cyan

$python = "C:\Users\aksha\miniconda3\envs\umc301\python.exe"
if (-not (Test-Path $python)) {
    $python = "C:\Users\aksha\AppData\Local\Programs\Python\Python312\python.exe"
}
if (-not (Test-Path $python)) {
    $python = "python"
}

Start-Process "http://localhost:8000"
& $python server.py
