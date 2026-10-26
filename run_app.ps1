# Clinical Shorthand-to-FHIR Compiler - App Launcher (umc301 Conda Environment)
$conda_streamlit = "C:\Users\aksha\miniconda3\envs\umc301\Scripts\streamlit.exe"
if (Test-Path $conda_streamlit) {
    Write-Host "Starting app using umc301 Conda Environment..." -ForegroundColor Green
    & $conda_streamlit run app.py
} else {
    Write-Host "Starting app using active environment..." -ForegroundColor Yellow
    streamlit run app.py
}

