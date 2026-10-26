@echo off
echo ==================================================================
echo   Clinical Shorthand-to-FHIR Compiler: Enterprise Workstation
echo   Runtime: umc301 Conda Environment
echo   Server: http://localhost:8000
echo ==================================================================

start "" "http://localhost:8000"
if exist "C:\Users\aksha\miniconda3\envs\umc301\python.exe" (
    "C:\Users\aksha\miniconda3\envs\umc301\python.exe" server.py
) else if exist "C:\Users\aksha\AppData\Local\Programs\Python\Python312\python.exe" (
    "C:\Users\aksha\AppData\Local\Programs\Python\Python312\python.exe" server.py
) else (
    python server.py
)
