@echo off
if exist "C:\Users\aksha\miniconda3\envs\umc301\Scripts\streamlit.exe" (
    echo Starting app using umc301 Conda Environment...
    "C:\Users\aksha\miniconda3\envs\umc301\Scripts\streamlit.exe" run app.py
) else (
    echo Starting app using active environment...
    streamlit run app.py
)
