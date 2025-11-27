@echo off
title Deepfake Detector - Backend
cd /d C:\Users\suman\Desktop\deepfake-detector\backend
echo Starting Backend...

"C:\Users\suman\AppData\Local\Programs\Python\Python311\python.exe" -m uvicorn app:app --reload --host 127.0.0.1 --port 8000

pause
