@echo off
echo Starting Celery Worker for Intelligent Barbell...
echo Note: Redis must be running for this to work.
.\venv\Scripts\celery.exe -A tasks worker --loglevel=info --pool=solo
pause
