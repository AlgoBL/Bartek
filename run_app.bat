@echo off
echo Starting Intelligent Barbell App (v9.5 - Optimized)...
.\\venv\\Scripts\\streamlit.exe run app.py ^
    --server.headless true ^
    --browser.gatherUsageStats false ^
    --logger.level warning ^
    --server.fileWatcherType none ^
    --runner.fastReruns true
pause
