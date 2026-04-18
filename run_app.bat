@echo off
echo Starting Intelligent Barbell App (v9.5 - Optimized) in Browser...
.\\venv\\Scripts\\streamlit.exe run app.py ^
    --server.headless false ^
    --browser.gatherUsageStats false ^
    --logger.level warning ^
    --server.fileWatcherType none ^
    --runner.fastReruns true
pause
