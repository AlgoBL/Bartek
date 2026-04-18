@echo off
if not exist .\\venv (
    echo [ERROR] Folder 'venv' nie istnieje!
    echo Zainstaluj srodowisko wpisujac: python -m venv venv
    echo Nastepnie: .\\venv\\Scripts\\pip install -r requirements.txt
    pause
    exit /b
)

echo Starting Intelligent Barbell App (v9.6 - Audit Stabilized) in Browser...
.\\venv\\Scripts\\streamlit.exe run app.py ^
    --server.headless false ^
    --browser.gatherUsageStats false ^
    --logger.level warning ^
    --server.fileWatcherType none ^
    --runner.fastReruns true
pause
