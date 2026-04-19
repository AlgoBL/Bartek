@echo off
echo [GitHub Update] Startuje proces aktualizacji...
cd /d "%~dp0"

echo [1/3] Dodawanie plikow (git add)...
git add .

echo [2/3] Tworzenie commita (git commit)...
git commit -m "Stabilize data pipeline and fix asyncio deadlocks and performance"

echo [3/3] Wysylanie do repozytorium (git push)...
git push

echo.
echo [GOTOWE] Twoj GitHub zostal zaktualizowany!
pause
