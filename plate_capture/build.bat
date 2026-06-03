@echo off
:: build.bat — Double-click this to build PlateDetector.exe
:: Make sure your .venv is activated before running, OR run from inside .venv

echo ============================================================
echo  PlateDetector EXE Builder
echo ============================================================
echo.

:: Step 1: Check pyinstaller is installed
python -c "import PyInstaller" 2>nul
if %errorlevel% neq 0 (
    echo [INFO] Installing PyInstaller...
    pip install pyinstaller==6.19.0
)

:: Step 2: Download models if not already done
echo [INFO] Checking PaddleOCR models...
python download_models.py
if %errorlevel% neq 0 (
    echo [ERROR] Model download failed. Fix errors above then retry.
    pause
    exit /b 1
)

:: Step 3: Clean old build
echo.
echo [INFO] Cleaning old build folders...
if exist build rmdir /s /q build
if exist dist  rmdir /s /q dist

:: Step 4: Build exe
echo.
echo [INFO] Building exe (this takes 3-10 minutes)...
pyinstaller main.spec

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Build FAILED. Check errors above.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  BUILD SUCCESSFUL!
echo  Your exe is in:  dist\PlateDetector\PlateDetector.exe
echo  Distribute the entire dist\PlateDetector\ FOLDER (not just the exe)
echo ============================================================
pause