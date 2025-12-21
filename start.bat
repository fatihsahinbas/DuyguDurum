@echo off
echo ========================================
echo Gercek Zamanli Duygu Tanima Sistemi
echo ========================================
echo.

REM Virtual environment kontrol
if not exist "venv\" (
    echo Virtual environment bulunamadi. Olusturuluyor...
    python -m venv venv
    echo Virtual environment olusturuldu.
    echo.
)

REM Virtual environment'i aktif et
echo Virtual environment aktif ediliyor...
call venv\Scripts\activate.bat

REM Temel bagimliliklar
echo Temel bagimliliklar kuruluyor...
pip install flask flask-socketio opencv-python pillow python-socketio eventlet

echo.
echo ========================================
echo Sistem baslatiliyor...
echo ========================================
echo.
echo Tarayicinizda http://localhost:5000 adresini acin
echo Durdurmak icin Ctrl+C basin
echo.

REM Uygulamayi baslat
python app.py

pause