@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    py -3 -m venv --system-site-packages .venv
    if errorlevel 1 (
        echo Khong tao duoc .venv bang py -3. Hay cai Python 3 va thu lai.
        exit /b 1
    )
)

set "PYTHON=%CD%\.venv\Scripts\python.exe"
"%PYTHON%" -c "import torch" >nul 2>&1
if errorlevel 1 (
    echo Khong tim thay PyTorch trong Python he thong.
    echo .venv dung system-site-packages va script khong tu cai hoac cap nhat PyTorch/CUDA.
    echo Hay cai PyTorch phu hop voi may bang trang chinh thuc cua PyTorch roi chay lai.
    exit /b 1
)

"%PYTHON%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo Cai dat package phu that bai. Kiem tra ket noi va requirements.txt.
    exit /b 1
)

"%PYTHON%" app.py
if errorlevel 1 (
    echo Ung dung dung voi loi. Xem thong bao o phia tren de xu ly.
    exit /b 1
)
endlocal
