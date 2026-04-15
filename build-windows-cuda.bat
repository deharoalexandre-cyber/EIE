@echo off
REM ============================================
REM EIE — Build with CUDA on Windows
REM Requires: Visual Studio 2022 Build Tools, CUDA Toolkit, CMake
REM Run from: Developer PowerShell for VS 2022
REM ============================================

echo.
echo  EIE - Elyne Inference Engine
echo  Building with CUDA backend (Windows)
echo  ============================================
echo.

REM Auto-detect CUDA Toolkit
if defined CUDA_PATH (
    echo [OK] CUDA_PATH: %CUDA_PATH%
) else (
    for %%V in (13.2 12.8 12.6 12.4) do (
        if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v%%V" (
            set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v%%V"
            goto :cuda_found
        )
    )
    echo [ERROR] CUDA Toolkit not found.
    echo         Install from https://developer.nvidia.com/cuda-downloads
    echo         Or set CUDA_PATH manually before running this script.
    exit /b 1
)
:cuda_found
echo [OK] Using CUDA: %CUDA_PATH%

set "PATH=%CUDA_PATH%\bin;%PATH%"
set "CudaToolkitDir=%CUDA_PATH%"

REM Check for nvcc
where nvcc >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] nvcc not found in PATH. Is CUDA Toolkit installed correctly?
    exit /b 1
)
echo [OK] nvcc found

REM Check for cl.exe (MSVC compiler)
where cl.exe >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] cl.exe not found. Run this script from Developer PowerShell for VS 2022.
    echo         Open: Start ^> Developer PowerShell for VS 2022
    exit /b 1
)
echo [OK] MSVC compiler found

REM Init submodules
echo.
echo [..] Initializing submodules...
git submodule update --init 2>nul

REM Fix ASM compiler issue (MASM not always available in Build Tools)
if exist "llama.cpp\ggml\CMakeLists.txt" (
    powershell -Command "(Get-Content 'llama.cpp\ggml\CMakeLists.txt') -replace 'project\(ggml C CXX ASM\)', 'project(ggml C CXX)' | Set-Content 'llama.cpp\ggml\CMakeLists.txt'"
    echo [OK] Applied ASM fix for MSVC Build Tools
)

REM Clean previous build
if exist build (
    echo [..] Cleaning previous build...
    rmdir /s /q build 2>nul
)

REM Configure
echo.
echo [..] Configuring with CMake...
cmake -B build -G "Visual Studio 17 2022" ^
    -DGGML_CUDA=ON ^
    -DCUDAToolkit_ROOT="%CUDA_PATH%" ^
    -DCMAKE_CUDA_COMPILER="%CUDA_PATH%\bin\nvcc.exe"

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] CMake configuration failed.
    echo.
    echo Common fixes:
    echo   1. Make sure you run from Developer PowerShell for VS 2022
    echo   2. Install Desktop development with C++ workload in VS Installer
    echo   3. Verify CUDA Toolkit: nvcc --version
    exit /b 1
)

REM Build
echo.
echo [..] Building (this may take 15-20 minutes with CUDA kernels)...
cmake --build build --config Release

if %errorlevel% neq 0 (
    echo [ERROR] Build failed.
    exit /b 1
)

echo.
echo  ============================================
echo  BUILD SUCCESSFUL
echo  ============================================
echo.
echo  Binary: build\bin\Release\llama-server.exe
echo.
echo  Quick start:
echo    Single model:
echo      build\bin\Release\llama-server.exe -m model.gguf -c 8192 --port 8090 -ngl 99
echo.
echo    Multi-model router:
echo      build\bin\Release\llama-server.exe --models-dir C:\path\to\models -c 8192 --port 8090 -ngl 99
echo.
echo  Tested configurations:
echo    - Windows 11 + RTX 4090 Laptop (16 GB) + CUDA 13.2
echo    - Gemma 4 E2B: 126 t/s generation, 3146 t/s prompt eval
echo    - Gemma 4 E4B: 70 t/s generation, 1883 t/s prompt eval
echo    - Both models loaded simultaneously: ~7.5 GB VRAM
echo.
