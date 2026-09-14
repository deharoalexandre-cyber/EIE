@echo off
REM EIE: Windows x64 CPU build (no CUDA), static and portable binary.
REM Requires on PATH: cmake, ninja and either g++ (MinGW-w64 / WinLibs) or cl.exe
REM (Visual Studio 2022 Build Tools, from a Developer PowerShell).
REM Usage: scripts\build-windows-cpu.bat [build-dir]   (default: build-windows-cpu)
REM Prerequisite, once per checkout: git submodule update --init
REM   and git -C llama.cpp apply ..\patches\ews-runtime-2168b0.patch
setlocal
cd /d "%~dp0.."
set "BUILD=%~1"
if "%BUILD%"=="" set "BUILD=build-windows-cpu"

if not exist "llama.cpp\src" (
    echo [ERROR] llama.cpp submodule missing: git submodule update --init
    exit /b 1
)
where cmake >nul 2>&1 || (echo [ERROR] cmake not on PATH & exit /b 1)

REM GGML_NATIVE=OFF: no -march=native, so the binary runs on any x86-64 CPU with
REM AVX2/FMA/F16C (Intel Haswell 2013+, AMD Zen). BUILD_SHARED_LIBS=OFF: one file.
REM GGML_OPENMP=OFF: ggml's own thread pool, so the binary does not depend on
REM libgomp-1.dll (GCC) or vcomp140.dll (MSVC).
set "COMMON=-DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF -DBUILD_SHARED_LIBS=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_MTMD=OFF"

where g++ >nul 2>&1
if %errorlevel%==0 (
    where ninja >nul 2>&1 || (echo [ERROR] ninja not on PATH & exit /b 1)
    echo [OK] Toolchain: GCC/MinGW-w64 + Ninja, fully static link
    REM _WIN32_WINNT=0x0A00: cpp-httplib refuses MinGW's older default target.
    cmake -B "%BUILD%" -G Ninja %COMMON% -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ ^
        -DCMAKE_C_FLAGS=-D_WIN32_WINNT=0x0A00 -DCMAKE_CXX_FLAGS=-D_WIN32_WINNT=0x0A00 ^
        -DCMAKE_EXE_LINKER_FLAGS=-static || exit /b 1
    cmake --build "%BUILD%" --target eie-server -j %NUMBER_OF_PROCESSORS% || exit /b 1
    echo Binary: %BUILD%\eie-server.exe
    exit /b 0
)

where cl.exe >nul 2>&1
if %errorlevel%==0 (
    echo [OK] Toolchain: MSVC, static runtime
    REM MSVC Build Tools may lack MASM: ggml only needs C/C++ here.
    powershell -Command "(Get-Content 'llama.cpp\ggml\CMakeLists.txt') -replace 'project\(ggml C CXX ASM\)', 'project(ggml C CXX)' | Set-Content 'llama.cpp\ggml\CMakeLists.txt'"
    cmake -B "%BUILD%" -G "Visual Studio 17 2022" -A x64 %COMMON% ^
        -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded || exit /b 1
    cmake --build "%BUILD%" --config Release --target eie-server -j %NUMBER_OF_PROCESSORS% || exit /b 1
    echo Binary: %BUILD%\Release\eie-server.exe
    exit /b 0
)

echo [ERROR] No C++ compiler on PATH: install WinLibs GCC (winget install BrechtSanders.WinLibs.POSIX.UCRT)
echo         or run from a Developer PowerShell for VS 2022.
exit /b 1
