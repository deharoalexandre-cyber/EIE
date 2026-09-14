@echo off
REM EIE: start the bundled Windows CPU engine with its preset (port 8090, models\ next to this file).
cd /d "%~dp0"
"%~dp0eie-server.exe" --config "%~dp0presets\windows-cpu.yaml" %*
