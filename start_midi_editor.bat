@echo off
setlocal

rem ============================================================
rem  SoulX-Singer MIDI Editor launcher (Vite dev server)
rem  Usage: double-click this file, or run:  start_midi_editor.bat
rem ============================================================

rem Move to the midi_editor directory (relative to this script)
set "EDITOR_DIR=%~dp0preprocess\tools\midi_editor"
if not exist "%EDITOR_DIR%\package.json" (
    echo [ERROR] package.json not found: "%EDITOR_DIR%"
    pause
    exit /b 1
)

rem Locate node
where node >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Node.js not found. Please install Node.js 18+ and add it to PATH.
    pause
    exit /b 1
)

cd /d "%EDITOR_DIR%"

rem Install dependencies on first run
if not exist "node_modules" (
    echo [INFO] node_modules not found, running npm install ...
    call npm install
    if errorlevel 1 (
        echo [ERROR] npm install failed.
        pause
        exit /b 1
    )
)

echo.
echo [INFO] Starting MIDI Editor ...
echo        Press Ctrl+C in this window to stop the server.
echo.
call npm run dev -- --open --host

endlocal
