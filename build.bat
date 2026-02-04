@echo off
echo [INFO] Building C-Core Shared Library (Deep Navy Optimized)...

REM Locate VS Tools
where cl >nul 2>nul
if %errorlevel% neq 0 (
    if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" (
        call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64
    ) else (
        echo [ERROR] Visual Studio 2022 Community not found.
        exit /b 1
    )
)

REM Cleanup
if exist csrc\libbattleship.dll del csrc\libbattleship.dll
if exist battleship.obj del battleship.obj
if exist csrc\libbattleship.lib del csrc\libbattleship.lib
if exist csrc\libbattleship.exp del csrc\libbattleship.exp

echo [INFO] Compiling battleship.c -> libbattleship.dll
cl /nologo /O2 /LD /I csrc/include csrc/src/battleship.c /Fe:csrc/libbattleship.dll

if %errorlevel% neq 0 (
    echo [ERROR] Compilation failed.
    exit /b 1
)

echo [SUCCESS] C-Core built successfully.
