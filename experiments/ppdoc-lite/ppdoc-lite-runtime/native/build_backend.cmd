@echo off
setlocal
if "%~2"=="" (
  echo usage: build_backend.cmd PADDLE_INFERENCE_DIR OUTPUT_DIR 1>&2
  exit /b 2
)
call "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b %errorlevel%
if not exist "%~2" mkdir "%~2"
set "PPDOC_DEFINES="
if /I "%~3"=="legacy" set "PPDOC_DEFINES=/DPPDOC_LEGACY_ONEDNN"
cl /nologo /std:c++17 /O2 /EHsc /MT /LD %PPDOC_DEFINES% ^
  /I"%~1\paddle\include" ^
  "%~dp0ppdoc_paddle.cpp" ^
  /link /LIBPATH:"%~1\paddle\lib" paddle_inference.lib ^
  /OUT:"%~2\ppdoc_paddle.dll"
exit /b %errorlevel%
