@echo off
setlocal
if "%~2"=="" (
  echo usage: build_probe.cmd PADDLE_INFERENCE_DIR OUTPUT_DIR 1>&2
  exit /b 2
)
call "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b %errorlevel%
if not exist "%~2" mkdir "%~2"
cl /nologo /std:c++17 /O2 /EHsc /MT ^
  /I"%~1\paddle\include" ^
  "%~dp0paddle_probe.cpp" ^
  /link /LIBPATH:"%~1\paddle\lib" paddle_inference.lib ^
  /OUT:"%~2\paddle_probe.exe"
exit /b %errorlevel%
