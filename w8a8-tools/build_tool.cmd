@echo off
rem Builds one of the w8a8-tools programs (sm_86, the ggml-cuda flags: -O3 -use_fast_math).
rem Usage: build_tool.cmd name [output name]; extra nvcc arguments in the environment variable NVCC_EXTRA
rem (e.g. from Bash: NVCC_EXTRA="-DW8A8_WHATIF=1 -Xptxas=-v" cmd //c "...\build_tool.cmd" test_gemm test_gemm_w1)
cd /d "%~dp0"
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set "NVCC=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc.exe"
set "OUT=%2"
if "%OUT%"=="" set "OUT=%1"
"%NVCC%" -O3 -std=c++17 -arch=sm_86 -use_fast_math -extended-lambda -Xcompiler /Zc:__cplusplus -Xcudafe --diag_suppress=177 -I..\src\ggml-cuda %NVCC_EXTRA% %1.cu -o %OUT%.exe || exit /b 1
echo BUILD OK
