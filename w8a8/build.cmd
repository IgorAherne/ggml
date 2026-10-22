@echo off
rem Builds the W8A8 speed probes (sm_86 only). Usage from Bash: cmd //c "C:\_myDrive\repos\spellcast3d\temp_code\ggml-mmq3\w8a8\build.cmd" [target]
cd /d "%~dp0"
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set "NVCC=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc.exe"
set "CUT=C:\_myDrive\repos\flash-attention\csrc\cutlass"
set "FLAGS=-O3 -std=c++17 -arch=sm_86 --expt-relaxed-constexpr -Xcompiler /Zc:__cplusplus -Xcudafe --diag_suppress=177"
if "%1"=="" goto all
goto %1
:all
:cutlass
"%NVCC%" %FLAGS% -I"%CUT%\include" -I"%CUT%\tools\util\include" w8a8_cutlass.cu -o w8a8_cutlass.exe || exit /b 1
if not "%1"=="" goto done
:cublaslt
"%NVCC%" %FLAGS% w8a8_cublaslt.cu -o w8a8_cublaslt.exe -lcublasLt || exit /b 1
if not "%1"=="" goto done
:quant
"%NVCC%" %FLAGS% w8a8_quant.cu -o w8a8_quant.exe || exit /b 1
:done
echo BUILD OK
