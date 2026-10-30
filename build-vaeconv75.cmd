@echo off
cd /d "%~dp0"
rem Standalone compute_75 build (the Turing paths, run on newer cards with GGML_CUDA_FAKE_CC=750) of this ggml worktree (flags as stable-diffusion.cpp\build-sm75.cmd) (tests off)
rem Default target: ggml-cuda.
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin;C:\_myDrive\repos\spellcast3d\code\venvs\pixal3d\Scripts;%PATH%"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "CMAKE=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
set "TARGET=%~1"
if "%TARGET%"=="" set "TARGET=ggml-cuda"
if not exist build-sm75\CMakeCache.txt (
  "%CMAKE%" -S . -B build-sm75 -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DBUILD_SHARED_LIBS=ON ^
    -DCMAKE_CUDA_ARCHITECTURES=75 -DCMAKE_CUDA_FLAGS="-Xcudafe --diag_suppress=177 -Xcudafe --diag_suppress=550" ^
    -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DGGML_BUILD_TESTS=OFF -DGGML_BUILD_EXAMPLES=OFF || exit /b 1
)
rem re-configure (quick with the cache): picks up new source files (ggml-cuda globs *.cu at configure time)
"%CMAKE%" -S . -B build-sm75 >nul || exit /b 1
"%CMAKE%" --build build-sm75 --config Release -j 12 --target %TARGET% || exit /b 1
if not exist build-sm75\bin\cudart64_12.dll copy /y C:\_myDrive\repos\stable-diffusion.cpp\build-sm75\bin\cudart64_12.dll build-sm75\bin\ >nul
if not exist build-sm75\bin\cublas64_12.dll copy /y C:\_myDrive\repos\stable-diffusion.cpp\build-sm75\bin\cublas64_12.dll build-sm75\bin\ >nul
if not exist build-sm75\bin\cublasLt64_12.dll copy /y C:\_myDrive\repos\stable-diffusion.cpp\build-sm75\bin\cublasLt64_12.dll build-sm75\bin\ >nul
echo BUILD OK
