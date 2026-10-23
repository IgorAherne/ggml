@echo off
cd /d "%~dp0"
rem Standalone sm_86 build of this ggml worktree (flags as stable-diffusion.cpp\build-sm86.cmd)
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin;C:\_myDrive\repos\spellcast3d\code\venvs\pixal3d\Scripts;%PATH%"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "CMAKE=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
if not exist build-sm86\CMakeCache.txt (
  "%CMAKE%" -S . -B build-sm86 -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DBUILD_SHARED_LIBS=ON ^
    -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_CUDA_FLAGS="-Xcudafe --diag_suppress=177 -Xcudafe --diag_suppress=550" ^
    -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DGGML_BUILD_TESTS=OFF -DGGML_BUILD_EXAMPLES=OFF || exit /b 1
)
rem re-configure (quick with the cache): picks up new source files (ggml-cuda globs *.cu at configure time)
"%CMAKE%" -S . -B build-sm86 >nul || exit /b 1
"%CMAKE%" --build build-sm86 --config Release -j 24 || exit /b 1
if not exist build-sm86\bin\cudart64_12.dll copy /y C:\_myDrive\repos\stable-diffusion.cpp\build-sm86\bin\cudart64_12.dll build-sm86\bin\ >nul
if not exist build-sm86\bin\cublas64_12.dll copy /y C:\_myDrive\repos\stable-diffusion.cpp\build-sm86\bin\cublas64_12.dll build-sm86\bin\ >nul
if not exist build-sm86\bin\cublasLt64_12.dll copy /y C:\_myDrive\repos\stable-diffusion.cpp\build-sm86\bin\cublasLt64_12.dll build-sm86\bin\ >nul
echo BUILD OK
