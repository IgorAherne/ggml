@echo off
cd /d "%~dp0"
rem The ggml-cuda target of this worktree for the release build's GPU list (stable-diffusion.cpp\build-release.cmd:
rem sm_61;70;75;80;86;89;90;100;120, SASS + PTX each, the same CUDA flags), in build-relcheck: a compile check only.
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin;C:\_myDrive\repos\spellcast3d\code\venvs\pixal3d\Scripts;%PATH%"
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "CMAKE=C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
if not exist build-relcheck\CMakeCache.txt (
  "%CMAKE%" -S . -B build-relcheck -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DBUILD_SHARED_LIBS=ON ^
    "-DCMAKE_CUDA_ARCHITECTURES=61;70;75;80;86;89;90;100;120" ^
    -DCMAKE_CUDA_FLAGS="-Xcudafe --diag_suppress=177 -Xcudafe --diag_suppress=550" ^
    -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DGGML_BUILD_TESTS=OFF -DGGML_BUILD_EXAMPLES=OFF || exit /b 1
)
"%CMAKE%" --build build-relcheck --config Release -j 30 --target ggml-cuda || exit /b 1
echo BUILD OK
