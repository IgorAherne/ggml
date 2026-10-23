@echo off
rem Builds w8a8bench.exe against this worktree's ggml build (build-sm86)
cd /d "%~dp0"
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set G=..
set B=..\build-sm86\src
cl /nologo /O2 /EHsc /std:c++17 /I %G%\include w8a8bench.cpp /Fe:w8a8bench.exe /link %B%\ggml.lib %B%\ggml-base.lib || exit /b 1
echo BUILD OK
