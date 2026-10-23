#!/bin/bash
# lk.sh "<purpose>" <minutes> "<bash command>": runs the command with Git's bash (not WSL's) in this directory under the GPU lock
exec /c/_myDrive/repos/spellcast3d/code/venvs/pixal3d/Scripts/python.exe "C:/Users/Igor/AppData/Local/Temp/claude/C---myDrive-repos-spellcast3d/28dae9df-ec87-4365-a233-4b80e1acef34/scratchpad/qs3/lockrun.py" "$1" "$2" -- "C:\Program Files\Git\usr\bin\bash.exe" -c "cd '$(pwd)' && $3"
