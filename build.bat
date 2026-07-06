@echo off
REM ─────────────────────────────────────────────────────
REM  STAR Analyzer — Build Script
REM  Produces: dist\STAR Analyzer\STAR Analyzer.exe
REM ─────────────────────────────────────────────────────

echo [1/4] Installing build dependencies...
pip install pyinstaller Pillow

echo [2/4] Generating application icon...
python build_icon.py

echo [3/4] Building application with PyInstaller...
python -m PyInstaller star_analyzer.spec --noconfirm

echo [4/4] Build complete!
echo.
echo Executable: dist\STAR Analyzer\STAR Analyzer.exe
echo.

REM Optional: create desktop shortcut
set /p SHORTCUT="Create desktop shortcut? (y/n): "
if /i "%SHORTCUT%"=="y" (
    powershell -Command "$ws = New-Object -ComObject WScript.Shell; $s = $ws.CreateShortcut([IO.Path]::Combine([Environment]::GetFolderPath('Desktop'), 'STAR Analyzer.lnk')); $s.TargetPath = [IO.Path]::Combine('%cd%', 'dist', 'STAR Analyzer', 'STAR Analyzer.exe'); $s.WorkingDirectory = [IO.Path]::Combine('%cd%', 'dist', 'STAR Analyzer'); $s.IconLocation = $s.TargetPath; $s.Save()"
    echo Desktop shortcut created.
)

pause
