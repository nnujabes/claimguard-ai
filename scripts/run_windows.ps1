$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
Set-Location $root
if(-not (Test-Path '.venv\Scripts\python.exe')){throw '먼저 scripts\setup_windows.ps1 을 실행하세요.'}
& '.\.venv\Scripts\python.exe' -m streamlit run app\streamlit_app.py
