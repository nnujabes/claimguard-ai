$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
Set-Location $root
if(-not (Test-Path '.venv')){python -m venv .venv}
& '.\.venv\Scripts\python.exe' -m pip install --upgrade pip
& '.\.venv\Scripts\python.exe' -m pip install -r app\requirements.txt
Write-Host '설치 완료. scripts\run_windows.ps1 로 실행하세요.' -ForegroundColor Green
