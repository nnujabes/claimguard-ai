$ErrorActionPreference='Stop'
$root=Split-Path -Parent $PSScriptRoot
Set-Location $root
if(-not (Test-Path '.venv\Scripts\python.exe')){throw '먼저 scripts\setup_windows.ps1 을 실행하세요.'}
& '.\.venv\Scripts\python.exe' -m pip install -r training\requirements.txt
Write-Host 'FUSED 학습 환경 설치 완료.' -ForegroundColor Green
