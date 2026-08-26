$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'

if (-not (Test-Path $Python)) {
    throw 'LitheVoice is not installed. Run .\scripts\setup.ps1 first.'
}

$env:PYTHONUTF8 = '1'
$env:HF_HOME = Join-Path $ProjectRoot 'models\huggingface'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
& $Python (Join-Path $ProjectRoot 'realtime.py') @args
exit $LASTEXITCODE
