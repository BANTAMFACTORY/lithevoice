[CmdletBinding()]
param(
    [ValidateSet('auto', 'cpu', 'cuda', 'vulkan', 'hip')]
    [string]$LlamaBackend = 'auto',
    [switch]$CpuOnly,
    [switch]$IncludeGpuStt,
    [switch]$SkipModels,
    [switch]$SkipLlama,
    [switch]$ForceDownloads
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$env:PYTHONUTF8 = '1'
$env:HF_HOME = Join-Path $ProjectRoot 'models\huggingface'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'

function Invoke-Python {
    & $VenvPython @args
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE"
    }
}

if (-not [Environment]::Is64BitOperatingSystem) {
    throw 'LitheVoice requires 64-bit Windows.'
}

$BasePython = $null
if (Get-Command py -ErrorAction SilentlyContinue) {
    $BasePython = (& py -3.12 -c "import sys; print(sys.executable)" 2>$null)
}
if (-not $BasePython -and (Get-Command python -ErrorAction SilentlyContinue)) {
    $candidate = (& python -c "import sys; print(sys.executable)" 2>$null)
    $versionOk = (& $candidate -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)" 2>$null)
    if ($LASTEXITCODE -eq 0) {
        $BasePython = $candidate
    }
}
if (-not $BasePython) {
    throw 'Python 3.12 x64 is required. Install it from https://www.python.org/downloads/windows/ and rerun setup.'
}

$free = (Get-PSDrive -Name $ProjectRoot.Substring(0, 1)).Free
if (-not $SkipModels -and $free -lt 12GB) {
    throw ('At least 12 GB free is required for models and installation; only {0:N1} GB is available.' -f ($free / 1GB))
}

Push-Location $ProjectRoot
try {
    if (-not (Test-Path $VenvPython)) {
        Write-Host 'Creating .venv with Python 3.12...'
        & $BasePython -m venv (Join-Path $ProjectRoot '.venv')
        if ($LASTEXITCODE -ne 0) { throw 'Failed to create the virtual environment.' }
    }

    Write-Host 'Updating Python packaging tools...'
    Invoke-Python -m pip install --upgrade 'pip==25.0.1' 'setuptools>=75,<82' wheel

    $NvidiaDetected = $false
    if (-not $CpuOnly -and (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
        & nvidia-smi --query-gpu=name --format=csv,noheader 1>$null 2>$null
        $NvidiaDetected = $LASTEXITCODE -eq 0
    }

    if ($LlamaBackend -eq 'auto') {
        $ResolvedBackend = if ($NvidiaDetected) { 'cuda' } else { 'cpu' }
    } else {
        $ResolvedBackend = $LlamaBackend
    }
    if ($CpuOnly) {
        $ResolvedBackend = 'cpu'
        $NvidiaDetected = $false
    }
    if ($ResolvedBackend -eq 'cuda' -and -not $NvidiaDetected) {
        throw 'The CUDA llama.cpp backend was requested, but no working NVIDIA driver was detected.'
    }

    $TorchIndex = if ($NvidiaDetected) {
        'https://download.pytorch.org/whl/cu124'
    } else {
        'https://download.pytorch.org/whl/cpu'
    }
    Write-Host "Installing PyTorch 2.6.0 from $TorchIndex..."
    $InstalledTorch = (& $VenvPython -c "
try:
    import torch
    print(torch.__version__)
except Exception:
    print('')
").Trim()
    $ExpectedTorchTag = if ($NvidiaDetected) { '+cu124' } else { '+cpu' }
    $TorchArgs = @('-m', 'pip', 'install', '--upgrade')
    if ($InstalledTorch -and -not $InstalledTorch.EndsWith($ExpectedTorchTag)) {
        Write-Host "Replacing incompatible Torch build $InstalledTorch with $ExpectedTorchTag..."
        $TorchArgs += '--force-reinstall'
    }
    $TorchArgs += @('torch==2.6.0', 'torchaudio==2.6.0', '--index-url', $TorchIndex)
    Invoke-Python @TorchArgs
    # PyTorch's import-time NumPy bridge warns when NumPy is not present yet.
    # Install the project pin before validating CUDA and before ONNX resolves it.
    Invoke-Python -m pip install 'numpy==2.5.0'

    if ($NvidiaDetected) {
        & $VenvPython -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"
        if ($LASTEXITCODE -ne 0) {
            if ($LlamaBackend -eq 'cuda') {
                throw 'CUDA was explicitly requested, but PyTorch cannot use the NVIDIA GPU.'
            }
            Write-Warning 'The NVIDIA driver was found, but PyTorch CUDA validation failed. Falling back to CPU.'
            $NvidiaDetected = $false
            $ResolvedBackend = 'cpu'
            Invoke-Python -m pip install --upgrade --force-reinstall 'torch==2.6.0' 'torchaudio==2.6.0' --index-url 'https://download.pytorch.org/whl/cpu'
        }
    }

    Write-Host 'Installing the matching ONNX Runtime...'
    $PackageMap = @{}
    (& $VenvPython -m pip list --format=json | ConvertFrom-Json) |
        ForEach-Object { $PackageMap[$_.name.ToLowerInvariant()] = $_.version }
    $ExpectedOrt = if ($NvidiaDetected) { 'onnxruntime-gpu' } else { 'onnxruntime' }
    $OtherOrt = if ($NvidiaDetected) { 'onnxruntime' } else { 'onnxruntime-gpu' }
    $OrtReady = $PackageMap[$ExpectedOrt] -eq '1.22.0' -and -not $PackageMap.ContainsKey($OtherOrt)
    if (-not $OrtReady) {
        $OrtToRemove = @(@('onnxruntime', 'onnxruntime-gpu') |
            Where-Object { $PackageMap.ContainsKey($_) })
        if ($OrtToRemove.Count -gt 0) {
            $UninstallArgs = @('-m', 'pip', 'uninstall', '-y') + $OrtToRemove
            Invoke-Python @UninstallArgs
        }
        Invoke-Python -m pip install "$ExpectedOrt==1.22.0"
    } else {
        Write-Host "$ExpectedOrt 1.22.0 is already installed."
    }

    Write-Host 'Installing the LitheVoice Python dependencies...'
    Invoke-Python -m pip install -r (Join-Path $ProjectRoot 'requirements.txt')
    $HasEnglishModel = (& $VenvPython -c "import importlib.util; print('yes' if importlib.util.find_spec('en_core_web_sm') else 'no')").Trim()
    if ($HasEnglishModel -ne 'yes') {
        Write-Host 'Installing the Kokoro English language pipeline...'
        Invoke-Python -m pip install 'https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl'
    }

    $DownloadArgs = @(
        (Join-Path $PSScriptRoot 'download_models.py'),
        '--backend', $ResolvedBackend
    )
    if ($IncludeGpuStt) { $DownloadArgs += '--include-gpu-stt' }
    if ($SkipModels) { $DownloadArgs += '--skip-models' }
    if ($SkipLlama) { $DownloadArgs += '--skip-llama' }
    if ($ForceDownloads) { $DownloadArgs += '--force' }
    Write-Host "Downloading pinned models and llama.cpp ($ResolvedBackend)..."
    Invoke-Python @DownloadArgs

    $DoctorArgs = @((Join-Path $PSScriptRoot 'doctor.py'))
    if ($SkipModels) { $DoctorArgs += '--skip-models' }
    if ($SkipLlama) { $DoctorArgs += '--skip-llama' }
    Write-Host 'Checking the installation...'
    Invoke-Python @DoctorArgs

    Write-Host ''
    Write-Host 'LitheVoice is ready.' -ForegroundColor Green
    Write-Host 'Run: .\run.ps1 --barge-key'
} finally {
    Pop-Location
}
