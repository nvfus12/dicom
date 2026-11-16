# Cài đặt một worker từ bộ cài offline của máy điều phối: không cần Internet.
#
# Qua mạng LAB (máy điều phối đang chạy  python -m http.server 8000 --directory D:\lab-offline):
#   powershell -ExecutionPolicy Bypass -Command "iwr http://192.168.0.101:8000/setup_worker.ps1 -OutFile $env:TEMP\setup_worker.ps1; & $env:TEMP\setup_worker.ps1 -Source http://192.168.0.101:8000 -Machine trinh-duc"
# Từ USB:
#   powershell -ExecutionPolicy Bypass -File E:\lab-offline\setup_worker.ps1 -Source E:\lab-offline -Machine trinh-duc
#
# Chạy lại an toàn: phần đã xong được bỏ qua. Cuối cùng in bảng ĐẠT / LỖI.
param(
    [Parameter(Mandatory = $true)][string]$Source,
    [string]$Machine = "",
    [string]$Target = "C:\gpubench",
    [string]$PythonVersion = "3.13.2",
    [switch]$NoTorch        # chỉ để thử script, không cài PyTorch
)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$CifarMd5 = "C58F30108F718F92721AF3B95E74349A"
$PyTag = ($PythonVersion -split "\.")[0..1] -join "."
$IsUrl = $Source -match "^https?://"
$Source = $Source.TrimEnd("/", "\")
$results = [System.Collections.Generic.List[object]]::new()

function Record($item, $ok, $detail) {
    $results.Add([pscustomobject]@{ "Mục" = $item; "Kết quả" = $(if ($ok) { "ĐẠT" } else { "LỖI" }); "Chi tiết" = $detail })
}
function Step($text) { Write-Host "`n== $text" -ForegroundColor Cyan }
function Fetch($name, $dest) {
    if ($IsUrl) { Invoke-WebRequest "$Source/$name" -OutFile $dest }
    else { Copy-Item (Join-Path $Source $name) $dest -Force }
}
function Find-Python {
    $direct = Join-Path $env:LOCALAPPDATA "Programs\Python\Python$($PyTag -replace '\.', '')\python.exe"
    if (Test-Path $direct) { return $direct }
    try {
        $exe = & py "-$PyTag" -c "import sys; print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $exe) { return $exe.Trim() }
    } catch { }
    return $null
}

New-Item -ItemType Directory -Force $Target, "$Target\data" | Out-Null

Step "GPU NVIDIA"
try {
    $smi = (& nvidia-smi) -join "`n"
    if ($smi -match "CUDA Version:\s*([\d.]+)") {
        $cuda = [version]$Matches[1]
        Record "Driver NVIDIA" ($cuda -ge [version]"12.4") "CUDA Version $cuda (cần >= 12.4)"
    } else { Record "Driver NVIDIA" $false "nvidia-smi không báo CUDA Version" }
} catch { Record "Driver NVIDIA" $false "không có nvidia-smi: cài driver NVIDIA trước" }

Step "Python $PyTag"
$py = Find-Python
if (-not $py) {
    $installer = Join-Path $env:TEMP "python-$PythonVersion-amd64.exe"
    Write-Host "Chưa có Python $PyTag; đang cài bản $PythonVersion cho người dùng hiện tại..."
    Fetch "python-$PythonVersion-amd64.exe" $installer
    $p = Start-Process $installer -ArgumentList "/quiet", "InstallAllUsers=0", "Include_launcher=1",
        "InstallLauncherAllUsers=0", "PrependPath=0", "Include_test=0" -Wait -PassThru
    if ($p.ExitCode -ne 0) { Write-Host "Bộ cài Python trả mã $($p.ExitCode)" }
    $py = Find-Python
}
Record "Python $PyTag" ([bool]$py) $(if ($py) { $py } else { "cài không thành công" })
if (-not $py) { $results | Format-Table -AutoSize -Wrap; exit 1 }

Step "Code"
try {
    $zip = Join-Path $env:TEMP "gpubench-code.zip"
    Fetch "code.zip" $zip
    Expand-Archive $zip -DestinationPath $Target -Force   # giữ nguyên .venv và data đã có
    Record "Code" $true $Target
} catch { Record "Code" $false $_.Exception.Message }

Step "Môi trường .venv và thư viện"
$venvPy = Join-Path $Target ".venv\Scripts\python.exe"
try {
    if (-not (Test-Path $venvPy)) { & $py -m venv (Join-Path $Target ".venv"); if ($LASTEXITCODE) { throw "tạo .venv lỗi" } }
    $wheels = if ($IsUrl) { "$Source/wheels/" } else { Join-Path $Source "wheels" }
    $packages = @("pyyaml", "numpy", "nvidia-ml-py")
    if (-not $NoTorch) { $packages = @("torch==2.6.0", "torchvision==0.21.0") + $packages }
    & $venvPy -m pip install --no-index --find-links $wheels --disable-pip-version-check @packages
    if ($LASTEXITCODE) { throw "pip install trả mã $LASTEXITCODE" }
    Record "Thư viện" $true ($packages -join ", ")
} catch { Record "Thư viện" $false $_.Exception.Message }

Step "CIFAR-10"
$cifar = Join-Path $Target "data\cifar-10-python.tar.gz"
try {
    if (-not ((Test-Path $cifar) -and (Get-FileHash $cifar -Algorithm MD5).Hash -eq $CifarMd5)) {
        Fetch "data/cifar-10-python.tar.gz" "$cifar.part"
        Move-Item -Force "$cifar.part" $cifar
    }
    $ok = (Get-FileHash $cifar -Algorithm MD5).Hash -eq $CifarMd5
    Record "CIFAR-10" $ok $(if ($ok) { "đúng MD5" } else { "sai MD5: file CIFAR-10 trong bộ cài bị hỏng; chạy lại prepare_offline.ps1 trên máy điều phối" })
} catch { Record "CIFAR-10" $false $_.Exception.Message }

if (-not $NoTorch -and (Test-Path $venvPy)) {
    Step "PyTorch thấy GPU"
    $check = & $venvPy -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')" 2>&1
    $text = ($check | Select-Object -Last 1)
    Record "PyTorch + CUDA" ($LASTEXITCODE -eq 0 -and "$text" -match "\+cu124 True") "$text"
    if ($Machine) {
        Step "gpubench check -m $Machine"
        Push-Location $Target
        & $venvPy -m gpubench check -m $Machine
        Record "gpubench check" ($LASTEXITCODE -eq 0) "xem chi tiết ở trên"
        Pop-Location
    }
}

Write-Host ""
$results | Format-Table -AutoSize -Wrap
if ($results | Where-Object { $_."Kết quả" -eq "LỖI" }) {
    Write-Host "Còn mục LỖI. Chụp màn hình bảng này gửi cho người điều phối." -ForegroundColor Yellow
    exit 1
}
Write-Host "Máy đã sẵn sàng. Chạy agent:  cd $Target; .\.venv\Scripts\python.exe -m gpubench agent -m <tên máy>" -ForegroundColor Green
