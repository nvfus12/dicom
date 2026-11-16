# Chuẩn bị bộ cài offline cho worker: chạy MỘT LẦN trên máy điều phối, khi có Internet nhanh.
#
#   powershell -ExecutionPolicy Bypass -File tools\offline\prepare_offline.ps1
#
# Tạo thư mục (mặc định D:\lab-offline) chứa: bộ cài Python, wheel PyTorch CUDA và mọi thư viện,
# CIFAR-10, code (code.zip) và setup_worker.ps1. Worker cài từ thư mục này qua mạng LAB hoặc USB,
# không cần Internet. Chạy lại an toàn: phần đã tải xong được bỏ qua.
param(
    [string]$Out = "D:\lab-offline",
    [string]$PythonVersion = "3.13.2",
    [string]$TorchIndex = "https://download.pytorch.org/whl/cu124",
    [switch]$SkipTorch,     # chỉ để thử script nhanh, bộ cài sẽ thiếu PyTorch
    [switch]$SkipData       # chỉ để thử script nhanh, bộ cài sẽ thiếu CIFAR-10
)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # thanh tiến độ làm Invoke-WebRequest chậm hàng chục lần
$CifarMd5 = "C58F30108F718F92721AF3B95E74349A"
$Prototype = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$PyTag = ($PythonVersion -split "\.")[0..1] -join "."

function Step($text) { Write-Host "`n== $text" -ForegroundColor Cyan }

New-Item -ItemType Directory -Force $Out, "$Out\wheels", "$Out\data" | Out-Null

Step "Bộ cài Python $PythonVersion"
$installer = "$Out\python-$PythonVersion-amd64.exe"
if (Test-Path $installer) { Write-Host "Đã có." } else {
    Invoke-WebRequest "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-amd64.exe" -OutFile "$installer.part"
    Move-Item "$installer.part" $installer
}

Step "Wheel cho Python $PyTag trên Windows 64-bit"
$pipArgs = @("-m", "pip", "download", "--dest", "$Out\wheels", "--only-binary=:all:",
             "--platform", "win_amd64", "--python-version", $PyTag, "--implementation", "cp")
if (-not $SkipTorch) {
    # Gói phụ thuộc (sympy, networkx, pillow...) lấy từ PyPI khi index PyTorch không có.
    & py "-$PyTag" @pipArgs "torch==2.6.0" "torchvision==0.21.0" --index-url $TorchIndex `
        --extra-index-url "https://pypi.org/simple"
    if ($LASTEXITCODE -ne 0) { throw "Tải wheel PyTorch thất bại" }
}
& py "-$PyTag" @pipArgs "pyyaml" "numpy" "nvidia-ml-py"
if ($LASTEXITCODE -ne 0) { throw "Tải wheel thư viện phụ thất bại" }

Step "CIFAR-10"
$cifar = "$Out\data\cifar-10-python.tar.gz"
if ($SkipData) { Write-Host "Bỏ qua (-SkipData)." }
elseif ((Test-Path $cifar) -and (Get-FileHash $cifar -Algorithm MD5).Hash -eq $CifarMd5) { Write-Host "Đã có, đúng MD5." }
else {
    Invoke-WebRequest "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz" -OutFile "$cifar.part"
    Move-Item -Force "$cifar.part" $cifar
    if ((Get-FileHash $cifar -Algorithm MD5).Hash -ne $CifarMd5) { throw "CIFAR-10 tải về bị hỏng (sai MD5); chạy lại script" }
}

Step "Code (code.zip)"
$stage = Join-Path $env:TEMP "lab-offline-code"
if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
robocopy $Prototype $stage /E /XD .venv results results-thu data __pycache__ _runs /XF lab.yaml PLAN.md /NFL /NDL /NJH /NJS | Out-Null
if ($LASTEXITCODE -ge 8) { throw "robocopy lỗi $LASTEXITCODE" }
Compress-Archive -Path "$stage\*" -DestinationPath "$Out\code.zip" -Force
Remove-Item -Recurse -Force $stage
Copy-Item (Join-Path $PSScriptRoot "setup_worker.ps1") $Out -Force

Step "Xong"
$size = (Get-ChildItem $Out -Recurse -File | Measure-Object Length -Sum).Sum
Write-Host ("Bộ cài offline: {0}  ({1:N2} GB, {2} wheel)" -f $Out, ($size / 1GB), (Get-ChildItem "$Out\wheels").Count)
Write-Host "Chia sẻ qua mạng LAB:  python -m http.server 8000 --directory $Out"
