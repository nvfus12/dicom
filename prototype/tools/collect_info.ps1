# Thu thap cau hinh phuc vu prototype. Chi doc thong tin, khong thay doi he thong.
# Cach chay:  powershell -ExecutionPolicy Bypass -File collect_info.ps1
# Ket qua: file <TEN-MAY>-info.txt canh script; gui file nay lai.
# Luu y: muon co thong tin Wi-Fi (SSID, chuan, toc do lien ket), bat Location services
# trong Settings > Privacy & security > Location truoc khi chay.

$out = Join-Path $PSScriptRoot "$env:COMPUTERNAME-info.txt"

& {
    "=== Thoi diem"
    Get-Date -Format o

    "`n=== He thong"
    Get-CimInstance Win32_ComputerSystem | Select-Object Manufacturer, Model, @{n='RAM_GB';e={[math]::Round($_.TotalPhysicalMemory/1GB,1)}} | Format-List
    Get-CimInstance Win32_OperatingSystem | Select-Object Caption, Version, BuildNumber | Format-List
    Get-CimInstance Win32_Processor | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors | Format-List

    "`n=== GPU (nvidia-smi)"
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        nvidia-smi
        nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,power.limit,temperature.gpu --format=csv
    } else { "nvidia-smi khong co trong PATH" }

    "`n=== O dia"
    Get-PSDrive -PSProvider FileSystem | Select-Object Name, @{n='FreeGB';e={[math]::Round($_.Free/1GB,1)}}, @{n='UsedGB';e={[math]::Round($_.Used/1GB,1)}} | Format-Table -AutoSize

    "`n=== Mang"
    Get-NetIPConfiguration | Where-Object { $_.IPv4DefaultGateway } | ForEach-Object {
        "Interface: $($_.InterfaceAlias)  IPv4: $($_.IPv4Address.IPAddress)  Gateway: $($_.IPv4DefaultGateway.NextHop)"
    }
    Get-NetConnectionProfile | Select-Object Name, InterfaceAlias, NetworkCategory | Format-List
    netsh wlan show interfaces

    "`n=== Python / PyTorch"
    if (Get-Command python -ErrorAction SilentlyContinue) {
        python --version
        python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())"
    } else { "python khong co trong PATH" }

    "`n=== Che do nguon"
    powercfg /getactivescheme
} *>&1 | Out-File -Encoding utf8 $out

"Da ghi $out"
