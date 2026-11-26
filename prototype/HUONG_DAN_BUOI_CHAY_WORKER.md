# Buổi thí nghiệm: việc của chủ máy worker

Dành cho chủ hai máy worker. Người điều phối làm mọi việc còn lại trên máy của họ; bạn chỉ cần chuẩn bị máy, mở agent và bấm **Cho phép** khi được hỏi.

| Tên máy | GPU | Thư mục code |
|---|---|---|
| `trinh-duc` | Quadro T1000 | `D:\gpubench` |
| `doan-tuan` | RTX 4050 Laptop | `E:\gpubench` |

Ở các lệnh dưới, chọn đúng dòng của máy mình. Mọi lệnh gõ trong **PowerShell** (bấm Start, gõ PowerShell, Enter), **không phải cmd**. Lệnh ghi "(Admin)" thì mở bằng chuột phải nút Start → **Terminal (Admin)**.

---

## 1. Cập nhật code (khi người điều phối gửi gói mới; mạng nào cũng được)

**Đóng cửa sổ agent trước** (nếu đang mở): agent đang chạy vẫn dùng code cũ cho tới khi mở lại.

Đặt `gpubench-pack.zip` vào Downloads, rồi giải nén **đè** lên thư mục cũ. Lệnh chỉ ghi đè các file có trong gói: `.venv`, dataset (`data\`), kết quả và log (`results\`) giữ nguyên, không cài lại gì. File cấu hình máy (`configs\machines\`) trở về bản trong gói; không sao, vì IP giờ được nhập trên giao diện của người điều phối.

`trinh-duc`:

```bash
Expand-Archive $HOME\Downloads\gpubench-pack.zip -DestinationPath D:\gpubench -Force
```

`doan-tuan`:

```bash
Expand-Archive $HOME\Downloads\gpubench-pack.zip -DestinationPath E:\gpubench -Force
```

Kiểm tra, trong thư mục code của máy mình (`cd D:\gpubench` hoặc `cd E:\gpubench`):

```bash
.\.venv\Scripts\python.exe -m gpubench porttest --help
```

Phải thấy chữ `pair`.

## 2. Nguồn (ngay khi tới LAB)

1. Cắm sạc.
2. Settings → System → Power & battery: Power mode **Best performance**; Screen and sleep khi cắm sạc: **Never**. Ghi lại giá trị cũ để trả lại sau.
3. Máy `doan-tuan` (ASUS): mở Armoury Crate, chọn **Turbo** hoặc **Performance**.
4. Đóng game, trình duyệt nặng, ứng dụng dùng GPU.
5. Kiểm tra (chỉ đọc, không thay đổi gì):

```bash
$b = Get-CimInstance Win32_Battery; $o = (Get-ItemProperty "HKLM:\SYSTEM\CurrentControlSet\Control\Power\User\PowerSchemes" -ErrorAction SilentlyContinue).ActiveOverlayAcPowerScheme; $mode = @{"ded574b5-45a0-4f42-8737-46345c09c238"="Best performance"; "00000000-0000-0000-0000-000000000000"="Balanced"; "961cc777-2547-4f9d-8174-7d86181b8a7a"="Best power efficiency"}[$o]; $sleep = @(powercfg /query SCHEME_CURRENT SUB_SLEEP STANDBYIDLE | Select-String ":\s*0x[0-9a-fA-F]+\s*$")[-2].ToString().Split(":")[-1].Trim(); "Cắm sạc: " + ($b.BatteryStatus -eq 2) + " | Power mode (khi cắm sạc): " + $(if ($mode) { $mode } else { "không đọc được ($o)" }) + " | Sleep khi cắm sạc: " + $(if ([Convert]::ToInt32($sleep, 16) -eq 0) { "Never" } else { "sau $([Convert]::ToInt32($sleep, 16) / 60) phút" })
```

Đạt khi in ra `Cắm sạc: True | Power mode (khi cắm sạc): Best performance | Sleep khi cắm sạc: Never`. "không đọc được" thì nhìn tận mắt trong Settings.

## 3. Tường lửa (kiểm tra; tạo nếu chưa có)

```bash
Get-NetFirewallRule -DisplayName gpubench-lab -ErrorAction SilentlyContinue | Select-Object DisplayName, Enabled, Direction, Action
```

- Có một dòng `Enabled True`, `Inbound`, `Allow`: xong.
- Không hiện gì: chạy hai lệnh này trong **Terminal (Admin)**:

```bash
New-NetFirewallRule -DisplayName gpubench-lab -Direction Inbound -Protocol TCP -RemoteAddress 192.168.0.0/24 -Action Allow
```

```bash
Get-NetFirewallRule -DisplayName python.exe -ErrorAction SilentlyContinue | Where-Object { $_.Direction -eq 'Inbound' -and $_.Action -eq 'Block' } | Disable-NetFirewallRule
```

Quy tắc chỉ cho các máy trong mạng router của LAB (`192.168.0.x`) kết nối vào, không mở ra Internet.

## 4. Nối Wi-Fi router của LAB và báo IP

1. Settings → Network & internet → Wi-Fi → **Manage known networks** → Wi-Fi của trường → **tắt "Connect automatically"** (để máy không tự nhảy về mạng trường).
2. Nối vào Wi-Fi của router LAB. Windows báo "No internet" là bình thường.
3. Chạy:

```bash
ipconfig
```

4. Đọc dòng **IPv4 Address** trong khối của card Wi-Fi thật, khối có `Default Gateway . . . : 192.168.0.1`:
   - `trinh-duc`: khối **`Wireless LAN adapter Wi-Fi 2`**.
   - `doan-tuan`: khối **`Wireless LAN adapter Wi-Fi`**.
   - Bỏ qua các khối `vEthernet (...)` có địa chỉ `172.x.x.x`: đó là card ảo.
5. **Báo địa chỉ này cho người điều phối.** Router có thể cấp số khác mỗi buổi; người điều phối nhập nó vào giao diện. Không sửa file nào trên máy bạn.

## 5. Mở agent và để mở suốt buổi

`trinh-duc`:

```bash
cd D:\gpubench
```

```bash
.\.venv\Scripts\python.exe -m gpubench agent -m trinh-duc
```

`doan-tuan`:

```bash
cd E:\gpubench
```

```bash
.\.venv\Scripts\python.exe -m gpubench agent -m doan-tuan
```

- Chờ vài giây tới vài chục giây (kiểm tra GPU) cho tới khi cửa sổ **"Agent LAB — <tên máy>"** hiện "Rảnh — đang chờ yêu cầu từ máy điều phối".
- Windows hỏi về tường lửa: bấm **Allow access**, không bấm Cancel.
- Dòng "Địa chỉ cho máy điều phối" trên cửa sổ lấy từ file cấu hình cũ, có thể sai: bỏ qua, tin `ipconfig`.
- **Không đóng cửa sổ agent và cửa sổ PowerShell** cho tới hết buổi.

## 6. Trong buổi chạy

**Khi hộp thoại xin phép hiện ra:** đọc nội dung (chế độ, model, thời gian tối đa, các máy cùng tham gia) rồi bấm **Cho phép**. Không trả lời trong 120 giây thì hộp thoại tự từ chối. Sẽ có các lượt sau, mỗi lượt hỏi một lần:

| Lượt | Hộp thoại ghi | Thời gian |
|---|---|---|
| Tự thử | "Tự thử gloo trên máy này (…)" | khoảng 1 phút |
| Thông đường TCP | "Thử thông đường TCP hai chiều giữa hai máy (…)" | khoảng 1 phút |
| glootest | "Chẩn đoán kết nối gloo giữa các máy (…)" | tối đa 4 phút |
| Kiểm tra mạng | "Kiểm tra mạng (…)" | 1–2 phút |
| Đo GPU | "Đo tốc độ GPU (khoảng 3 phút)" | khoảng 3 phút |
| Chạy thử | Dữ liệu tổng hợp, Tiny CNN | 1–2 phút |
| Chạy thử 1 epoch | CIFAR-10, ResNet-20, 1 epoch | 1–3 phút |
| Lượt chính | CIFAR-10, ResNet-20, 30 phút | khoảng 35 phút |
| Single GPU | "Single GPU (huấn luyện trên một máy)", chỉ một máy được hỏi | khoảng 35 phút |

Các phép chẩn đoán có thể được chạy lại vài lần (ví dụ đổi máy làm điểm gặp); mỗi lần hỏi lại. Kết quả và log tự về máy người điều phối, **không cần chụp màn hình**.

**Trong lúc huấn luyện:** không dùng máy, không đóng nắp, không để máy ngủ, không đóng cửa sổ agent. Cửa sổ agent hiện "Đang được dùng để huấn luyện: …, epoch x/y".

**Cần máy gấp:** bấm **Dừng huấn luyện** trên cửa sổ agent rồi xác nhận. Cả lượt dừng an toàn; người điều phối thấy ngay ai đã dừng.

## 6b. Lệnh chẩn đoán bằng tay (chỉ khi được nhờ)

Mọi phép chẩn đoán giờ chạy từ máy người điều phối qua agent. Chỉ khi giao diện không dùng được, người điều phối mới đọc cho bạn một lệnh cụ thể (`gpubench glootest …`, `gpubench porttest …`); khi đó đóng agent trước, chạy lệnh trong thư mục code của máy mình.

## 7. Khi người điều phối nhờ tắt IPv6 (chỉ khi được yêu cầu)

Dùng khi kết nối giữa hai máy bị lỗi. Chỉ tắt IPv6 trên card Wi-Fi; IPv4 (mạng router, mạng trường) vẫn chạy bình thường. **Terminal (Admin)**:

`trinh-duc`:

```bash
Disable-NetAdapterBinding -Name "Wi-Fi 2" -ComponentID ms_tcpip6
```

`doan-tuan`:

```bash
Disable-NetAdapterBinding -Name "Wi-Fi" -ComponentID ms_tcpip6
```

Sau đó chạy lại `ipconfig` và **báo lại địa chỉ IPv4** (có thể đổi). Nhớ bật lại ở bước 8.

## 8. Sau buổi chạy

1. Bấm **Thoát** trên cửa sổ agent.
2. Nếu đã tắt IPv6 ở bước 7, bật lại (Admin):

`trinh-duc`:

```bash
Enable-NetAdapterBinding -Name "Wi-Fi 2" -ComponentID ms_tcpip6
```

`doan-tuan`:

```bash
Enable-NetAdapterBinding -Name "Wi-Fi" -ComponentID ms_tcpip6
```

3. Bật lại "Connect automatically" cho Wi-Fi trường, nối lại mạng trường.
4. Trả lại chế độ nguồn, sleep, Armoury Crate như cũ.
5. Quy tắc tường lửa `gpubench-lab`: giữ nguyên nếu còn buổi chạy sau. Muốn gỡ (Admin):

```bash
Remove-NetFirewallRule -DisplayName gpubench-lab
```

## Sự cố thường gặp

| Triệu chứng | Cách xử lý |
|---|---|
| `'New-NetFirewallRule' is not recognized` | Đang gõ trong cmd. Mở **PowerShell** (Admin) rồi gõ lại |
| Agent báo lỗi cổng 8765 đang dùng | Đã có một cửa sổ agent khác đang mở. Dùng cửa sổ đó, hoặc đóng nó rồi mở lại |
| Người điều phối báo "không kết nối được" tới máy bạn | Agent còn mở không? Máy còn ở Wi-Fi router không (`ipconfig`)? Tường lửa (bước 3) đã có chưa? |
| Máy tự nhảy về Wi-Fi trường | Tắt "Connect automatically" của Wi-Fi trường (bước 4), nối lại router |
| Hộp thoại xin phép không hiện | Cửa sổ agent có thể bị che; nhìn thanh taskbar. Không thấy trong 2 phút thì báo người điều phối |

Log lỗi của mọi lượt tự được gửi về máy người điều phối; bạn không cần gửi gì thêm trừ khi được nhờ.
