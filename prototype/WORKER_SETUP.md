# Cài đặt máy worker cho buổi thí nghiệm LAB

Tài liệu này dành cho **người hoặc AI agent đang cài đặt một laptop worker** từ gói `gpubench-pack.zip`. Chỉ làm đúng các bước dưới đây. Các phần khác của `README.md` (VPN, Tailscale, `profile`, sửa file máy, chạy thí nghiệm) **không áp dụng** cho việc này.

## Bối cảnh

- Máy này là một **worker** có GPU NVIDIA. Nó sẽ huấn luyện mô hình cùng một worker khác, qua Wi-Fi router riêng của LAB (`192.168.0.x`).
- Máy điều phối (không có GPU) sẽ gửi yêu cầu tới **agent** chạy trên máy này; chủ máy bấm Cho phép rồi huấn luyện tự chạy. Không ai phải gõ lệnh trong lúc thí nghiệm.
- Mục tiêu bây giờ: cài đủ phần mềm, tải dữ liệu, kiểm tra mọi thứ chạy được, rồi báo kết quả cho người điều phối.

## Những điều KHÔNG được làm

- Không sửa gì trong `configs\` (đặc biệt `address`, `network_interface` trong file máy: đã được điền đúng cho mạng router).
- Không cài PyTorch bản khác **2.6.0**, không nâng cấp torch, torchvision (bản 2.10 không chạy được nhiều máy trên Windows).
- Không cài VPN, không đổi cài đặt mạng, nguồn, tường lửa khi chưa hỏi chủ máy.
- Không chạy `run`, `netcheck`, `profile`, `controller` hay thí nghiệm nào; không xoá thư mục nào ngoài `C:\gpubench`.

## Bước 0. Xác định tên máy

```bash
nvidia-smi
```

| Tên GPU trong `nvidia-smi` có chứa | Tên máy (`<tên máy>` ở các lệnh dưới) |
|---|---|
| `RTX 4050` | `doan-tuan` |
| `T1000` | `trinh-duc` |

Chỉ có hai máy này tham gia. Nếu GPU khác hai dòng trên, hoặc không chắc, **dừng lại và hỏi người điều phối**.

Ghi lại `Driver Version` và `CUDA Version` (góc trên bên phải). `CUDA Version` phải **≥ 12.4**. Nếu thấp hơn hoặc không có lệnh `nvidia-smi`: chủ máy cần cài driver NVIDIA mới nhất từ nvidia.com và khởi động lại; dừng cho tới khi xong.

## Bước 1. Đặt code vào `C:\gpubench`

Gói đã giải nén phải nằm ở **đúng `C:\gpubench`** (đường dẫn dài sẽ làm cài PyTorch lỗi vì giới hạn 260 ký tự của Windows). Nếu chưa:

```bash
Expand-Archive <đường dẫn tới gpubench-pack.zip> -DestinationPath C:\gpubench -Force
```

Kiểm tra: `C:\gpubench\gpubench`, `C:\gpubench\configs\machines\<tên máy>.yaml`, `C:\gpubench\requirements.txt` đều tồn tại. Mọi lệnh sau chạy trong `C:\gpubench` (PowerShell):

```bash
cd C:\gpubench
```

## Bước 2. Chọn Python

```bash
py -0p
```

Cần một bản **3.10, 3.11, 3.12 hoặc 3.13** (64-bit). Ưu tiên 3.13 hoặc 3.12. **Không dùng 3.14** (chưa có PyTorch 2.6.0). Nếu không có bản phù hợp: cài Python 3.13 từ python.org (bản Windows 64-bit, cho người dùng hiện tại). Dưới đây giả sử dùng `py -3.13`; thay số phiên bản nếu khác.

## Bước 3. Tạo môi trường và cài thư viện (tải khoảng 2,5 GB)

Luôn gọi thẳng `.\.venv\Scripts\python.exe`; không dùng `activate` (Windows có thể chặn chạy script).

```bash
py -3.13 -m venv .venv
```

```bash
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

```bash
.\.venv\Scripts\python.exe -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
```

```bash
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Kiểm tra:

```bash
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Đúng là `2.6.0+cu124 True <tên GPU>`. Nếu ra `+cpu` hoặc `False`: chạy lại lệnh cài torch, thêm `--force-reinstall`. Vẫn `False` thì kiểm tra lại driver (Bước 0).

## Bước 4. Tải CIFAR-10 (khoảng 163 MB, máy chủ chậm, có thể 5–10 phút)

```bash
.\.venv\Scripts\python.exe -c "from torchvision import datasets; datasets.CIFAR10('data', download=True); datasets.CIFAR10('data', train=False, download=True)"
```

Dữ liệu phải nằm trong `C:\gpubench\data` (file máy dùng `data_dir: ./data`). Bị ngắt giữa chừng thì chạy lại; lần chạy lại thành công in `Files already downloaded and verified`.

## Bước 5. Kiểm tra tổng thể

```bash
.\.venv\Scripts\python.exe -m gpubench check -m <tên máy>
```

Phải có đủ: `cuda available True`, `gloo device: OK`, `Device: {'type': 'cuda', ...}`, `Phép nhân ma trận thử: OK`.

Kiểm tra tên card Wi-Fi khớp với file máy:

```bash
Get-NetAdapter -Physical | Select-Object Name, InterfaceDescription, Status
```

Giá trị `network_interface` trong `configs\machines\<tên máy>.yaml` (`Wi-Fi` cho `doan-tuan`, `Wi-Fi 2` cho `trinh-duc`) phải trùng **tên** card Wi-Fi trong danh sách. Nếu không trùng: **không sửa file**, ghi lại tên thật để báo người điều phối.

## Bước 6. Tường lửa (CHỦ MÁY làm, cần quyền Admin)

Nếu không có quyền Admin, đưa hai lệnh này cho chủ máy chạy trong **Terminal (Admin)** (chuột phải nút Start):

```bash
New-NetFirewallRule -DisplayName gpubench-lab -Direction Inbound -Protocol TCP -RemoteAddress 192.168.0.0/24 -Action Allow
```

```bash
Get-NetFirewallRule -DisplayName python.exe -ErrorAction SilentlyContinue | Where-Object { $_.Direction -eq 'Inbound' -and $_.Action -eq 'Block' } | Disable-NetFirewallRule
```

Lệnh 1 chỉ cho các máy trong mạng router LAB kết nối vào (agent cổng 8765 và kết nối huấn luyện gloo), không mở ra Internet. Lệnh 2 tạm tắt quy tắc chặn Python (thắng mọi quy tắc cho phép), nếu có. Kiểm tra:

```bash
Get-NetFirewallRule -DisplayName gpubench-lab | Select-Object DisplayName, Enabled, Action
```

## Bước 7. Thử mở agent (CHỦ MÁY nhìn cửa sổ)

```bash
.\.venv\Scripts\python.exe -m gpubench agent -m <tên máy>
```

Lệnh in `Đang kiểm tra GPU...`, sau vài giây tới vài chục giây mở cửa sổ **"Agent LAB — <tên máy>"** với dòng "Rảnh — đang chờ yêu cầu từ máy điều phối". Nếu Windows hỏi về tường lửa: bấm **Allow access**, không bấm Cancel. Chủ máy xác nhận thấy cửa sổ rồi bấm **Thoát**. (AI agent không nhìn được cửa sổ: hỏi chủ máy, hoặc coi là đạt nếu lệnh chạy không báo lỗi tới khi cửa sổ được đóng.)

## Bước 8. Báo kết quả cho người điều phối

Gửi bảng này (kèm ảnh chụp `nvidia-smi` và kết quả `gpubench check`):

| Mục | Giá trị |
|---|---|
| Tên máy | |
| GPU, Driver Version, CUDA Version | |
| Python dùng cho `.venv` | |
| `torch.__version__`, `cuda.is_available()` | |
| CIFAR-10 đã tải | có / không |
| `gpubench check` đạt đủ 4 dòng | có / không |
| Tên card Wi-Fi thật / có khớp `network_interface` | |
| Tường lửa `gpubench-lab` đã tạo | có / không |
| Cửa sổ agent mở được | có / không |
| Lỗi gặp phải (nguyên văn) | |

## Hôm thí nghiệm (chủ máy làm)

1. Cắm sạc. Settings → System → Power: Power mode **Best performance**; Screen and sleep khi cắm sạc: **Never**.
2. Nối Wi-Fi router của LAB. Tắt "Connect automatically" của Wi-Fi trường để máy không tự nhảy mạng.
3. `ipconfig`: báo người điều phối địa chỉ IPv4 của card Wi-Fi (`doan-tuan` dự kiến `192.168.0.103`, `trinh-duc` dự kiến `192.168.0.102`).
4. Mở agent như Bước 7 và để cửa sổ mở suốt buổi. Khi hộp thoại xin phép hiện ra, đọc nội dung rồi bấm **Cho phép**.
