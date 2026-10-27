# Demo: xin phép dùng GPU của máy khác

Demo cho phần "chủ máy giữ quyền quyết định" của đề tài. Máy điều phối chọn một
máy worker và gửi yêu cầu xin dùng tài nguyên; trên máy đó hiện một hộp thoại để
chủ máy bấm **Cho phép** hoặc **Từ chối**, và câu trả lời quay về máy điều phối.

**Demo này không chiếm tài nguyên thật và không chạy huấn luyện.** Nó chỉ minh hoạ
bước xin phép. Việc giới hạn CPU/RAM/VRAM thật và thu hồi GPU đang chạy chưa làm.

Chỉ dùng thư viện chuẩn của Python, **không cần cài thêm gì** trên máy các bạn.

## Chạy thử ngay trên một máy

Mở ba cửa sổ PowerShell trong thư mục `demo`:

```bash
python agent.py --name may-a --port 8765
```

```bash
python agent.py --name may-b --port 8766
```

```bash
python controller.py
```

Trong cửa sổ controller: bấm **Kiểm tra kết nối tất cả máy**, chọn `may-a`,
rồi bấm **Gửi yêu cầu đến máy đã chọn**. Hộp thoại sẽ hiện ở cửa sổ `may-a`.

## Chạy trên nhiều máy thật

Trên **mỗi máy worker**:

1. Chép `agent.py` sang máy đó (chỉ cần đúng một tệp này).
2. Chạy, đặt tên cho dễ nhận:
   ```bash
   python agent.py --name may-a
   ```
3. Cửa sổ agent hiện dòng **Địa chỉ cho máy điều phối**, ví dụ `192.168.1.21:8765`.
   Ghi lại địa chỉ này.
4. Lần đầu chạy, Windows Defender Firewall sẽ hỏi; bấm **Allow access**.

Trên **máy điều phối**:

1. Chép `machines.example.json` thành `machines.json`.
2. Điền tên và địa chỉ của từng máy worker:
   ```json
   { "name": "may-a", "address": "192.168.1.21", "port": 8765 }
   ```
3. Chạy `python controller.py`.

Mọi máy phải dùng **cùng một `token`**. Mặc định là `dicom-demo`; đổi bằng
`--token` ở agent và trường `token` trong `machines.json`.

## Cách hoạt động

```
Máy điều phối                         Máy worker
  controller.py                         agent.py
      |                                    |
      |  POST /request  (JSON, kèm token)  |
      |----------------------------------->|  đưa vào hàng đợi
      |                                    |  hiện hộp thoại cho chủ máy
      |        (giữ kết nối, chờ)          |  chủ máy bấm nút
      |<-----------------------------------|
      |  { "decision": "allow" | "deny" }  |  ghi vào agent-log.jsonl
```

- Máy điều phối chỉ **gửi yêu cầu**; quyền quyết định nằm hoàn toàn ở chủ máy.
- Không trả lời trong 120 giây thì agent **tự động từ chối**.
- Đóng cửa sổ hộp thoại, hoặc đóng agent, cũng tính là từ chối, nên máy điều phối
  không bị treo chờ.
- Mọi quyết định được ghi vào `agent-log.jsonl` trên máy worker.

## Giới hạn

| Việc | Trạng thái |
|---|---|
| Hộp thoại xin phép, trả lời về máy điều phối | Có |
| Ghi nhật ký quyết định | Có |
| Tự động từ chối khi hết thời gian chờ | Có |
| Giới hạn CPU/RAM/VRAM thật | **Chưa** |
| Chạy công việc huấn luyện sau khi được cho phép | **Chưa** |
| Thu hồi GPU khi chủ máy cần dùng lại | **Chưa** |
| Xác thực đúng nghĩa | **Chưa**: token chỉ là chuỗi chia sẻ, gửi qua HTTP không mã hoá |

Vì chưa có xác thực thật, **chỉ chạy demo trong mạng tin cậy**, và tắt agent khi
dùng xong. Bất kỳ ai biết token và địa chỉ đều có thể làm hiện hộp thoại.

## Kiểm thử

```bash
python -m unittest discover -s demo -v
```

13 test, chạy không cần người bấm nút: chúng gọi thẳng vào nút của hộp thoại.
Có kiểm tra cả vòng đầy đủ (controller gửi, hộp thoại hiện, bấm nút, câu trả lời
quay về) cũng như token sai, JSON hỏng và ghi nhật ký. Cần có màn hình, không chạy
được trên máy chỉ có dòng lệnh.
