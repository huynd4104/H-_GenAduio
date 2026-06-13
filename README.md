# KhanhTTS Voice Cloning Studio (H-_GenAudio)

Ứng dụng Web cục bộ (Local Web App) giúp nhân bản giọng nói và tạo âm thanh hàng loạt trực quan, sử dụng mô hình **OmniVoice** và mô hình **KhanhTTS**.

---

## 🌟 Tính năng chính

- **Nhân bản giọng nói (Voice Cloning):** Tự động bắt chước giọng nói từ một file âm thanh mẫu (Reference Audio).
- **Nhận diện nội dung mẫu:** Tự động transcribe (chuyển giọng nói thành văn bản) nội dung của file âm thanh mẫu.
- **Tạo âm thanh hàng loạt:** Nhập danh sách văn bản (mỗi dòng một câu) để tạo ra hàng loạt file âm thanh tương ứng cùng lúc.
- **Quản lý âm thanh tích hợp:** Nghe thử trực tiếp trên trình duyệt, tìm kiếm nhanh và xóa file dễ dàng.
- **Hỗ trợ điều khiển:** Tạm dừng hoặc dừng hẳn quá trình tạo âm thanh một cách linh hoạt.

---

## 📋 Yêu cầu hệ thống

1. **Python 3.10+**
2. **FFmpeg:** Cần thiết cho thư viện `pydub` để xử lý định dạng âm thanh (`.mp3`, `.wav`).
3. **Node.js** (Tùy chọn, để chạy script nhanh qua `npm`).

---

## 🛠️ Hướng dẫn cài đặt

### Bước 1: Chuẩn bị mã nguồn và mô hình
1. Clone hoặc tải dự án này về máy của bạn.
2. Tải mô hình **KhanhTTS-OmniVoice** từ Hugging Face vào thư mục `khanhtts_model` bằng một trong hai cách dưới đây:

#### Cách A: Sử dụng `huggingface-cli` (Khuyên dùng - nhanh và đơn giản)
Cách này không cần cài đặt Git LFS. Bạn có thể thực hiện sau khi tạo môi trường ảo ở Bước 3, hoặc chạy trực tiếp bằng python:
```bash
# Cài đặt thư viện tải từ Hugging Face
pip install huggingface_hub

# Tải toàn bộ mô hình về thư mục khanhtts_model
huggingface-cli download kjanh/KhanhTTS-OmniVoice --local-dir khanhtts_model
```

#### Cách B: Sử dụng `git clone` (Yêu cầu cài đặt Git LFS)
Nếu bạn thích dùng Git, hãy đảm bảo đã cài đặt [Git LFS](https://git-lfs.com/):
```bash
# Khởi tạo Git LFS (chỉ cần làm một lần)
git lfs install

# Clone repository của mô hình vào thư mục khanhtts_model
git clone https://huggingface.co/kjanh/KhanhTTS-OmniVoice khanhtts_model
```

> [!NOTE]
> Thư mục `khanhtts_model` chứa các file trọng số mô hình lớn (~3GB) nên mặc định đã bị bỏ qua (ignored) trong file `.gitignore` để tránh đẩy lên GitHub của bạn.

### Bước 2: Cài đặt FFmpeg
- **macOS** (Sử dụng Homebrew):
  ```bash
  brew install ffmpeg
  ```
- **Windows**: Tải bản build FFmpeg từ trang chủ, giải nén và thêm đường dẫn thư mục `bin` vào Environment Variables (PATH).
- **Ubuntu/Linux**:
  ```bash
  sudo apt update && sudo apt install ffmpeg
  ```

### Bước 3: Thiết lập môi trường ảo Python
Mở Terminal hoặc Command Prompt tại thư mục dự án và chạy các lệnh sau:

```bash
# 1. Tạo môi trường ảo
python3 -m venv tts_env

# 2. Kích hoạt môi trường ảo
# Trên macOS / Linux:
source tts_env/bin/activate

# Trên Windows:
tts_env\Scripts\activate

# 3. Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

---

## 🚀 Hướng dẫn khởi chạy ứng dụng

Khi môi trường và mô hình đã sẵn sàng, bạn có thể khởi chạy ứng dụng theo một trong các cách sau:

### Cách 1: Sử dụng Shell Script (Chỉ dành cho macOS / Linux)
Chạy script tự động kích hoạt môi trường ảo và khởi động server:
```bash
chmod +x run_web_app.sh
./run_web_app.sh
```

### Cách 2: Sử dụng NPM (Nếu máy đã cài Node.js)
```bash
npm run dev
```

### Cách 3: Chạy trực tiếp bằng Python
```bash
# Đảm bảo bạn đã activate môi trường ảo tts_env trước đó
python web_app.py
```

Sau khi chạy thành công, mở trình duyệt và truy cập địa chỉ:
👉 **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

---

## 📖 Hướng dẫn sử dụng cho người mới

1. **Nhập văn bản cần tạo:** Nhập hoặc dán các câu bạn muốn tạo vào ô *"Danh sách câu nói cần gen"*. Hãy xuống dòng cho mỗi câu (mỗi câu sẽ tạo ra một file `.wav` riêng).
2. **Chọn thư mục lưu kết quả:** 
   - Bạn có thể nhấn nút **🔍 Finder...** để chọn một thư mục bất kỳ trên máy của mình.
   - Hoặc nhập đường dẫn tuyệt đối trực tiếp vào ô nhập đường dẫn.
3. **Thiết lập âm thanh mẫu (Voice Reference):**
   - Chọn một trong các file mẫu có sẵn trong menu thả xuống.
   - Hoặc tải lên file mẫu mới bằng nút **📁 Chọn tệp mẫu**.
   - Kiểm tra xem ô **"Nội dung chính xác của file mẫu"** có khớp với những gì giọng mẫu nói hay không (hệ thống sẽ tự nhận dạng trước, bạn có thể chỉnh sửa lại nếu cần).
4. **Bắt đầu tạo âm thanh:** Nhấn **🚀 Bắt đầu tạo hàng loạt** và theo dõi thanh tiến trình.
5. **Quản lý file:** Các file âm thanh được tạo ra sẽ hiển thị ở bảng bên phải. Bạn có thể nhấn nút phát để nghe thử trực tiếp hoặc nhấn nút **Xóa** để xóa file.
