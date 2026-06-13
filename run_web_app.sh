#!/bin/bash
# Chuyển vào thư mục chứa script này
cd "$(dirname "$0")"

# Kích hoạt môi trường ảo
source tts_env/bin/activate

# Chạy ứng dụng web
python web_app.py
