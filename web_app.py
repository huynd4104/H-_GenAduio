import os
import re
import json
import asyncio
import torch
import soundfile as sf
import subprocess
from typing import Generator
from fastapi import FastAPI, Request, Form, HTTPException, File, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydub import AudioSegment
from omnivoice import OmniVoice

app = FastAPI(title="KhanhTTS OmniVoice Studio")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "./khanhtts_model"
model = None

def get_model():
    global model
    if model is None:
        print("🤖 Loading OmniVoice model...")
        model = OmniVoice.from_pretrained(MODEL_NAME, device_map="cpu")
    return model

def clean_filename(text):
    text = text.strip()
    cleaned = re.sub(r'[\\/*?:"<>|.]', "", text)
    cleaned = "_".join(cleaned.split())
    return cleaned[:50]

# UI Template
HTML_CONTENT = """
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>KhanhTTS Voice Cloning Studio</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
            --glass-bg: rgba(255, 255, 255, 0.03);
            --glass-border: rgba(255, 255, 255, 0.08);
            --glass-focus: rgba(255, 255, 255, 0.15);
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --accent: #6366f1;
            --accent-glow: rgba(99, 102, 241, 0.4);
            --success: #10b981;
            --danger: #ef4444;
            --panel-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Plus Jakarta Sans', sans-serif;
            background: var(--bg-gradient);
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 2rem 1rem;
            overflow-x: hidden;
        }

        .container {
            width: 100%;
            max-width: 1200px;
            display: grid;
            grid-template-columns: 1.2fr 0.8fr;
            gap: 2rem;
            margin-top: 1.5rem;
        }

        @media (max-width: 900px) {
            .container {
                grid-template-columns: 1fr;
            }
        }

        header {
            text-align: center;
            margin-bottom: 1rem;
            width: 100%;
        }

        header h1 {
            font-family: 'Outfit', sans-serif;
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(to right, #818cf8, #c084fc, #f472b6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
            letter-spacing: -0.05em;
        }

        header p {
            color: var(--text-secondary);
            font-size: 1rem;
        }

        .panel {
            background: var(--glass-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--glass-border);
            border-radius: 16px;
            padding: 2rem;
            box-shadow: var(--panel-shadow);
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }

        .panel:hover {
            border-color: rgba(255, 255, 255, 0.12);
        }

        .form-group {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        label {
            font-size: 0.875rem;
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        input[type="text"], textarea, select {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid var(--glass-border);
            border-radius: 8px;
            padding: 0.75rem 1rem;
            color: var(--text-primary);
            font-family: inherit;
            font-size: 0.95rem;
            outline: none;
            transition: all 0.2s ease;
            width: 100%;
        }

        input[type="text"]:focus, textarea:focus, select:focus {
            border-color: var(--accent);
            box-shadow: 0 0 0 3px var(--accent-glow);
            background: rgba(15, 23, 42, 0.8);
        }

        textarea {
            resize: vertical;
            min-height: 120px;
        }

        .row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
        }

        @media (max-width: 600px) {
            .row {
                grid-template-columns: 1fr;
            }
        }

        .btn {
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
            border: none;
            border-radius: 8px;
            color: white;
            padding: 1rem;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
        }

        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(79, 70, 229, 0.5);
            background: linear-gradient(135deg, #5a52e6 0%, #8b4bf0 100%);
        }

        .btn-secondary {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--glass-border);
            color: var(--text-primary);
            padding: 0.75rem 1rem;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 500;
            font-size: 0.9rem;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.35rem;
        }

        .btn-secondary:hover {
            background: rgba(255, 255, 255, 0.1);
            border-color: rgba(255, 255, 255, 0.2);
        }

        .upload-area {
            display: flex;
            gap: 0.5rem;
            width: 100%;
        }
        
        .upload-area input[type="file"] {
            display: none;
        }

        /* Progress Visualizer */
        .progress-container {
            display: none;
            background: rgba(15, 23, 42, 0.4);
            border-radius: 12px;
            padding: 1.25rem;
            border: 1px solid var(--glass-border);
        }

        .progress-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 0.75rem;
            font-size: 0.9rem;
        }

        .progress-bar-bg {
            background: rgba(255, 255, 255, 0.05);
            height: 8px;
            border-radius: 4px;
            overflow: hidden;
            width: 100%;
        }

        .progress-bar-fill {
            background: linear-gradient(to right, #6366f1, #a855f7);
            height: 100%;
            width: 0%;
            transition: width 0.3s ease;
        }

        .status-text {
            margin-top: 0.5rem;
            font-size: 0.85rem;
            color: var(--text-secondary);
            font-style: italic;
        }

        /* Audio Manager & File List */
        .file-list {
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
            max-height: 520px;
            overflow-y: auto;
            padding-right: 0.25rem;
        }

        .file-list::-webkit-scrollbar {
            width: 6px;
        }

        .file-list::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.02);
        }

        .file-list::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 3px;
        }

        .file-item {
            background: rgba(255, 255, 255, 0.02);
            border: 1px solid var(--glass-border);
            border-radius: 8px;
            padding: 0.75rem 1rem;
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
            transition: all 0.2s ease;
        }

        .file-item:hover {
            background: rgba(255, 255, 255, 0.04);
            border-color: rgba(255, 255, 255, 0.15);
        }

        .file-info {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .file-name {
            font-size: 0.9rem;
            font-weight: 500;
            color: var(--text-primary);
            word-break: break-all;
            margin-right: 0.5rem;
        }

        .file-actions {
            display: flex;
            gap: 0.5rem;
            align-items: center;
        }

        .btn-sm-danger {
            background: transparent;
            border: 1px solid rgba(239, 68, 68, 0.3);
            color: var(--danger);
            border-radius: 4px;
            padding: 0.25rem 0.5rem;
            font-size: 0.75rem;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .btn-sm-danger:hover {
            background: var(--danger);
            color: white;
            border-color: var(--danger);
        }

        audio {
            width: 100%;
            height: 32px;
            border-radius: 4px;
            outline: none;
        }

        audio::-webkit-media-controls-panel {
            background-color: rgba(30, 27, 75, 0.85);
        }
        audio::-webkit-media-controls-current-time-display,
        audio::-webkit-media-controls-time-remaining-display {
            color: var(--text-primary);
        }

        .empty-state {
            text-align: center;
            color: var(--text-secondary);
            padding: 2rem;
            font-size: 0.9rem;
            border: 1px dashed var(--glass-border);
            border-radius: 8px;
        }
        
        .manager-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        /* Dialog Modal Styles */
        dialog {
            background: rgba(15, 23, 42, 0.95);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--glass-border);
            border-radius: 16px;
            padding: 2rem;
            box-shadow: var(--panel-shadow);
            color: var(--text-primary);
            max-width: 400px;
            width: 90%;
            margin: auto;
            border: 1px solid var(--glass-border);
            outline: none;
        }

        dialog::backdrop {
            background: rgba(0, 0, 0, 0.6);
            backdrop-filter: blur(4px);
        }

        .dialog-header {
            font-size: 1.25rem;
            font-weight: 700;
            margin-bottom: 1rem;
            color: #f472b6;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-family: 'Outfit', sans-serif;
        }

        .dialog-body {
            color: var(--text-secondary);
            font-size: 0.95rem;
            line-height: 1.5;
            margin-bottom: 1.5rem;
        }

        .dialog-actions {
            display: flex;
            justify-content: flex-end;
            gap: 1rem;
        }

        .btn-danger {
            background: linear-gradient(135deg, #ef4444 0%, #b91c1c 100%);
            border: none;
            border-radius: 8px;
            color: white;
            padding: 0.75rem 1.2rem;
            font-size: 0.95rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 12px rgba(239, 68, 68, 0.3);
        }

        .btn-danger:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(239, 68, 68, 0.5);
        }

        .control-group {
            display: none;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
            width: 100%;
        }
    </style>
</head>
<body>
    <header>
        <h1>KhanhTTS OmniVoice Studio</h1>
        <p>Ứng dụng nhân bản giọng nói và tạo âm thanh hàng loạt trực quan</p>
    </header>

    <div class="container">
        <div class="panel">
            <div class="form-group">
                <label for="sentences">Danh sách câu nói cần gen (Mỗi câu một dòng)</label>
                <textarea id="sentences" placeholder="Ví dụ:&#10;Chưa đúng rồi, con chọn lại nhé.&#10;Tuyệt vời! Con giỏi quá."></textarea>
            </div>

            <div class="form-group">
                <label for="folder-select">Thư mục lưu kết quả</label>
                <select id="folder-select" style="margin-bottom: 0.5rem;">
                    <option value="ROOT">Chọn thư mục có sẵn...</option>
                </select>
                <div style="display: flex; gap: 0.5rem; width: 100%;">
                    <input type="text" id="folder-path" placeholder="Đường dẫn thư mục lưu..." style="flex: 1;">
                    <button class="btn-secondary" id="btn-browse-folder" style="white-space: nowrap; min-width: 110px;">🔍 Finder...</button>
                </div>
            </div>

            <div class="row">
                <div class="form-group">
                    <label for="sample-select">File mẫu (Reference Audio)</label>
                    <select id="sample-select" style="margin-bottom: 0.5rem;">
                        <!-- Loaded dynamically -->
                    </select>
                    <!-- Audio player để nghe thử file mẫu -->
                    <audio id="sample-audio-player" controls style="height: 32px; width: 100%;"></audio>
                </div>
                <div class="form-group">
                    <label>Tải lên file mẫu mới</label>
                    <div class="upload-area">
                        <button class="btn-secondary" onclick="document.getElementById('file-uploader').click()" style="width: 100%; white-space: nowrap; margin-bottom: 0.5rem;">📁 Chọn tệp mẫu</button>
                        <input type="file" id="file-uploader" accept=".mp3,.wav" onchange="uploadSampleFile(this)">
                    </div>
                </div>
            </div>

            <div class="form-group">
                <label for="ref-text">Nội dung chính xác của file mẫu</label>
                <input type="text" id="ref-text" value="Tuyệt vời, đây là số 3 nhé!">
                <span id="upload-status" style="font-size: 0.85rem; color: var(--success); min-height: 1.2rem; display: block; margin-top: -0.25rem;"></span>
            </div>

            <div class="form-group">
                <label>Tốc độ đọc: <span id="speed-val">1.00</span>x</label>
                <div style="display: flex; align-items: center; height: 100%; padding-top: 0.25rem;">
                    <input type="range" id="speed-slider" min="0.5" max="2.0" step="0.05" value="1.0" style="width: 100%; accent-color: var(--accent);">
                </div>
            </div>

            <div class="row" style="grid-template-columns: 1.2fr 0.8fr; align-items: center; gap: 1rem; margin-top: -0.5rem;">
                <div class="form-group" style="flex-direction: row; align-items: center; gap: 0.5rem; height: 100%;">
                    <input type="checkbox" id="enable-numbering" style="width: 1.2rem; height: 1.2rem; accent-color: var(--accent); cursor: pointer;">
                    <label for="enable-numbering" style="cursor: pointer; margin: 0; text-transform: none; font-size: 0.85rem; font-weight: 500;">Đánh số thứ tự file</label>
                </div>
                <div class="form-group" id="start-number-group" style="display: none; height: 100%;">
                    <label for="start-number" style="font-size: 0.75rem;">Số bắt đầu</label>
                    <input type="number" id="start-number" value="1" min="0" style="background: rgba(15, 23, 42, 0.6); border: 1px solid var(--glass-border); border-radius: 8px; padding: 0.5rem 0.75rem; color: var(--text-primary); outline: none; width: 100%;">
                </div>
            </div>

            <button id="btn-generate" class="btn">
                <span>🚀 Bắt đầu tạo hàng loạt</span>
            </button>

            <!-- Control buttons during generation -->
            <div id="control-group" class="control-group">
                <button id="btn-pause" class="btn" style="background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); box-shadow: 0 4px 12px rgba(245, 158, 11, 0.3);">
                    ⏸️ Tạm dừng
                </button>
                <button id="btn-stop" class="btn" style="background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); box-shadow: 0 4px 12px rgba(239, 68, 68, 0.3);">
                    ⏹️ Dừng hẳn
                </button>
            </div>

            <!-- Progress Visualizer -->
            <div id="progress-container" class="progress-container">
                <div class="progress-header">
                    <span id="progress-label">Đang chuẩn bị...</span>
                    <span id="progress-percentage">0%</span>
                </div>
                <div class="progress-bar-bg">
                    <div id="progress-bar-fill" class="progress-bar-fill"></div>
                </div>
                <div id="status-text" class="status-text">Đang khởi tạo mô hình...</div>
            </div>
        </div>

        <!-- Panel bên phải: Quản lý Audio -->
        <div class="panel">
            <div class="form-group">
                <div class="manager-header">
                    <label>Các file đã tạo trong thư mục</label>
                    <div style="display: flex; gap: 0.5rem;">
                        <button class="btn-secondary" style="padding: 0.4rem 0.8rem; font-size: 0.8rem;" onclick="loadFiles()">Reset</button>
                        <button id="btn-clear-all" class="btn-sm-danger" style="padding: 0.4rem 0.8rem; font-size: 0.8rem; border-color: rgba(239, 68, 68, 0.4);">Delete</button>
                    </div>
                </div>
                <input type="text" id="search-files" placeholder="🔍 Tìm kiếm file nhanh..." style="margin-top: 0.5rem; margin-bottom: 0.5rem; padding: 0.5rem 0.75rem; font-size: 0.85rem;">
                <div id="file-list-container" class="file-list">
                    <div class="empty-state">Nhập thư mục để tải danh sách file âm thanh.</div>
                </div>
            </div>
        </div>
    </div>

    <!-- Dialog xác nhận dừng -->
    <dialog id="confirm-dialog">
        <div class="dialog-header">
            ⚠️ Xác nhận dừng
        </div>
        <div class="dialog-body">
            Bạn có chắc chắn muốn dừng hẳn quá trình tạo âm thanh không? Câu đang chạy dở sẽ được hoàn thành trước khi dừng hẳn.
        </div>
        <div class="dialog-actions">
            <button id="btn-confirm-no" class="btn-secondary">Hủy</button>
            <button id="btn-confirm-yes" class="btn-danger">Dừng hẳn</button>
        </div>
    </dialog>

    <script>
        const sentencesInput = document.getElementById('sentences');
        const folderSelect = document.getElementById('folder-select');
        const folderPathInput = document.getElementById('folder-path');
        const sampleSelect = document.getElementById('sample-select');
        const refTextInput = document.getElementById('ref-text');
        const btnGenerate = document.getElementById('btn-generate');
        const progressContainer = document.getElementById('progress-container');
        const progressLabel = document.getElementById('progress-label');
        const progressPercentage = document.getElementById('progress-percentage');
        const progressBarFill = document.getElementById('progress-bar-fill');
        const statusText = document.getElementById('status-text');
        const fileListContainer = document.getElementById('file-list-container');
        const uploadStatus = document.getElementById('upload-status');
        const btnBrowseFolder = document.getElementById('btn-browse-folder');
        const sampleAudioPlayer = document.getElementById('sample-audio-player');

        const controlGroup = document.getElementById('control-group');
        const btnPause = document.getElementById('btn-pause');
        const btnStop = document.getElementById('btn-stop');
        const confirmDialog = document.getElementById('confirm-dialog');
        const btnConfirmNo = document.getElementById('btn-confirm-no');
        const btnConfirmYes = document.getElementById('btn-confirm-yes');
        const speedSlider = document.getElementById('speed-slider');
        const speedVal = document.getElementById('speed-val');
        const searchFiles = document.getElementById('search-files');
        const btnClearAll = document.getElementById('btn-clear-all');
        const enableNumberingCheckbox = document.getElementById('enable-numbering');
        const startNumberGroup = document.getElementById('start-number-group');
        const startNumberInput = document.getElementById('start-number');

        speedSlider.oninput = () => {
            speedVal.textContent = parseFloat(speedSlider.value).toFixed(2);
        };

        enableNumberingCheckbox.onchange = () => {
            if (enableNumberingCheckbox.checked) {
                startNumberGroup.style.display = 'flex';
            } else {
                startNumberGroup.style.display = 'none';
            }
        };

        btnClearAll.onclick = async () => {
            const folder = folderPathInput.value.trim();
            if (!folder) return;
            if (confirm("⚠️ Bạn có chắc chắn muốn XÓA TẤT CẢ các file âm thanh .wav trong thư mục này không? Thao tác này không thể hoàn tác.")) {
                try {
                    const res = await fetch(`/api/clear-all?folder=${encodeURIComponent(folder)}`, { method: 'DELETE' });
                    const result = await res.json();
                    if (result.success) {
                        loadFiles();
                    } else {
                        alert("Lỗi khi xóa tất cả: " + result.error);
                    }
                } catch (err) {
                    alert("Lỗi kết nối: " + err);
                }
            }
        };

        let loadedFiles = [];
        let defaultBaseDir = "";
        let sentenceList = [];
        let currentIndex = 0;
        let totalSentences = 0;
        let isPaused = false;
        let isStopped = false;

        // Load initial folders and samples list
        async function loadMetadata() {
            try {
                const res = await fetch('/api/metadata');
                const data = await res.json();
                
                defaultBaseDir = data.base_dir;
                
                // Load preset folders list
                folderSelect.innerHTML = '';
                data.folders.forEach(folder => {
                    const opt = document.createElement('option');
                    opt.value = folder.path;
                    opt.textContent = folder.name;
                    folderSelect.appendChild(opt);
                });

                // Set default folder path to 'Kết quả' folder if empty and exists
                if(!folderPathInput.value) {
                    const ketQuaFolder = data.folders.find(f => f.name === "Kết quả");
                    if (ketQuaFolder) {
                        folderPathInput.value = ketQuaFolder.path;
                        folderSelect.value = ketQuaFolder.path;
                    } else if (data.folders.length > 0) {
                        folderPathInput.value = data.folders[0].path;
                        folderSelect.value = data.folders[0].path;
                    } else {
                        folderPathInput.value = defaultBaseDir;
                    }
                }

                // Load samples
                const currentSample = sampleSelect.value;
                sampleSelect.innerHTML = '';
                data.samples.forEach(sample => {
                    const opt = document.createElement('option');
                    opt.value = sample.path;
                    opt.textContent = sample.name;
                    if (sample.name === 'sample.mp3') opt.selected = true;
                    sampleSelect.appendChild(opt);
                });
                
                if (currentSample) {
                    sampleSelect.value = currentSample;
                }
                
                // Cập nhật nguồn âm thanh mẫu ban đầu
                updateSampleAudio();
            } catch (err) {
                console.error("Lỗi khi tải metadata:", err);
            }
        }

        async function transcribeSample(path) {
            refTextInput.value = "";
            refTextInput.placeholder = "⌛ Đang tự động nhận diện nội dung giọng nói mẫu...";
            refTextInput.disabled = true;
            try {
                const res = await fetch("/api/transcribe", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ path: path })
                });
                const data = await res.json();
                if (data.success && data.text) {
                    refTextInput.value = data.text;
                } else {
                    refTextInput.placeholder = "Nhập nội dung tương ứng của file mẫu...";
                }
            } catch (err) {
                console.error("Lỗi nhận diện giọng nói:", err);
                refTextInput.placeholder = "Nhập nội dung tương ứng của file mẫu...";
            } finally {
                refTextInput.disabled = false;
            }
        }

        function updateSampleAudio() {
            const selectedPath = sampleSelect.value;
            if (selectedPath) {
                sampleAudioPlayer.src = `/api/audio?path=${encodeURIComponent(selectedPath)}`;
                sampleAudioPlayer.style.display = 'block';
                transcribeSample(selectedPath);
            } else {
                sampleAudioPlayer.style.display = 'none';
            }
        }

        // Dropdown selection updates text input
        folderSelect.onchange = () => {
            folderPathInput.value = folderSelect.value;
            loadFiles();
        };

        // Custom path input change reloads files
        folderPathInput.onchange = loadFiles;
        
        // Cập nhật audio player khi thay đổi tệp mẫu
        sampleSelect.onchange = updateSampleAudio;

        btnBrowseFolder.onclick = async () => {
            btnBrowseFolder.disabled = true;
            btnBrowseFolder.textContent = "⌛ Đang chọn...";
            try {
                const res = await fetch('/api/browse-folder', { method: 'POST' });
                const data = await res.json();
                if (data.path) {
                    folderPathInput.value = data.path;
                    loadFiles();
                } else if (data.error) {
                    alert("Lỗi khi chọn thư mục: " + data.error);
                }
            } catch (err) {
                console.error("Lỗi kết nối:", err);
            } finally {
                btnBrowseFolder.disabled = false;
                btnBrowseFolder.textContent = "🔍 Finder...";
            }
        };

        async function loadFiles() {
            const folder = folderPathInput.value.trim();
            if (!folder) return;
            try {
                const res = await fetch(`/api/files?folder=${encodeURIComponent(folder)}`);
                const data = await res.json();
                
                if (data.error) {
                    fileListContainer.innerHTML = `<div class="empty-state" style="color:var(--danger)">Không thể đọc thư mục: ${data.error}</div>`;
                    return;
                }

                loadedFiles = data.files || [];
                renderFilesList(searchFiles.value.trim());
            } catch (err) {
                console.error("Lỗi khi tải file:", err);
            }
        }

        function renderFilesList(filterText = "") {
            const query = filterText.toLowerCase();
            const filtered = loadedFiles.filter(file => file.name.toLowerCase().includes(query));

            if (filtered.length === 0) {
                fileListContainer.innerHTML = loadedFiles.length === 0 
                    ? '<div class="empty-state">Thư mục này chưa có file âm thanh .wav nào.</div>'
                    : '<div class="empty-state">Không tìm thấy file nào khớp với tìm kiếm.</div>';
                return;
            }

            fileListContainer.innerHTML = '';
            filtered.forEach(file => {
                const item = document.createElement('div');
                item.className = 'file-item';
                
                const info = document.createElement('div');
                info.className = 'file-info';
                
                const name = document.createElement('span');
                name.className = 'file-name';
                name.textContent = file.name;
                
                const actions = document.createElement('div');
                actions.className = 'file-actions';
                
                const deleteBtn = document.createElement('button');
                deleteBtn.className = 'btn-sm-danger';
                deleteBtn.textContent = 'Xóa';
                deleteBtn.onclick = () => deleteFile(file.path);
                
                actions.appendChild(deleteBtn);
                info.appendChild(name);
                info.appendChild(actions);
                
                const audio = document.createElement('audio');
                audio.controls = true;
                audio.preload = 'metadata';
                audio.src = `/api/audio?path=${encodeURIComponent(file.path)}`;
                
                item.appendChild(info);
                item.appendChild(audio);
                fileListContainer.appendChild(item);
            });
        }

        searchFiles.oninput = (e) => {
            renderFilesList(e.target.value.trim());
        };

        async function deleteFile(path) {
            if (!confirm(`Bạn có chắc muốn xóa file này?`)) return;
            try {
                const res = await fetch(`/api/delete-audio?path=${encodeURIComponent(path)}`, { method: 'DELETE' });
                const result = await res.json();
                if (result.success) {
                    loadFiles();
                } else {
                    alert("Không thể xóa file: " + result.error);
                }
            } catch (err) {
                alert("Lỗi khi xóa file: " + err);
            }
        }

        async function uploadSampleFile(input) {
            if (!input.files || input.files.length === 0) return;
            const file = input.files[0];
            const formData = new FormData();
            formData.append("file", file);

            uploadStatus.style.color = "var(--text-secondary)";
            uploadStatus.textContent = "Đang tải lên...";

            try {
                const res = await fetch("/api/upload-sample", {
                    method: "POST",
                    body: formData
                });
                const data = await res.json();
                if (data.success) {
                    uploadStatus.style.color = "var(--success)";
                    uploadStatus.textContent = `Đã tải lên: ${data.name}`;
                    
                    await loadMetadata();
                    sampleSelect.value = data.path;
                    updateSampleAudio();
                } else {
                    uploadStatus.style.color = "var(--danger)";
                    uploadStatus.textContent = "Lỗi: " + data.error;
                }
            } catch (err) {
                uploadStatus.style.color = "var(--danger)";
                uploadStatus.textContent = "Lỗi kết nối";
            }
        }

        function resetUI() {
            btnGenerate.style.display = 'flex';
            controlGroup.style.display = 'none';
            btnPause.disabled = false;
            btnStop.disabled = false;
            btnPause.innerHTML = '⏸️ Tạm dừng';
            btnStop.innerHTML = '⏹️ Dừng hẳn';
        }

        async function processNextSentence() {
            if (isStopped) {
                statusText.textContent = '⛔ Quá trình đã bị dừng bởi người dùng.';
                resetUI();
                loadFiles();
                return;
            }

            if (isPaused) {
                statusText.textContent = '⏸️ Đang tạm dừng. Nhấp "Tiếp tục" để chạy tiếp.';
                btnPause.innerHTML = '▶️ Tiếp tục';
                btnPause.disabled = false;
                return;
            }

            if (currentIndex >= totalSentences) {
                progressBarFill.style.width = '100%';
                progressPercentage.textContent = '100%';
                progressLabel.textContent = 'Hoàn thành!';
                statusText.textContent = `Tạo thành công toàn bộ ${totalSentences} câu nói.`;
                resetUI();
                loadFiles();
                return;
            }

            const currentText = sentenceList[currentIndex];
            const percent = Math.round((currentIndex) / totalSentences * 100);
            
            progressBarFill.style.width = `${percent}%`;
            progressPercentage.textContent = `${percent}%`;
            progressLabel.textContent = `Đang xử lý câu ${currentIndex + 1}/${totalSentences}`;
            statusText.textContent = `Chuẩn bị sinh giọng cho: "${currentText}"`;

            const payload = {
                sentences: currentText,
                folder: folderPathInput.value.trim(),
                ref_audio: sampleSelect.value,
                ref_text: refTextInput.value,
                speed: parseFloat(speedSlider.value),
                enable_numbering: enableNumberingCheckbox.checked,
                start_number: enableNumberingCheckbox.checked ? parseInt(startNumberInput.value) + currentIndex : 1
            };

            try {
                const response = await fetch('/api/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                const reader = response.body.getReader();
                const decoder = new TextDecoder("utf-8");
                let buffer = "";
                let hasError = false;

                while (true) {
                    const { value, done } = await reader.read();
                    if (done) break;

                    buffer += decoder.decode(value, { stream: true });
                    const lines = buffer.split("\\n");
                    buffer = lines.pop();

                    for (const line of lines) {
                        if (!line.trim()) continue;
                        try {
                            const data = JSON.parse(line);
                            if (data.status === 'model_loading') {
                                statusText.textContent = "🤖 Đang khởi tạo mô hình KhanhTTS-OmniVoice...";
                            } else if (data.status === 'processing') {
                                statusText.textContent = `Đang sinh giọng cho: "${currentText}"`;
                            } else if (data.status === 'completed') {
                                // Câu này hoàn thành
                            } else if (data.status === 'error') {
                                statusText.textContent = `❌ Lỗi ở câu: "${currentText}": ${data.message}`;
                                hasError = true;
                            }
                        } catch (e) {
                            console.error("Parse error:", e, line);
                        }
                    }
                }

                if (hasError) {
                    resetUI();
                    return;
                }

                currentIndex++;
                const nextPercent = Math.round((currentIndex) / totalSentences * 100);
                progressBarFill.style.width = `${nextPercent}%`;
                progressPercentage.textContent = `${nextPercent}%`;

                processNextSentence();
            } catch (err) {
                statusText.textContent = `❌ Lỗi kết nối: ${err}`;
                resetUI();
            }
        }

        btnGenerate.onclick = async () => {
            const rawSentences = sentencesInput.value.trim();
            if (!rawSentences) {
                alert("Vui lòng nhập ít nhất một câu để tạo giọng nói.");
                return;
            }

            const folderPath = folderPathInput.value.trim();
            if (!folderPath) {
                alert("Vui lòng nhập đường dẫn thư mục lưu kết quả.");
                return;
            }

            sentenceList = rawSentences.split("\\n").map(s => s.trim()).filter(s => s);
            totalSentences = sentenceList.length;
            currentIndex = 0;
            isPaused = false;
            isStopped = false;

            btnGenerate.style.display = 'none';
            controlGroup.style.display = 'grid';
            btnPause.innerHTML = '⏸️ Tạm dừng';

            progressContainer.style.display = 'block';
            progressBarFill.style.width = '0%';
            progressPercentage.textContent = '0%';
            progressLabel.textContent = 'Đang chuẩn bị...';
            statusText.textContent = 'Bắt đầu quá trình tạo hàng loạt...';

            processNextSentence();
        };

        btnPause.onclick = () => {
            if (isPaused) {
                isPaused = false;
                btnPause.innerHTML = '⏸️ Tạm dừng';
                statusText.textContent = 'Đang tiếp tục quá trình tạo...';
                processNextSentence();
            } else {
                isPaused = true;
                btnPause.disabled = true;
                btnPause.innerHTML = '⌛ Đang dừng...';
                statusText.textContent = 'Đang đợi hoàn thành câu hiện tại để tạm dừng...';
            }
        };

        btnStop.onclick = () => {
            confirmDialog.showModal();
        };

        btnConfirmNo.onclick = () => {
            confirmDialog.close();
        };

        btnConfirmYes.onclick = () => {
            confirmDialog.close();
            isStopped = true;
            if (isPaused) {
                resetUI();
                statusText.textContent = '⛔ Đang dừng hẳn...';
                loadFiles();
            } else {
                btnStop.disabled = true;
                btnStop.innerHTML = '⌛ Đang dừng...';
                statusText.textContent = 'Đang đợi hoàn thành câu hiện tại để dừng hẳn...';
            }
        };

        loadMetadata().then(loadFiles);
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_CONTENT

@app.post("/api/browse-folder")
async def browse_folder():
    try:
        cmd = """osascript -e 'POSIX path of (choose folder with prompt "Chọn thư mục lưu kết quả:")'"""
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode == 0:
            selected_path = stdout.decode('utf-8').strip()
            return {"path": selected_path}
        else:
            return {"path": None, "message": "User cancelled or error occurred."}
    except Exception as e:
        return {"path": None, "error": str(e)}

@app.get("/api/metadata")
async def get_metadata():
    folders = []
    samples = []
    
    for item in os.listdir(BASE_DIR):
        full_path = os.path.join(BASE_DIR, item)
        if os.path.isdir(full_path) and not item.startswith(".") and item not in ["tts_env", "__pycache__", "khanhtts_model"]:
            folders.append({
                "name": item,
                "path": full_path
            })
            
    sample_dir = os.path.join(BASE_DIR, "Audio sample")
    if os.path.exists(sample_dir):
        for item in os.listdir(sample_dir):
            full_path = os.path.join(sample_dir, item)
            if os.path.isfile(full_path) and (item.lower().endswith(".mp3") or item.lower().endswith(".wav")) and not item.startswith("temp_"):
                samples.append({
                    "name": item,
                    "path": full_path
                })
            
    folders.sort(key=lambda x: x["name"])
    samples.sort(key=lambda x: x["name"])
    
    return {
        "base_dir": BASE_DIR,
        "folders": folders,
        "samples": samples
    }

@app.post("/api/upload-sample")
async def upload_sample(file: UploadFile = File(...)):
    filename = file.filename
    filename = re.sub(r'[\\/*?:"<>|]', "", filename)
    sample_dir = os.path.join(BASE_DIR, "Audio sample")
    os.makedirs(sample_dir, exist_ok=True)
    target_path = os.path.join(sample_dir, filename)
    try:
        content = await file.read()
        with open(target_path, "wb") as f:
            f.write(content)
        return {"success": True, "name": filename, "path": target_path}
    except Exception as e:
        return {"success": False, "error": str(e)}

asr_loaded = False

@app.post("/api/transcribe")
async def transcribe_audio(payload: dict):
    global asr_loaded
    audio_path = payload.get("path")
    if not audio_path or not os.path.exists(audio_path):
        raise HTTPException(status_code=400, detail="File not found")
    try:
        model_instance = get_model()
        if not asr_loaded:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, model_instance.load_asr_model)
            asr_loaded = True
        
        loop = asyncio.get_event_loop()
        transcription = await loop.run_in_executor(None, model_instance.transcribe, audio_path)
        return {"success": True, "text": transcription}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/files")
async def get_files(folder: str):
    target_dir = os.path.abspath(folder)
    if not os.path.exists(target_dir):
        return {"files": []}
        
    try:
        files = []
        for item in os.listdir(target_dir):
            full_path = os.path.join(target_dir, item)
            if os.path.isfile(full_path) and item.lower().endswith(".wav") and not item.startswith("temp_"):
                files.append({
                    "name": item,
                    "path": full_path
                })
        files.sort(key=lambda x: x["name"])
        return {"files": files}
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/audio")
async def get_audio(path: str):
    full_path = os.path.abspath(path)
    if not (full_path.lower().endswith(".wav") or full_path.lower().endswith(".mp3")):
        raise HTTPException(status_code=403, detail="Format not allowed")
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(full_path)

@app.delete("/api/delete-audio")
async def delete_audio(path: str):
    full_path = os.path.abspath(path)
    if not (full_path.lower().endswith(".wav") or full_path.lower().endswith(".mp3")):
        raise HTTPException(status_code=403, detail="Format not allowed")
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="File not found")
    try:
        os.remove(full_path)
        return {"success": True}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.delete("/api/clear-all")
async def clear_all(folder: str):
    target_dir = os.path.abspath(folder)
    if not os.path.exists(target_dir):
        raise HTTPException(status_code=404, detail="Directory not found")
    try:
        count = 0
        for item in os.listdir(target_dir):
            full_path = os.path.join(target_dir, item)
            if os.path.isfile(full_path) and item.lower().endswith(".wav") and not item.startswith("temp_"):
                os.remove(full_path)
                count += 1
        return {"success": True, "count": count}
    except Exception as e:
        return {"success": False, "error": str(e)}

from pydantic import BaseModel
class GenerateRequest(BaseModel):
    sentences: str
    folder: str
    ref_audio: str
    ref_text: str
    speed: float = 1.0
    enable_numbering: bool = False
    start_number: int = 1

@app.post("/api/generate")
async def generate_voice(req: GenerateRequest):
    async def generator() -> Generator[str, None, None]:
        try:
            yield json.dumps({"status": "model_loading"}) + "\n"
            model_instance = get_model()
            
            ref_audio_path = os.path.abspath(req.ref_audio)
            if not os.path.exists(ref_audio_path):
                yield json.dumps({"status": "error", "message": f"Không tìm thấy file mẫu: {req.ref_audio}"}) + "\n"
                return

            wav_sample = os.path.join(BASE_DIR, "temp_sample.wav")
            sound = AudioSegment.from_file(ref_audio_path)
            sound = sound.set_frame_rate(24000).set_channels(1)
            sound.export(wav_sample, format="wav")

            sentences = [s.strip() for s in req.sentences.strip().split("\n") if s.strip()]
            total = len(sentences)
            
            target_dir = os.path.abspath(req.folder)
            os.makedirs(target_dir, exist_ok=True)

            for idx, text in enumerate(sentences, start=1):
                yield json.dumps({"status": "processing", "index": idx, "total": total, "text": text}) + "\n"
                
                ten_file_sach = clean_filename(text)
                if req.enable_numbering:
                    current_num = req.start_number + (idx - 1)
                    ten_file_sach = f"{current_num}_{ten_file_sach}"
                file_output = os.path.join(target_dir, f"{ten_file_sach}.wav")
                
                if os.path.exists(file_output):
                    counter = 1
                    while True:
                        temp_output = os.path.join(target_dir, f"{ten_file_sach}({counter}).wav")
                        if not os.path.exists(temp_output):
                            file_output = temp_output
                            ten_file_sach = f"{ten_file_sach}({counter})"
                            break
                        counter += 1
                
                cau_co_moi = text
                
                loop = asyncio.get_event_loop()
                def run_gen():
                    gen_kwargs = {}
                    if req.speed and req.speed != 1.0:
                        gen_kwargs["speed"] = req.speed
                        
                    return model_instance.generate(
                        text=cau_co_moi,
                        ref_audio=wav_sample,
                        ref_text=req.ref_text,
                        **gen_kwargs
                    )
                
                try:
                    audio = await loop.run_in_executor(None, run_gen)
                    sf.write(file_output, audio[0], 24000)
                    yield json.dumps({"status": "completed", "index": idx, "total": total, "filename": f"{ten_file_sach}.wav"}) + "\n"
                except Exception as e:
                    yield json.dumps({"status": "error", "message": f"Lỗi ở câu '{text}': {str(e)}"}) + "\n"

            if os.path.exists(wav_sample):
                os.remove(wav_sample)
                
            yield json.dumps({"status": "done", "count": total}) + "\n"
            
        except Exception as e:
            yield json.dumps({"status": "error", "message": str(e)}) + "\n"

    return StreamingResponse(generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    import webbrowser
    
    print("\n🚀 Khởi chạy studio thành công!")
    print("👉 Click hoặc Command + Click vào đường dẫn sau:")
    print("http://127.0.0.1:8000\n")
    
    # Tự động mở trình duyệt
    try:
        webbrowser.open("http://127.0.0.1:8000")
    except Exception:
        pass
        
    uvicorn.run(app, host="127.0.0.1", port=8000)
