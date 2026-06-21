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
import librosa
from routes import audio_editor

app = FastAPI(title="KhanhTTS OmniVoice Studio")
app.include_router(audio_editor.router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import unicodedata

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "./khanhtts_model"
model = None

def normalize_path(path: str) -> str:
    if not path:
        return path
    return unicodedata.normalize("NFC", os.path.abspath(path))

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
            width: 100% !important;
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

        .btn-sm-secondary {
            background: transparent;
            border: 1px solid var(--glass-border);
            color: var(--text-secondary);
            border-radius: 4px;
            padding: 0.25rem 0.5rem;
            font-size: 0.75rem;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .btn-sm-secondary:hover {
            background: rgba(255, 255, 255, 0.1);
            color: var(--text-primary);
            border-color: rgba(255, 255, 255, 0.2);
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

        .tabs-nav-wrapper {
            width: 100%;
            max-width: 1200px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-top: 1rem;
            background: rgba(255, 255, 255, 0.02);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid var(--glass-border);
            padding: 0.5rem 1rem;
            border-radius: 12px;
            gap: 1rem;
        }

        .tabs-headers {
            display: flex;
            gap: 0.5rem;
            overflow-x: auto;
            flex: 1;
            padding-bottom: 2px;
        }

        .tabs-headers::-webkit-scrollbar {
            height: 4px;
        }
        .tabs-headers::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 2px;
        }

        .tab-header-btn {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--glass-border);
            color: var(--text-secondary);
            padding: 0.6rem 1.2rem;
            border-radius: 8px;
            font-size: 0.9rem;
            font-weight: 600;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            transition: all 0.2s ease;
            white-space: nowrap;
        }

        .tab-header-btn.active {
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
            border-color: transparent;
            color: white;
            box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
        }

        .tab-header-btn:hover:not(.active) {
            background: rgba(255, 255, 255, 0.08);
            color: var(--text-primary);
        }

        .tab-close-btn {
            font-size: 1.1rem;
            font-weight: bold;
            line-height: 1;
            opacity: 0.6;
            cursor: pointer;
            border: none;
            background: transparent;
            color: inherit;
            padding: 0px 4px;
            margin-left: 0.5rem;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 4px;
            transition: all 0.2s ease;
        }

        .tab-close-btn:hover {
            background: rgba(255, 255, 255, 0.2);
            opacity: 1;
        }

        .btn-add-tab {
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid rgba(16, 185, 129, 0.3);
            color: var(--success);
            padding: 0.6rem 1.2rem;
            border-radius: 8px;
            font-size: 0.9rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
            white-space: nowrap;
        }

        .btn-add-tab:hover {
            background: rgba(16, 185, 129, 0.2);
            border-color: var(--success);
        }

        .btn-run-all {
            background: linear-gradient(135deg, #ec4899 0%, #d946ef 100%);
            border: none;
            color: white;
            padding: 0.6rem 1.2rem;
            border-radius: 8px;
            font-size: 0.9rem;
            font-weight: 700;
            cursor: pointer;
            box-shadow: 0 4px 12px rgba(236, 72, 153, 0.3);
            transition: all 0.2s ease;
            white-space: nowrap;
        }

        .btn-run-all:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 16px rgba(236, 72, 153, 0.5);
        }
        
        .btn-run-all.running {
            background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
            box-shadow: 0 4px 12px rgba(239, 68, 68, 0.3);
        }

        .tab-content-panel {
            display: none !important;
        }
        
        .tab-content-panel.active {
            display: grid !important;
            width: 100% !important;
        }
    </style>
</head>
<body>
    <header>
        <h1>Hẹ Hẹ hẹ</h1>
        <p>Ứng dụng nhân bản giọng nói và tạo âm thanh hàng loạt trực quan</p>
        <div style="margin-top: 1rem; display: flex; justify-content: center; gap: 1rem;">
            <a href="/audio-editor" style="display: inline-flex; align-items: center; gap: 0.5rem; padding: 0.6rem 1.2rem; background: var(--accent); color: var(--text-primary); text-decoration: none; border-radius: 8px; font-size: 0.85rem; font-weight: 600; border: 1px solid var(--glass-border); transition: background 0.2s;" onmouseover="this.style.background='#4f46e5'" onmouseout="this.style.background='var(--accent)'">
                🎬 Trình ghép & Chỉnh sửa âm thanh (Audio Editor)
            </a>
        </div>
    </header>

    <div class="tabs-nav-wrapper">
        <div id="tabs-headers" class="tabs-headers"></div>
        <button id="btn-add-tab" class="btn-add-tab" onclick="addNewTab()">➕ Thêm Tab mới</button>
        <button id="btn-run-all" class="btn-run-all" onclick="runAllTabs()">🚀 Chạy tất cả các Tab</button>
    </div>

    <div id="tabs-container-wrapper" style="width: 100%; max-width: 1200px;">
        <!-- Tab content panels will be loaded dynamically here -->
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
        let tabs = []; // Array of tab data
        let activeTabId = null;
        let isRunningAll = false;
        let runAllQueue = [];
        let tabToStop = null;
        
        let globalPresetFolders = [];
        let globalSamples = [];
        let globalBaseDir = "";
        
        let tabLoadedFiles = {}; // tabId -> list of files

        function requestNotificationPermission() {
            if ("Notification" in window) {
                if (Notification.permission !== "granted" && Notification.permission !== "denied") {
                    Notification.requestPermission();
                }
            }
        }
        
        function sendDesktopNotification(title, body) {
            if ("Notification" in window && Notification.permission === "granted") {
                try {
                    new Notification(title, { body: body });
                } catch (e) {
                    console.error("Lỗi hiển thị thông báo:", e);
                }
            }
        }

        const confirmDialog = document.getElementById('confirm-dialog');
        const btnConfirmNo = document.getElementById('btn-confirm-no');
        const btnConfirmYes = document.getElementById('btn-confirm-yes');
        
        btnConfirmNo.onclick = () => {
            confirmDialog.close();
            tabToStop = null;
        };
        
        btnConfirmYes.onclick = () => {
            confirmDialog.close();
            if (tabToStop) {
                stopGeneration(tabToStop);
                tabToStop = null;
            }
        };

        async function loadMetadataGlobal() {
            try {
                const res = await fetch('/api/metadata');
                const data = await res.json();
                globalBaseDir = data.base_dir;
                globalPresetFolders = data.folders || [];
                globalSamples = data.samples || [];
            } catch (err) {
                console.error("Lỗi khi tải metadata toàn cục:", err);
            }
        }

        function getPrefixForIndex(startValue, index) {
            if (!startValue) return String(index + 1);
            const match = startValue.match(/^(.*?)(\\d+)$/);
            if (match) {
                const prefix = match[1];
                const numStr = match[2];
                const startNum = parseInt(numStr, 10);
                const currentNum = startNum + index;
                const paddedNum = String(currentNum).padStart(numStr.length, '0');
                return prefix + paddedNum;
            } else {
                if (index === 0) return startValue;
                return startValue + "_" + (index + 1);
            }
        }

        function syncTabCustomNumbers(tabId) {
            const tab = tabs.find(t => t.id === tabId);
            if (!tab) return;
            
            const sentencesInput = document.getElementById(`sentences-${tabId}`);
            if (!sentencesInput) return;
            
            const rawSentences = sentencesInput.value.trim();
            const newLines = rawSentences.split("\\n").map(s => s.trim()).filter(s => s);
            
            // Get previous lines from tab state
            const oldLines = tab.state.sentenceList || [];
            const oldNumbers = tab.customNumbers || [];
            const newNumbers = [];
            const usedOldIndices = new Set();
            
            for (let i = 0; i < newLines.length; i++) {
                const newLine = newLines[i];
                // 1. Try to match same index first
                if (oldLines[i] === newLine && !usedOldIndices.has(i)) {
                    newNumbers.push(oldNumbers[i] || "");
                    usedOldIndices.add(i);
                    continue;
                }
                
                // 2. Try to find the same content elsewhere
                let foundIndex = -1;
                for (let j = 0; j < oldLines.length; j++) {
                    if (oldLines[j] === newLine && !usedOldIndices.has(j)) {
                        foundIndex = j;
                        break;
                    }
                }
                
                if (foundIndex !== -1) {
                    newNumbers.push(oldNumbers[foundIndex] || "");
                    usedOldIndices.add(foundIndex);
                } else {
                    newNumbers.push(null);
                }
            }
            
            // Fill default prefixes
            for (let i = 0; i < newNumbers.length; i++) {
                if (newNumbers[i] === null || newNumbers[i] === "") {
                    let defaultVal = "";
                    if (i > 0 && newNumbers[i - 1]) {
                        const prevVal = newNumbers[i - 1];
                        const match = prevVal.match(/^(.*?)(\\d+)$/);
                        if (match) {
                            const prefix = match[1];
                            const numStr = match[2];
                            const nextNum = parseInt(numStr, 10) + 1;
                            const paddedNum = String(nextNum).padStart(numStr.length, '0');
                            defaultVal = prefix + paddedNum;
                        } else {
                            defaultVal = String(i + 1);
                        }
                    } else {
                        defaultVal = String(i + 1);
                    }
                    newNumbers[i] = defaultVal;
                }
            }
            
            tab.customNumbers = newNumbers;
            tab.state.sentenceList = newLines;
            
            saveAllTabs();
            
            // Render custom numbering list UI
            const customListEl = document.getElementById(`custom-numbering-list-${tabId}`);
            if (customListEl) {
                customListEl.innerHTML = "";
                if (newLines.length === 0) {
                    customListEl.innerHTML = `<div style="font-size: 0.85rem; color: var(--text-secondary); text-align: center; padding: 1rem;">Nhập danh sách câu để tùy chỉnh số thứ tự.</div>`;
                    return;
                }
                
                newLines.forEach((line, index) => {
                    const item = document.createElement("div");
                    item.className = "custom-number-item";
                    item.style.cssText = "display: flex; align-items: center; justify-content: space-between; gap: 1rem; background: rgba(255, 255, 255, 0.02); padding: 0.4rem 0.6rem; border-radius: 6px; border: 1px solid rgba(255, 255, 255, 0.03);";
                    
                    const label = document.createElement("span");
                    label.className = "custom-number-sentence";
                    label.style.cssText = "font-size: 0.85rem; color: var(--text-secondary); text-overflow: ellipsis; white-space: nowrap; overflow: hidden; flex: 1;";
                    label.textContent = `${index + 1}. ${line}`;
                    label.title = line;
                    
                    const input = document.createElement("input");
                    input.type = "text";
                    input.className = "custom-number-input-field";
                    input.value = newNumbers[index] || "";
                    input.style.cssText = "width: 100px; padding: 0.35rem 0.6rem; font-size: 0.85rem; border-radius: 6px; background: rgba(15, 23, 42, 0.6); border: 1px solid var(--glass-border); color: var(--text-primary); text-align: center; outline: none;";
                    
                    input.addEventListener("input", () => {
                        tab.customNumbers[index] = input.value;
                        saveAllTabs();
                    });
                    
                    item.appendChild(label);
                    item.appendChild(input);
                    customListEl.appendChild(item);
                });
            }
        }

        async function initTabs() {
            await loadMetadataGlobal();
            
            const savedTabsStr = localStorage.getItem('tts_multi_tabs');
            const savedActiveTabId = localStorage.getItem('tts_active_tab_id');
            
            if (savedTabsStr) {
                try {
                    const parsed = JSON.parse(savedTabsStr);
                    if (Array.isArray(parsed) && parsed.length > 0) {
                        tabs = parsed.map(t => {
                            // Convert old structures
                            let method = t.numberingMethod;
                            if (method === undefined) {
                                method = t.enableNumbering ? 'auto' : 'none';
                            }
                            let sNum = t.startNumber;
                            if (sNum === undefined) {
                                sNum = '001';
                            } else {
                                sNum = String(sNum);
                            }
                            return {
                                ...t,
                                numberingMethod: method,
                                startNumber: sNum,
                                customNumbers: Array.isArray(t.customNumbers) ? t.customNumbers : [],
                                state: {
                                    sentenceList: [],
                                    currentIndex: 0,
                                    totalSentences: 0,
                                    isPaused: false,
                                    isStopped: false,
                                    isGenerating: false
                                }
                            };
                        });
                    }
                } catch (e) {
                    console.error("Lỗi parse local storage:", e);
                }
            }
            
            if (tabs.length === 0) {
                const defaultTab = {
                    id: 'tab_default',
                    name: 'Cửa sổ 1',
                    sentences: '',
                    folderPath: '',
                    selectedSample: '',
                    refText: '',
                    speed: 1.0,
                    speedMethod: 'model',
                    numberingMethod: 'none',
                    startNumber: '001',
                    customNumbers: [],
                    state: {
                        sentenceList: [],
                        currentIndex: 0,
                        totalSentences: 0,
                        isPaused: false,
                        isStopped: false,
                        isGenerating: false
                    }
                };
                tabs.push(defaultTab);
                saveAllTabs();
            }
            
            const wrapper = document.getElementById('tabs-container-wrapper');
            wrapper.innerHTML = '';
            
            tabs.forEach(tab => {
                renderTabPanel(tab);
                populateTabDropdowns(tab.id);
                // Render custom list if the method is custom
                if (tab.numberingMethod === 'custom') {
                    syncTabCustomNumbers(tab.id);
                }
            });
            
            renderTabHeaders();
            
            if (savedActiveTabId && tabs.some(t => t.id === savedActiveTabId)) {
                selectTab(savedActiveTabId);
            } else {
                selectTab(tabs[0].id);
            }
        }

        function saveAllTabs() {
            const tabsToSave = tabs.map(t => ({
                id: t.id,
                name: t.name,
                sentences: t.sentences,
                folderPath: t.folderPath,
                selectedSample: t.selectedSample,
                refText: t.refText,
                speed: t.speed,
                speedMethod: t.speedMethod,
                numberingMethod: t.numberingMethod,
                startNumber: t.startNumber,
                customNumbers: t.customNumbers
            }));
            localStorage.setItem('tts_multi_tabs', JSON.stringify(tabsToSave));
            localStorage.setItem('tts_active_tab_id', activeTabId);
        }

        function selectTab(tabId) {
            activeTabId = tabId;
            saveAllTabs();
            
            document.querySelectorAll('.tab-header-btn').forEach(btn => {
                if (btn.dataset.id === tabId) {
                    btn.classList.add('active');
                } else {
                    btn.classList.remove('active');
                }
            });
            
            document.querySelectorAll('.tab-content-panel').forEach(panel => {
                if (panel.id === `panel-${tabId}`) {
                    panel.classList.add('active');
                } else {
                    panel.classList.remove('active');
                }
            });
            
            loadFilesForTab(tabId);
        }

        function addNewTab() {
            const tabNum = tabs.length + 1;
            const newTab = {
                id: 'tab_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9),
                name: `Cửa sổ ${tabNum}`,
                sentences: '',
                folderPath: '',
                selectedSample: '',
                refText: '',
                speed: 1.0,
                speedMethod: 'model',
                numberingMethod: 'none',
                startNumber: '001',
                customNumbers: [],
                state: {
                    sentenceList: [],
                    currentIndex: 0,
                    totalSentences: 0,
                    isPaused: false,
                    isStopped: false,
                    isGenerating: false
                }
            };
            
            tabs.push(newTab);
            saveAllTabs();
            
            renderTabHeaders();
            renderTabPanel(newTab);
            populateTabDropdowns(newTab.id);
            
            selectTab(newTab.id);
        }

        function closeTab(tabId) {
            const tab = tabs.find(t => t.id === tabId);
            if (!tab) return;
            
            // Confirm deletion
            if (!confirm(`⚠️ Bạn có chắc chắn muốn xóa "${tab.name}" không? Tất cả các thiết lập của cửa sổ này sẽ bị mất.`)) {
                return;
            }
            
            if (tab.state && tab.state.isGenerating) {
                tab.state.isStopped = true;
            }
            
            const panel = document.getElementById(`panel-${tabId}`);
            if (panel) panel.remove();
            
            tabs = tabs.filter(t => t.id !== tabId);
            
            // Reset tab names sequentially
            tabs.forEach((t, index) => {
                t.name = `Cửa sổ ${index + 1}`;
            });
            
            saveAllTabs();
            
            if (activeTabId === tabId && tabs.length > 0) {
                selectTab(tabs[tabs.length - 1].id);
            }
            
            renderTabHeaders();
        }

        function renderTabHeaders() {
            const container = document.getElementById('tabs-headers');
            container.innerHTML = '';
            
            tabs.forEach(tab => {
                const btn = document.createElement('button');
                btn.className = `tab-header-btn ${tab.id === activeTabId ? 'active' : ''}`;
                btn.dataset.id = tab.id;
                btn.onclick = () => selectTab(tab.id);
                
                const label = document.createElement('span');
                label.textContent = tab.name;
                btn.appendChild(label);
                
                if (tabs.length > 1) {
                    const closeBtn = document.createElement('span');
                    closeBtn.className = 'tab-close-btn';
                    closeBtn.textContent = '×';
                    closeBtn.onclick = (e) => {
                        e.stopPropagation();
                        closeTab(tab.id);
                    };
                    btn.appendChild(closeBtn);
                }
                
                container.appendChild(btn);
            });
        }

        function renderTabPanel(tab) {
            const wrapper = document.getElementById('tabs-container-wrapper');
            const panelHtml = `
            <div class="container tab-content-panel" id="panel-${tab.id}">
                <!-- Left Panel: Configurations -->
                <div class="panel">
                    <div class="form-group">
                        <label for="sentences-${tab.id}">Danh sách câu nói cần gen (Mỗi câu một dòng)</label>
                        <textarea id="sentences-${tab.id}" class="sentences-textarea" placeholder="Ví dụ:&#10;Chưa đúng rồi, con chọn lại nhé.&#10;Tuyệt vời! Con giỏi quá.">${tab.sentences || ''}</textarea>
                    </div>

                    <div class="form-group">
                        <label for="folder-select-${tab.id}">Thư mục lưu kết quả</label>
                        <select id="folder-select-${tab.id}" class="folder-select" style="margin-bottom: 0.5rem;">
                            <option value="ROOT">Chọn thư mục có sẵn...</option>
                        </select>
                        <div style="display: flex; gap: 0.5rem; width: 100%;">
                            <input type="text" id="folder-path-${tab.id}" class="folder-path-input" placeholder="Đường dẫn thư mục lưu..." value="${tab.folderPath || ''}" style="flex: 1;">
                            <button class="btn-secondary btn-browse-folder" id="btn-browse-folder-${tab.id}" style="white-space: nowrap; min-width: 110px;">🔍 Finder...</button>
                        </div>
                    </div>

                    <div class="row">
                        <div class="form-group">
                            <label for="sample-select-${tab.id}">File mẫu (Reference Audio)</label>
                            <select id="sample-select-${tab.id}" class="sample-select" style="margin-bottom: 0.5rem;">
                                <!-- Loaded dynamically -->
                            </select>
                            <!-- Audio player để nghe thử file mẫu -->
                            <audio id="sample-audio-player-${tab.id}" class="sample-audio-player" controls style="height: 32px; width: 100%; display: none;"></audio>
                        </div>
                        <div class="form-group">
                            <label>Tải lên file mẫu mới</label>
                            <div class="upload-area">
                                <button class="btn-secondary" onclick="document.getElementById('file-uploader-${tab.id}').click()" style="width: 100%; white-space: nowrap; margin-bottom: 0.5rem;">📁 Chọn tệp mẫu</button>
                                <input type="file" id="file-uploader-${tab.id}" class="file-uploader" accept=".mp3,.wav" style="display:none;">
                            </div>
                        </div>
                    </div>

                    <div class="form-group">
                        <label for="ref-text-${tab.id}">Nội dung chính xác của file mẫu</label>
                        <input type="text" id="ref-text-${tab.id}" class="ref-text-input" value="${tab.refText || 'Tuyệt vời, đây là số 3 nhé!'}">
                        <span id="upload-status-${tab.id}" class="upload-status" style="font-size: 0.85rem; color: var(--success); min-height: 1.2rem; display: block; margin-top: -0.25rem;"></span>
                    </div>

                    <div class="row" style="grid-template-columns: 1fr 1.2fr; gap: 1rem; align-items: center;">
                        <div class="form-group">
                            <label>Tốc độ đọc: <span id="speed-val-${tab.id}" class="speed-val">${parseFloat(tab.speed).toFixed(2)}</span>x</label>
                            <div style="display: flex; align-items: center; height: 100%; padding-top: 0.25rem;">
                                <input type="range" id="speed-slider-${tab.id}" class="speed-slider" min="0.5" max="2.0" step="0.05" value="${tab.speed || 1.0}" style="width: 100%; accent-color: var(--accent);">
                            </div>
                        </div>
                        <div class="form-group">
                            <label for="speed-method-${tab.id}">Phương pháp tinh chỉnh</label>
                            <select id="speed-method-${tab.id}" class="speed-method-select">
                                <option value="model" ${tab.speedMethod === 'model' ? 'selected' : ''}>AI Tự nhiên (Model-based)</option>
                                <option value="dsp" ${tab.speedMethod === 'dsp' ? 'selected' : ''}>Chính xác (DSP-based)</option>
                            </select>
                        </div>
                    </div>

                    <div class="row" style="grid-template-columns: 1fr 1fr; align-items: center; gap: 1rem; margin-top: -0.5rem;">
                        <div class="form-group">
                            <label for="numbering-method-${tab.id}">Đánh số thứ tự file</label>
                            <select id="numbering-method-${tab.id}" class="numbering-method-select">
                                <option value="none" ${tab.numberingMethod === 'none' ? 'selected' : ''}>Không đánh số</option>
                                <option value="auto" ${tab.numberingMethod === 'auto' ? 'selected' : ''}>Tự động tăng dần</option>
                                <option value="custom" ${tab.numberingMethod === 'custom' ? 'selected' : ''}>Tự đánh số (Tùy chỉnh)</option>
                            </select>
                        </div>
                        <div class="form-group start-number-group" id="start-number-group-${tab.id}" style="display: ${tab.numberingMethod === 'auto' ? 'flex' : 'none'}; height: 100%;">
                            <label for="start-number-${tab.id}">Số bắt đầu</label>
                            <input type="text" id="start-number-${tab.id}" class="start-number-input" value="${tab.startNumber !== undefined ? tab.startNumber : '001'}" style="background: rgba(15, 23, 42, 0.6); border: 1px solid var(--glass-border); border-radius: 8px; padding: 0.75rem 1rem; color: var(--text-primary); outline: none; width: 100%;">
                        </div>
                    </div>

                    <div id="custom-numbering-container-${tab.id}" class="form-group custom-numbering-container" style="display: ${tab.numberingMethod === 'custom' ? 'flex' : 'none'}; flex-direction: column; gap: 0.5rem; margin-top: -0.5rem;">
                        <label>Đánh số tùy chỉnh từng câu</label>
                        <div id="custom-numbering-list-${tab.id}" class="custom-numbering-list" style="max-height: 200px; overflow-y: auto; border: 1px solid var(--glass-border); border-radius: 8px; padding: 0.5rem; background: rgba(15, 23, 42, 0.4); display: flex; flex-direction: column; gap: 0.5rem;">
                            <!-- Will be rendered dynamically -->
                        </div>
                    </div>

                    <button id="btn-generate-${tab.id}" class="btn btn-generate" onclick="startGeneration('${tab.id}')">
                        <span>🚀 Bắt đầu tạo hàng loạt</span>
                    </button>

                    <!-- Control buttons during generation -->
                    <div id="control-group-${tab.id}" class="control-group" style="display: none;">
                        <button id="btn-pause-${tab.id}" class="btn btn-pause" onclick="togglePause('${tab.id}')" style="background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); box-shadow: 0 4px 12px rgba(245, 158, 11, 0.3);">
                            ⏸️ Tạm dừng
                        </button>
                        <button id="btn-stop-${tab.id}" class="btn btn-stop" onclick="confirmStop('${tab.id}')" style="background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); box-shadow: 0 4px 12px rgba(239, 68, 68, 0.3);">
                            ⏹️ Dừng hẳn
                        </button>
                    </div>

                    <!-- Progress Visualizer -->
                    <div id="progress-container-${tab.id}" class="progress-container" style="display: none;">
                        <div class="progress-header">
                            <span id="progress-label-${tab.id}" class="progress-label">Đang chuẩn bị...</span>
                            <span id="progress-percentage-${tab.id}" class="progress-percentage">0%</span>
                        </div>
                        <div class="progress-bar-bg">
                            <div id="progress-bar-fill-${tab.id}" class="progress-bar-fill"></div>
                        </div>
                        <div id="status-text-${tab.id}" class="status-text">Đang khởi tạo mô hình...</div>
                    </div>
                </div>

                <!-- Right Panel: Audio Management -->
                <div class="panel" style="height: 100%; min-height: 0;">
                    <div class="form-group" style="flex: 1; display: flex; flex-direction: column; min-height: 0; gap: 0.5rem;">
                        <div class="manager-header">
                            <label>Các file đã tạo trong thư mục</label>
                            <div style="display: flex; gap: 0.5rem;">
                                <button class="btn-secondary" style="padding: 0.4rem 0.8rem; font-size: 0.8rem;" onclick="loadFilesForTab('${tab.id}')">Reset</button>
                                <button id="btn-clear-all-${tab.id}" class="btn-sm-danger btn-clear-all" style="padding: 0.4rem 0.8rem; font-size: 0.8rem; border-color: rgba(239, 68, 68, 0.4);" onclick="clearAllFiles('${tab.id}')">Delete</button>
                            </div>
                        </div>
                        <input type="text" id="search-files-${tab.id}" class="search-files" placeholder="🔍 Tìm kiếm file nhanh..." style="margin-top: 0.5rem; margin-bottom: 0.5rem; padding: 0.5rem 0.75rem; font-size: 0.85rem;" oninput="filterFilesList('${tab.id}')">
                        <div id="file-list-container-${tab.id}" class="file-list" style="flex: 1; max-height: none; overflow-y: auto;">
                            <div class="empty-state">Nhập thư mục để tải danh sách file âm thanh.</div>
                        </div>
                    </div>
                </div>
            </div>
            `;
            
            const tempDiv = document.createElement('div');
            tempDiv.innerHTML = panelHtml.trim();
            wrapper.appendChild(tempDiv.firstChild);
            
            setupTabEventListeners(tab.id);
        }

        function populateTabDropdowns(tabId) {
            const tab = tabs.find(t => t.id === tabId);
            if (!tab) return;
            
            const folderSelect = document.getElementById(`folder-select-${tabId}`);
            const sampleSelect = document.getElementById(`sample-select-${tabId}`);
            
            folderSelect.innerHTML = '';
            const defaultOpt = document.createElement('option');
            defaultOpt.value = "ROOT";
            defaultOpt.textContent = "Chọn thư mục có sẵn...";
            folderSelect.appendChild(defaultOpt);
            
            globalPresetFolders.forEach(folder => {
                const opt = document.createElement('option');
                opt.value = folder.path;
                opt.textContent = folder.name;
                folderSelect.appendChild(opt);
            });
            
            sampleSelect.innerHTML = '';
            globalSamples.forEach(sample => {
                const opt = document.createElement('option');
                opt.value = sample.path;
                opt.textContent = sample.name;
                sampleSelect.appendChild(opt);
            });
            
            const folderPathInput = document.getElementById(`folder-path-${tabId}`);
            if (tab.folderPath) {
                folderPathInput.value = tab.folderPath;
            } else {
                const ketQuaFolder = globalPresetFolders.find(f => f.name === "Kết quả");
                if (ketQuaFolder) {
                    folderPathInput.value = ketQuaFolder.path;
                } else if (globalPresetFolders.length > 0) {
                    folderPathInput.value = globalPresetFolders[0].path;
                } else {
                    folderPathInput.value = globalBaseDir;
                }
                tab.folderPath = folderPathInput.value;
            }
            
            syncTabFolderSelect(tabId);
            
            if (tab.selectedSample) {
                const normSelected = tab.selectedSample.normalize('NFC');
                let foundSample = globalSamples.find(s => s.path.normalize('NFC') === normSelected);
                if (foundSample) {
                    sampleSelect.value = foundSample.path;
                    tab.selectedSample = foundSample.path;
                } else if (globalSamples.length > 0) {
                    sampleSelect.value = globalSamples[0].path;
                    tab.selectedSample = sampleSelect.value;
                }
            } else if (globalSamples.length > 0) {
                const sampleMp3 = globalSamples.find(s => s.name === 'sample.mp3');
                if (sampleMp3) {
                    sampleSelect.value = sampleMp3.path;
                } else {
                    sampleSelect.value = globalSamples[0].path;
                }
                tab.selectedSample = sampleSelect.value;
            }
            
            updateTabSampleAudio(tabId, !!tab.refText);
        }

        function syncTabFolderSelect(tabId) {
            const folderPathInput = document.getElementById(`folder-path-${tabId}`);
            const folderSelect = document.getElementById(`folder-select-${tabId}`);
            const currentPath = folderPathInput.value.trim();
            let matched = false;
            for (let i = 0; i < folderSelect.options.length; i++) {
                if (folderSelect.options[i].value === currentPath) {
                    folderSelect.value = currentPath;
                    matched = true;
                    break;
                }
            }
            if (!matched) {
                folderSelect.value = "ROOT";
            }
        }

        async function updateTabSampleAudio(tabId, restoreText = false) {
            const sampleSelect = document.getElementById(`sample-select-${tabId}`);
            const sampleAudioPlayer = document.getElementById(`sample-audio-player-${tabId}`);
            const refTextInput = document.getElementById(`ref-text-${tabId}`);
            
            const selectedPath = sampleSelect.value;
            if (selectedPath) {
                sampleAudioPlayer.src = `/api/audio?path=${encodeURIComponent(selectedPath)}&t=${new Date().getTime()}`;
                sampleAudioPlayer.style.display = 'block';
                sampleAudioPlayer.load();
                if (restoreText) {
                    const tab = tabs.find(t => t.id === tabId);
                    if (tab && tab.refText) {
                        refTextInput.value = tab.refText;
                        return;
                    }
                }
                await transcribeTabSample(tabId, selectedPath);
            } else {
                sampleAudioPlayer.style.display = 'none';
            }
        }

        async function transcribeTabSample(tabId, path) {
            const refTextInput = document.getElementById(`ref-text-${tabId}`);
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
                    const tab = tabs.find(t => t.id === tabId);
                    if (tab) {
                        tab.refText = data.text;
                        saveAllTabs();
                    }
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

        function setupTabEventListeners(tabId) {
            const tab = tabs.find(t => t.id === tabId);
            if (!tab) return;
            
            const sentencesEl = document.getElementById(`sentences-${tabId}`);
            const folderPathEl = document.getElementById(`folder-path-${tabId}`);
            const folderSelectEl = document.getElementById(`folder-select-${tabId}`);
            const sampleSelectEl = document.getElementById(`sample-select-${tabId}`);
            const refTextEl = document.getElementById(`ref-text-${tabId}`);
            const speedSliderEl = document.getElementById(`speed-slider-${tabId}`);
            const speedValEl = document.getElementById(`speed-val-${tabId}`);
            const speedMethodEl = document.getElementById(`speed-method-${tabId}`);
            const numberingMethodEl = document.getElementById(`numbering-method-${tabId}`);
            const startNumberEl = document.getElementById(`start-number-${tabId}`);
            const startNumberGroupEl = document.getElementById(`start-number-group-${tabId}`);
            const customNumberingContainerEl = document.getElementById(`custom-numbering-container-${tabId}`);
            const btnBrowseFolderEl = document.getElementById(`btn-browse-folder-${tabId}`);
            const fileUploaderEl = document.getElementById(`file-uploader-${tabId}`);
            
            sentencesEl.addEventListener('input', () => {
                tab.sentences = sentencesEl.value;
                saveAllTabs();
                syncTabCustomNumbers(tabId);
            });
            
            folderPathEl.addEventListener('change', () => {
                tab.folderPath = folderPathEl.value.trim();
                saveAllTabs();
                syncTabFolderSelect(tabId);
                loadFilesForTab(tabId);
            });
            
            folderSelectEl.addEventListener('change', () => {
                if (folderSelectEl.value !== "ROOT") {
                    folderPathEl.value = folderSelectEl.value;
                    tab.folderPath = folderSelectEl.value;
                    saveAllTabs();
                    loadFilesForTab(tabId);
                }
            });
            
            sampleSelectEl.addEventListener('change', () => {
                tab.selectedSample = sampleSelectEl.value;
                saveAllTabs();
                updateTabSampleAudio(tabId);
            });
            
            refTextEl.addEventListener('input', () => {
                tab.refText = refTextEl.value;
                saveAllTabs();
            });
            
            speedSliderEl.addEventListener('input', () => {
                const val = parseFloat(speedSliderEl.value);
                speedValEl.textContent = val.toFixed(2);
                tab.speed = val;
                saveAllTabs();
            });
            
            speedMethodEl.addEventListener('change', () => {
                tab.speedMethod = speedMethodEl.value;
                saveAllTabs();
            });
            
            numberingMethodEl.addEventListener('change', () => {
                tab.numberingMethod = numberingMethodEl.value;
                startNumberGroupEl.style.display = tab.numberingMethod === 'auto' ? 'flex' : 'none';
                customNumberingContainerEl.style.display = tab.numberingMethod === 'custom' ? 'flex' : 'none';
                saveAllTabs();
                if (tab.numberingMethod === 'custom') {
                    syncTabCustomNumbers(tabId);
                }
            });
            
            startNumberEl.addEventListener('input', () => {
                tab.startNumber = startNumberEl.value.trim();
                saveAllTabs();
            });
            
            btnBrowseFolderEl.addEventListener('click', async () => {
                btnBrowseFolderEl.disabled = true;
                btnBrowseFolderEl.textContent = "⌛ Đang chọn...";
                try {
                    const res = await fetch('/api/browse-folder', { method: 'POST' });
                    const data = await res.json();
                    if (data.path) {
                        folderPathEl.value = data.path;
                        tab.folderPath = data.path;
                        saveAllTabs();
                        syncTabFolderSelect(tabId);
                        loadFilesForTab(tabId);
                    } else if (data.error) {
                        alert("Lỗi khi chọn thư mục: " + data.error);
                    }
                } catch (err) {
                    console.error("Lỗi kết nối:", err);
                } finally {
                    btnBrowseFolderEl.disabled = false;
                    btnBrowseFolderEl.textContent = "🔍 Finder...";
                }
            });
            
            fileUploaderEl.addEventListener('change', async () => {
                if (!fileUploaderEl.files || fileUploaderEl.files.length === 0) return;
                const file = fileUploaderEl.files[0];
                const formData = new FormData();
                formData.append("file", file);
                
                const uploadStatusEl = document.getElementById(`upload-status-${tabId}`);
                uploadStatusEl.style.color = "var(--text-secondary)";
                uploadStatusEl.textContent = "Đang tải lên...";
                
                try {
                    const res = await fetch("/api/upload-sample", {
                        method: "POST",
                        body: formData
                    });
                    const data = await res.json();
                    if (data.success) {
                        uploadStatusEl.style.color = "var(--success)";
                        uploadStatusEl.textContent = `Đã tải lên: ${data.name}`;
                        
                        await loadMetadataGlobal();
                        
                        tabs.forEach(t => {
                            const sampleSelectOther = document.getElementById(`sample-select-${t.id}`);
                            if (sampleSelectOther) {
                                const valBefore = sampleSelectOther.value;
                                sampleSelectOther.innerHTML = '';
                                globalSamples.forEach(sample => {
                                    const opt = document.createElement('option');
                                    opt.value = sample.path;
                                    opt.textContent = sample.name;
                                    sampleSelectOther.appendChild(opt);
                                });
                                if (t.id === tabId) {
                                    const normPath = data.path.normalize('NFC');
                                    const matchedSample = globalSamples.find(s => s.path.normalize('NFC') === normPath);
                                    if (matchedSample) {
                                        sampleSelectOther.value = matchedSample.path;
                                        tab.selectedSample = matchedSample.path;
                                    } else {
                                        sampleSelectOther.value = data.path;
                                        tab.selectedSample = data.path;
                                    }
                                } else {
                                    const normBefore = valBefore ? valBefore.normalize('NFC') : '';
                                    const matchedBefore = globalSamples.find(s => s.path.normalize('NFC') === normBefore);
                                    if (matchedBefore) {
                                        sampleSelectOther.value = matchedBefore.path;
                                    } else {
                                        sampleSelectOther.value = valBefore;
                                    }
                                }
                            }
                        });
                        
                        saveAllTabs();
                        updateTabSampleAudio(tabId);
                    } else {
                        uploadStatusEl.style.color = "var(--danger)";
                        uploadStatusEl.textContent = "Lỗi: " + data.error;
                    }
                } catch (err) {
                    uploadStatusEl.style.color = "var(--danger)";
                    uploadStatusEl.textContent = "Lỗi kết nối";
                }
            });
        }

        async function loadFilesForTab(tabId) {
            const folderPathInput = document.getElementById(`folder-path-${tabId}`);
            const fileListContainer = document.getElementById(`file-list-container-${tabId}`);
            const searchFilesInput = document.getElementById(`search-files-${tabId}`);
            if (!folderPathInput || !fileListContainer || !searchFilesInput) return;
            
            const folder = folderPathInput.value.trim();
            if (!folder) return;
            try {
                const res = await fetch(`/api/files?folder=${encodeURIComponent(folder)}`);
                const data = await res.json();
                
                if (data.error) {
                    fileListContainer.innerHTML = `<div class="empty-state" style="color:var(--danger)">Không thể đọc thư mục: ${data.error}</div>`;
                    return;
                }

                tabLoadedFiles[tabId] = data.files || [];
                renderFilesListForTab(tabId, searchFilesInput.value.trim());
            } catch (err) {
                console.error("Lỗi khi tải file:", err);
            }
        }

        function renderFilesListForTab(tabId, filterText = "") {
            const fileListContainer = document.getElementById(`file-list-container-${tabId}`);
            if (!fileListContainer) return;
            const files = tabLoadedFiles[tabId] || [];
            const query = filterText.toLowerCase();
            const filtered = files.filter(file => file.name.toLowerCase().includes(query));

            if (filtered.length === 0) {
                fileListContainer.innerHTML = files.length === 0 
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
                info.style.display = 'flex';
                info.style.alignItems = 'center';
                info.style.justifyContent = 'space-between';
                info.style.gap = '0.5rem';
                info.style.width = '100%';
                
                const name = document.createElement('span');
                name.className = 'file-name';
                name.textContent = file.name;
                name.style.flex = '1';
                name.style.wordBreak = 'break-all';
                
                const nameInput = document.createElement('input');
                nameInput.type = 'text';
                nameInput.value = file.name;
                nameInput.className = 'rename-input';
                nameInput.style.display = 'none';
                nameInput.style.flex = '1';
                nameInput.style.fontSize = '0.85rem';
                nameInput.style.padding = '0.2rem 0.4rem';
                nameInput.style.background = 'rgba(15, 23, 42, 0.8)';
                nameInput.style.border = '1px solid var(--glass-border)';
                nameInput.style.borderRadius = '4px';
                nameInput.style.color = 'var(--text-primary)';
                nameInput.style.outline = 'none';
                
                const actions = document.createElement('div');
                actions.className = 'file-actions';
                actions.style.display = 'flex';
                actions.style.gap = '0.25rem';
                
                const renameBtn = document.createElement('button');
                renameBtn.className = 'btn-sm-secondary';
                renameBtn.textContent = 'Sửa';
                
                const saveRename = async () => {
                    const newName = nameInput.value.trim();
                    if (!newName) {
                        alert("Tên file không được để trống!");
                        return;
                    }
                    if (newName === file.name) {
                        name.style.display = 'block';
                        nameInput.style.display = 'none';
                        renameBtn.textContent = 'Sửa';
                        return;
                    }
                    try {
                        const res = await fetch('/api/rename-audio', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ path: file.path, new_name: newName })
                        });
                        const result = await res.json();
                        if (result.success) {
                            loadFilesForTab(tabId);
                        } else {
                            alert("Không thể đổi tên: " + result.error);
                        }
                    } catch (e) {
                        alert("Lỗi đổi tên: " + e.message);
                    }
                };
                
                renameBtn.onclick = () => {
                    if (nameInput.style.display === 'none') {
                        name.style.display = 'none';
                        nameInput.style.display = 'block';
                        nameInput.focus();
                        const dotIndex = file.name.lastIndexOf('.');
                        if (dotIndex !== -1) {
                            nameInput.setSelectionRange(0, dotIndex);
                        } else {
                            nameInput.select();
                        }
                        renameBtn.textContent = 'Lưu';
                    } else {
                        saveRename();
                    }
                };
                
                nameInput.onkeydown = (e) => {
                    if (e.key === 'Enter') {
                        saveRename();
                    } else if (e.key === 'Escape') {
                        nameInput.value = file.name;
                        name.style.display = 'block';
                        nameInput.style.display = 'none';
                        renameBtn.textContent = 'Sửa';
                    }
                };
                
                const deleteBtn = document.createElement('button');
                deleteBtn.className = 'btn-sm-danger';
                deleteBtn.textContent = 'Xóa';
                deleteBtn.onclick = () => deleteFileForTab(tabId, file.path);
                
                actions.appendChild(renameBtn);
                actions.appendChild(deleteBtn);
                info.appendChild(name);
                info.appendChild(nameInput);
                info.appendChild(actions);
                
                const audio = document.createElement('audio');
                audio.controls = true;
                audio.preload = 'metadata';
                audio.src = `/api/audio?path=${encodeURIComponent(file.path)}&t=${new Date().getTime()}`;
                
                item.appendChild(info);
                item.appendChild(audio);
                fileListContainer.appendChild(item);
            });
        }

        function filterFilesList(tabId) {
            const searchInput = document.getElementById(`search-files-${tabId}`);
            if (searchInput) {
                renderFilesListForTab(tabId, searchInput.value.trim());
            }
        }

        async function deleteFileForTab(tabId, path) {
            if (!confirm(`Bạn có chắc muốn xóa file này?`)) return;
            try {
                const res = await fetch(`/api/delete-audio?path=${encodeURIComponent(path)}`, { method: 'DELETE' });
                const result = await res.json();
                if (result.success) {
                    loadFilesForTab(tabId);
                } else {
                    alert("Không thể xóa file: " + result.error);
                }
            } catch (err) {
                alert("Lỗi khi xóa file: " + err);
            }
        }

        async function clearAllFiles(tabId) {
            const folderPathEl = document.getElementById(`folder-path-${tabId}`);
            const folder = folderPathEl.value.trim();
            if (!folder) return;
            if (confirm("⚠️ Bạn có chắc chắn muốn XÓA TẤT CẢ các file âm thanh .wav trong thư mục này không? Thao tác này không thể hoàn tác.")) {
                try {
                    const res = await fetch(`/api/clear-all?folder=${encodeURIComponent(folder)}`, { method: 'DELETE' });
                    const result = await res.json();
                    if (result.success) {
                        loadFilesForTab(tabId);
                    } else {
                        alert("Lỗi khi xóa tất cả: " + result.error);
                    }
                } catch (err) {
                    alert("Lỗi kết nối: " + err);
                }
            }
        }

        function resetTabUI(tabId) {
            const btnGenerate = document.getElementById(`btn-generate-${tabId}`);
            const controlGroup = document.getElementById(`control-group-${tabId}`);
            const btnPause = document.getElementById(`btn-pause-${tabId}`);
            const btnStop = document.getElementById(`btn-stop-${tabId}`);
            
            if (btnGenerate) btnGenerate.style.display = 'flex';
            if (controlGroup) controlGroup.style.display = 'none';
            if (btnPause) {
                btnPause.disabled = false;
                btnPause.innerHTML = '⏸️ Tạm dừng';
            }
            if (btnStop) {
                btnStop.disabled = false;
                btnStop.innerHTML = '⏹️ Dừng hẳn';
            }
        }

        async function processNextSentenceForTab(tabId) {
            const tab = tabs.find(t => t.id === tabId);
            if (!tab) return;
            
            const { state } = tab;
            
            const progressContainer = document.getElementById(`progress-container-${tabId}`);
            const progressLabel = document.getElementById(`progress-label-${tabId}`);
            const progressPercentage = document.getElementById(`progress-percentage-${tabId}`);
            const progressBarFill = document.getElementById(`progress-bar-fill-${tabId}`);
            const statusText = document.getElementById(`status-text-${tabId}`);
            const btnPause = document.getElementById(`btn-pause-${tabId}`);
            
            if (state.isStopped) {
                if (statusText) statusText.textContent = '⛔ Quá trình đã bị dừng bởi người dùng.';
                state.isGenerating = false;
                resetTabUI(tabId);
                loadFilesForTab(tabId);
                
                if (isRunningAll) {
                    stopRunAllQueue();
                }
                return;
            }

            if (state.isPaused) {
                if (statusText) statusText.textContent = '⏸️ Đang tạm dừng. Nhấp "Tiếp tục" để chạy tiếp.';
                if (btnPause) {
                    btnPause.innerHTML = '▶️ Tiếp tục';
                    btnPause.disabled = false;
                }
                return;
            }

            if (state.currentIndex >= state.totalSentences) {
                if (progressBarFill) progressBarFill.style.width = '100%';
                if (progressPercentage) progressPercentage.textContent = '100%';
                if (progressLabel) progressLabel.textContent = 'Hoàn thành!';
                if (statusText) statusText.textContent = `Tạo thành công toàn bộ ${state.totalSentences} câu nói.`;
                state.isGenerating = false;
                resetTabUI(tabId);
                loadFilesForTab(tabId);
                
                if (isRunningAll) {
                    processNextTabInQueue();
                } else {
                    sendDesktopNotification("🎉 KhanhTTS OmniVoice Studio", `Đã hoàn thành sinh giọng cho ${state.totalSentences} câu nói ở "${tab.name}"!`);
                }
                return;
            }

            const currentText = state.sentenceList[state.currentIndex];
            const percent = Math.round((state.currentIndex) / state.totalSentences * 100);
            
            if (progressBarFill) progressBarFill.style.width = `${percent}%`;
            if (progressPercentage) progressPercentage.textContent = `${percent}%`;
            if (progressLabel) progressLabel.textContent = `Đang xử lý câu ${state.currentIndex + 1}/${state.totalSentences}`;
            if (statusText) statusText.textContent = `Chuẩn bị sinh giọng cho: "${currentText}"`;

            const folderPath = document.getElementById(`folder-path-${tabId}`).value.trim();
            const sampleSelect = document.getElementById(`sample-select-${tabId}`);
            const refTextInput = document.getElementById(`ref-text-${tabId}`);
            const speedSlider = document.getElementById(`speed-slider-${tabId}`);
            const speedMethodSelect = document.getElementById(`speed-method-${tabId}`);
            const numberingMethodEl = document.getElementById(`numbering-method-${tabId}`);
            const startNumberInput = document.getElementById(`start-number-${tabId}`);

            const isNumberingEnabled = numberingMethodEl.value !== 'none';
            let startNumberStr = "1";
            if (numberingMethodEl.value === 'auto') {
                startNumberStr = getPrefixForIndex(startNumberInput.value.trim(), state.currentIndex);
            } else if (numberingMethodEl.value === 'custom') {
                startNumberStr = (tab.customNumbers && tab.customNumbers[state.currentIndex]) || String(state.currentIndex + 1);
            }

            const payload = {
                sentences: currentText,
                folder: folderPath,
                ref_audio: sampleSelect.value,
                ref_text: refTextInput.value,
                speed: parseFloat(speedSlider.value),
                speed_method: speedMethodSelect.value,
                enable_numbering: isNumberingEnabled,
                start_number: startNumberStr
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
                                if (statusText) statusText.textContent = "🤖 Đang khởi tạo mô hình KhanhTTS-OmniVoice...";
                            } else if (data.status === 'processing') {
                                if (statusText) statusText.textContent = `Đang sinh giọng cho: "${currentText}"`;
                            } else if (data.status === 'completed') {
                                // Completed sentence
                            } else if (data.status === 'error') {
                                if (statusText) statusText.textContent = `❌ Lỗi ở câu: "${currentText}": ${data.message}`;
                                hasError = true;
                            }
                        } catch (e) {
                            console.error("Parse error:", e, line);
                        }
                    }
                }

                if (hasError) {
                    state.isGenerating = false;
                    resetTabUI(tabId);
                    if (isRunningAll) {
                        stopRunAllQueue();
                        sendDesktopNotification("❌ KhanhTTS OmniVoice Studio - Lỗi", `Quá trình sinh giọng bị dừng do gặp lỗi ở một câu.`);
                    } else {
                        sendDesktopNotification("❌ KhanhTTS OmniVoice Studio - Lỗi", `Gặp lỗi khi sinh giọng ở "${tab.name}".`);
                    }
                    return;
                }

                state.currentIndex++;
                const nextPercent = Math.round((state.currentIndex) / state.totalSentences * 100);
                if (progressBarFill) progressBarFill.style.width = `${nextPercent}%`;
                if (progressPercentage) progressPercentage.textContent = `${nextPercent}%`;

                processNextSentenceForTab(tabId);
            } catch (err) {
                if (statusText) statusText.textContent = `❌ Lỗi kết nối: ${err}`;
                state.isGenerating = false;
                resetTabUI(tabId);
                if (isRunningAll) {
                    stopRunAllQueue();
                    sendDesktopNotification("❌ KhanhTTS OmniVoice Studio - Lỗi", `Quá trình sinh giọng bị dừng do gặp lỗi kết nối.`);
                } else {
                    sendDesktopNotification("❌ KhanhTTS OmniVoice Studio - Lỗi", `Lỗi sinh giọng ở "${tab.name}": ${err}`);
                }
            }
        }

        async function startGeneration(tabId) {
            const tab = tabs.find(t => t.id === tabId);
            if (!tab) return;
            
            const sentencesInput = document.getElementById(`sentences-${tabId}`);
            const folderPathInput = document.getElementById(`folder-path-${tabId}`);
            
            const rawSentences = sentencesInput.value.trim();
            if (!rawSentences) {
                alert("Vui lòng nhập ít nhất một câu để tạo giọng nói.");
                if (isRunningAll) stopRunAllQueue();
                return;
            }

            const folderPath = folderPathInput.value.trim();
            if (!folderPath) {
                alert("Vui lòng nhập đường dẫn thư mục lưu kết quả.");
                if (isRunningAll) stopRunAllQueue();
                return;
            }

            tab.state.sentenceList = rawSentences.split("\\n").map(s => s.trim()).filter(s => s);
            tab.state.totalSentences = tab.state.sentenceList.length;
            tab.state.currentIndex = 0;
            tab.state.isPaused = false;
            tab.state.isStopped = false;
            tab.state.isGenerating = true;

            const btnGenerate = document.getElementById(`btn-generate-${tabId}`);
            const controlGroup = document.getElementById(`control-group-${tabId}`);
            const btnPause = document.getElementById(`btn-pause-${tabId}`);
            const progressContainer = document.getElementById(`progress-container-${tabId}`);
            const progressBarFill = document.getElementById(`progress-bar-fill-${tabId}`);
            const progressPercentage = document.getElementById(`progress-percentage-${tabId}`);
            const progressLabel = document.getElementById(`progress-label-${tabId}`);
            const statusText = document.getElementById(`status-text-${tabId}`);

            if (btnGenerate) btnGenerate.style.display = 'none';
            if (controlGroup) controlGroup.style.display = 'grid';
            if (btnPause) btnPause.innerHTML = '⏸️ Tạm dừng';

            if (progressContainer) progressContainer.style.display = 'block';
            if (progressBarFill) progressBarFill.style.width = '0%';
            if (progressPercentage) progressPercentage.textContent = '0%';
            if (progressLabel) progressLabel.textContent = 'Đang chuẩn bị...';
            if (statusText) statusText.textContent = 'Bắt đầu quá trình tạo hàng loạt...';

            processNextSentenceForTab(tabId);
        }

        function togglePause(tabId) {
            const tab = tabs.find(t => t.id === tabId);
            if (!tab) return;
            
            const btnPause = document.getElementById(`btn-pause-${tabId}`);
            const statusText = document.getElementById(`status-text-${tabId}`);
            
            if (tab.state.isPaused) {
                tab.state.isPaused = false;
                if (btnPause) btnPause.innerHTML = '⏸️ Tạm dừng';
                if (statusText) statusText.textContent = 'Đang tiếp tục quá trình tạo...';
                processNextSentenceForTab(tabId);
            } else {
                tab.state.isPaused = true;
                if (btnPause) {
                    btnPause.disabled = true;
                    btnPause.innerHTML = '⌛ Đang dừng...';
                }
                if (statusText) statusText.textContent = 'Đang đợi hoàn thành câu hiện tại để tạm dừng...';
            }
        }

        function stopGeneration(tabId) {
            const tab = tabs.find(t => t.id === tabId);
            if (!tab) return;
            
            const btnStop = document.getElementById(`btn-stop-${tabId}`);
            const statusText = document.getElementById(`status-text-${tabId}`);
            
            tab.state.isStopped = true;
            if (tab.state.isPaused) {
                tab.state.isGenerating = false;
                resetTabUI(tabId);
                if (statusText) statusText.textContent = '⛔ Đang dừng hẳn...';
                loadFilesForTab(tabId);
            } else {
                if (btnStop) {
                    btnStop.disabled = true;
                    btnStop.innerHTML = '⌛ Đang dừng...';
                }
                if (statusText) statusText.textContent = 'Đang đợi hoàn thành câu hiện tại để dừng hẳn...';
            }
        }

        function runAllTabs() {
            const btnRunAll = document.getElementById('btn-run-all');
            
            if (isRunningAll) {
                if (confirm("⚠️ Bạn có chắc chắn muốn dừng quá trình sinh giọng nói trên TẤT CẢ các Tab không?")) {
                    stopRunAllQueue();
                }
                return;
            }
            
            const tabsToRun = tabs.filter(t => {
                const textEl = document.getElementById(`sentences-${t.id}`);
                const sentences = textEl ? textEl.value.trim() : '';
                return sentences.length > 0;
            });
            
            if (tabsToRun.length === 0) {
                alert("Không có tab nào chứa danh sách câu nói để chạy.");
                return;
            }
            
            isRunningAll = true;
            runAllQueue = tabsToRun.map(t => t.id);
            
            if (btnRunAll) {
                btnRunAll.textContent = "⏹️ Dừng tất cả các Tab";
                btnRunAll.classList.add('running');
            }
            
            processNextTabInQueue();
        }

        function processNextTabInQueue() {
            if (!isRunningAll) return;
            
            if (runAllQueue.length === 0) {
                isRunningAll = false;
                const btnRunAll = document.getElementById('btn-run-all');
                if (btnRunAll) {
                    btnRunAll.textContent = "🚀 Chạy tất cả các Tab";
                    btnRunAll.classList.remove('running');
                }
                sendDesktopNotification("🎉 KhanhTTS OmniVoice Studio", "Đã hoàn thành sinh giọng nói cho TOÀN BỘ các Tab!");
                alert("🎉 Đã hoàn thành sinh giọng nói cho toàn bộ các Tab!");
                return;
            }
            
            const nextTabId = runAllQueue.shift();
            selectTab(nextTabId);
            startGeneration(nextTabId);
        }

        function stopRunAllQueue() {
            isRunningAll = false;
            runAllQueue = [];
            
            const btnRunAll = document.getElementById('btn-run-all');
            if (btnRunAll) {
                btnRunAll.textContent = "🚀 Chạy tất cả các Tab";
                btnRunAll.classList.remove('running');
            }
            
            tabs.forEach(t => {
                if (t.state && t.state.isGenerating) {
                    stopGeneration(t.id);
                }
            });
        }

        window.addEventListener('DOMContentLoaded', () => {
            initTabs();
            requestNotificationPermission();
        });
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_CONTENT

@app.get("/audio-editor", response_class=HTMLResponse)
async def audio_editor_page():
    template_path = os.path.join(BASE_DIR, "templates", "audio_editor.html")
    if os.path.exists(template_path):
        with open(template_path, "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content=content)
    raise HTTPException(status_code=404, detail="Template not found")

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
    
    # Only allow "Audio sample", "Kết quả" and subfolders of "Kết quả"
    allowed_base_folders = ["Audio sample", "Kết quả"]
    for item in allowed_base_folders:
        full_path = os.path.join(BASE_DIR, item)
        if os.path.isdir(full_path):
            folders.append({
                "name": item,
                "path": normalize_path(full_path)
            })
            
            if item == "Kết quả":
                for subitem in os.listdir(full_path):
                    sub_path = os.path.join(full_path, subitem)
                    if os.path.isdir(sub_path) and not subitem.startswith("."):
                        folders.append({
                            "name": f"Kết quả / {subitem}",
                            "path": normalize_path(sub_path)
                        })
            
    sample_dir = os.path.join(BASE_DIR, "Audio sample")
    if os.path.exists(sample_dir):
        for item in os.listdir(sample_dir):
            full_path = os.path.join(sample_dir, item)
            if os.path.isfile(full_path) and (item.lower().endswith(".mp3") or item.lower().endswith(".wav")) and not item.startswith("temp_"):
                samples.append({
                    "name": item,
                    "path": normalize_path(full_path)
                })
            
    folders.sort(key=lambda x: x["name"])
    samples.sort(key=lambda x: x["name"])
    
    return {
        "base_dir": normalize_path(BASE_DIR),
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
        return {"success": True, "name": filename, "path": normalize_path(target_path)}
    except Exception as e:
        return {"success": False, "error": str(e)}

asr_loaded = False

@app.post("/api/transcribe")
async def transcribe_audio(payload: dict):
    global asr_loaded
    audio_path = normalize_path(payload.get("path")) if payload.get("path") else None
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
    target_dir = normalize_path(folder)
    if not os.path.exists(target_dir):
        return {"files": []}
        
    try:
        files = []
        for item in os.listdir(target_dir):
            full_path = os.path.join(target_dir, item)
            if os.path.isfile(full_path) and item.lower().endswith(".wav") and not item.startswith("temp_"):
                files.append({
                    "name": item,
                    "path": normalize_path(full_path)
                })
        files.sort(key=lambda x: x["name"])
        return {"files": files}
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/audio")
async def get_audio(path: str):
    full_path = normalize_path(path)
    if not (full_path.lower().endswith(".wav") or full_path.lower().endswith(".mp3")):
        raise HTTPException(status_code=403, detail="Format not allowed")
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(full_path)

@app.delete("/api/delete-audio")
async def delete_audio(path: str):
    full_path = normalize_path(path)
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
    target_dir = normalize_path(folder)
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

class RenameRequest(BaseModel):
    path: str
    new_name: str

@app.post("/api/rename-audio")
async def rename_audio(req: RenameRequest):
    old_path = normalize_path(req.path)
    new_filename = req.new_name.strip()
    
    if not (old_path.lower().endswith(".wav") or old_path.lower().endswith(".mp3")):
        raise HTTPException(status_code=403, detail="Định dạng file không được phép")
    if not os.path.exists(old_path):
        raise HTTPException(status_code=404, detail="Không tìm thấy file nguồn")
    if not new_filename:
        raise HTTPException(status_code=400, detail="Tên file mới không được để trống")
        
    ext = os.path.splitext(old_path)[1].lower()
    if not new_filename.lower().endswith(ext):
        new_filename += ext
        
    parent_dir = os.path.dirname(old_path)
    new_path = normalize_path(os.path.join(parent_dir, new_filename))
    
    if os.path.exists(new_path) and new_path != old_path:
        raise HTTPException(status_code=400, detail="File mới đã tồn tại")
        
    try:
        os.rename(old_path, new_path)
        return {"success": True, "new_path": new_path, "new_name": new_filename}
    except Exception as e:
        return {"success": False, "error": str(e)}

class GenerateRequest(BaseModel):
    sentences: str
    folder: str
    ref_audio: str
    ref_text: str
    speed: float = 1.0
    speed_method: str = "model"
    enable_numbering: bool = False
    start_number: str = "1"

@app.post("/api/generate")
async def generate_voice(req: GenerateRequest):
    async def generator() -> Generator[str, None, None]:
        try:
            yield json.dumps({"status": "model_loading"}) + "\n"
            model_instance = get_model()
            
            ref_audio_path = normalize_path(req.ref_audio)
            if not os.path.exists(ref_audio_path):
                yield json.dumps({"status": "error", "message": f"Không tìm thấy file mẫu: {req.ref_audio}"}) + "\n"
                return

            import uuid
            wav_sample = os.path.join(BASE_DIR, f"temp_sample_{uuid.uuid4().hex}.wav")
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
                    try:
                        match = re.match(r"^(.*?)(\d+)$", req.start_number)
                        if match:
                            prefix_part = match.group(1)
                            digits_part = match.group(2)
                            start_val = int(digits_part)
                            current_val = start_val + (idx - 1)
                            padded_digits = str(current_val).zfill(len(digits_part))
                            num_str = f"{prefix_part}{padded_digits}"
                        else:
                            if idx > 1:
                                num_str = f"{req.start_number}_{idx}"
                            else:
                                num_str = req.start_number
                    except Exception:
                        num_str = req.start_number
                    ten_file_sach = f"{num_str}_{ten_file_sach}"
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
                
                # Tách câu theo các dấu kết thúc câu (. ! ?) để tránh model bị nuốt chữ hoặc dừng sớm
                loop = asyncio.get_event_loop()
                def run_gen():
                    import numpy as np
                    sub_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
                    if not sub_sentences:
                        sub_sentences = [text]
                    
                    sub_audios = []
                    for sub_text in sub_sentences:
                        # Chuẩn hóa dấu ba chấm (...) và ellipsis (…) thành dấu phẩy (,) để tránh model bỏ chữ/hallucinate
                        sub_clean = sub_text.replace("…", ", ").replace("...", ", ")
                        sub_clean = re.sub(r',\s*,', ',', sub_clean)
                        sub_clean = re.sub(r'\s+', ' ', sub_clean).strip()
                        if not sub_clean:
                            continue
                        
                        if req.speed_method == "model":
                            gen_kwargs = {}
                            if req.speed and req.speed != 1.0:
                                gen_kwargs["speed"] = req.speed
                            audio = model_instance.generate(
                                text=sub_clean,
                                ref_audio=wav_sample,
                                ref_text=req.ref_text,
                                **gen_kwargs
                            )
                        else:
                            audio = model_instance.generate(
                                text=sub_clean,
                                ref_audio=wav_sample,
                                ref_text=req.ref_text
                            )
                        sub_audios.append(audio[0])
                    
                    if not sub_audios:
                        return np.zeros(24000, dtype=np.float32)
                    
                    # Ghép các đoạn âm thanh lại với nhau, chèn khoảng lặng 0.3s giữa các câu
                    silence_len = int(24000 * 0.3)
                    silence = np.zeros(silence_len, dtype=np.float32)
                    
                    combined_segments = []
                    for i, segment in enumerate(sub_audios):
                        if i > 0:
                            combined_segments.append(silence)
                        combined_segments.append(segment)
                        
                    audio_data = np.concatenate(combined_segments)
                    
                    # Nếu dùng DSP-based speed, stretch toàn bộ audio sau khi ghép
                    if req.speed_method != "model" and req.speed and req.speed != 1.0:
                        audio_data = librosa.effects.time_stretch(audio_data, rate=req.speed, n_fft=512)
                        
                    return audio_data
                
                try:
                    audio_data = await loop.run_in_executor(None, run_gen)
                    sf.write(file_output, audio_data, 24000)
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
