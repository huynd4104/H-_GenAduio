import os
import uuid
import json
import tempfile
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict

from services.audio_processor import get_audio_duration, merge_audio_clips

router = APIRouter()

# In-memory database of active audio ID mappings
# id -> absolute file path
audio_id_map: Dict[str, str] = {}

class ClipInput(BaseModel):
    audioId: str
    trimStart: float
    trimEnd: float

class ExportRequest(BaseModel):
    folder: str
    clips: List[ClipInput]

class SaveProjectRequest(BaseModel):
    project_path: str
    clips: List[ClipInput]

@router.get("/api/audio-editor/files")
async def get_audio_files(folder: str = Query(..., description="Absolute path of the folder to scan")):
    global audio_id_map
    target_dir = os.path.abspath(folder)
    if not os.path.exists(target_dir):
        raise HTTPException(status_code=400, detail="Thư mục không tồn tại")
        
    try:
        files = []
        supported_exts = {".mp3", ".wav", ".m4a", ".ogg"}
        
        # Scan files in directory
        for item in os.listdir(target_dir):
            full_path = os.path.join(target_dir, item)
            if os.path.isfile(full_path):
                ext = os.path.splitext(item)[1].lower()
                if ext in supported_exts and not item.startswith("temp_"):
                    # Check if already has an ID, otherwise generate one
                    found_id = None
                    for k, v in audio_id_map.items():
                        if v == full_path:
                            found_id = k
                            break
                    if not found_id:
                        found_id = str(uuid.uuid4())
                        audio_id_map[found_id] = full_path
                        
                    duration = get_audio_duration(full_path)
                    files.append({
                        "id": found_id,
                        "filename": item,
                        "duration": duration
                    })
        files.sort(key=lambda x: x["filename"])
        return files
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/audio-editor/stream")
async def stream_audio_file(id: str = Query(..., description="Internal ID of the audio file")):
    file_path = audio_id_map.get(id)
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Không tìm thấy file âm thanh")
    return FileResponse(file_path)

@router.post("/api/audio-editor/export")
async def export_audio(req: ExportRequest):
    if not req.clips:
        raise HTTPException(status_code=400, detail="Không có clip nào trong dòng thời gian")
        
    try:
        resolved_clips = []
        for clip in req.clips:
            path = audio_id_map.get(clip.audioId)
            if not path or not os.path.exists(path):
                raise HTTPException(status_code=400, detail=f"Không tìm thấy file cho clip ID: {clip.audioId}")
            resolved_clips.append({
                "filePath": path,
                "trimStart": clip.trimStart,
                "trimEnd": clip.trimEnd
            })
            
        # Write to system temporary directory to prevent creating folders in the user's workspace
        temp_export_dir = os.path.join(tempfile.gettempdir(), "genaudio_temp_exports")
        os.makedirs(temp_export_dir, exist_ok=True)
        
        unique_prefix = str(uuid.uuid4())[:8]
        output_wav = os.path.join(temp_export_dir, f"export_{unique_prefix}.wav")
        output_mp3 = os.path.join(temp_export_dir, f"export_{unique_prefix}.mp3")
        
        merge_audio_clips(resolved_clips, output_wav, output_mp3)
        
        # Register the outputs in the map so they can be streamed
        wav_id = str(uuid.uuid4())
        mp3_id = str(uuid.uuid4())
        audio_id_map[wav_id] = output_wav
        audio_id_map[mp3_id] = output_mp3
        
        return {
            "success": True,
            "message": "Đã xuất file âm thanh thành công!",
            "wav_url": f"/api/audio-editor/stream?id={wav_id}",
            "mp3_url": f"/api/audio-editor/stream?id={mp3_id}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/audio-editor/save-project")
async def save_project(req: SaveProjectRequest):
    project_path = os.path.abspath(req.project_path)
    try:
        # Map IDs to actual filenames for portability
        project_clips = []
        for clip in req.clips:
            path = audio_id_map.get(clip.audioId)
            if not path:
                raise HTTPException(status_code=400, detail="ID âm thanh không hợp lệ")
            filename = os.path.basename(path)
            project_clips.append({
                "filename": filename,
                "trimStart": clip.trimStart,
                "trimEnd": clip.trimEnd
            })
            
        project_data = {"clips": project_clips}
        # Ensure parent folder exists
        os.makedirs(os.path.dirname(project_path), exist_ok=True)
        with open(project_path, "w", encoding="utf-8") as f:
            json.dump(project_data, f, ensure_ascii=False, indent=2)
            
        return {"success": True, "message": "Đã lưu dự án thành công!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/audio-editor/load-project")
async def load_project(path: str = Query(..., description="Absolute path to project JSON file")):
    project_path = os.path.abspath(path)
    if not os.path.exists(project_path):
        raise HTTPException(status_code=404, detail="Không tìm thấy file dự án")
        
    try:
        # Scan folder containing project JSON to resolve file paths
        target_dir = os.path.dirname(project_path)
        supported_exts = {".mp3", ".wav", ".m4a", ".ogg"}
        local_filename_to_id = {}
        for item in os.listdir(target_dir):
            full_path = os.path.join(target_dir, item)
            if os.path.isfile(full_path):
                ext = os.path.splitext(item)[1].lower()
                if ext in supported_exts and not item.startswith("temp_"):
                    # Check ID
                    found_id = None
                    for k, v in audio_id_map.items():
                        if v == full_path:
                            found_id = k
                            break
                    if not found_id:
                        found_id = str(uuid.uuid4())
                        audio_id_map[found_id] = full_path
                    local_filename_to_id[item] = found_id
                    
        with open(project_path, "r", encoding="utf-8") as f:
            project_data = json.load(f)
            
        loaded_clips = []
        for clip in project_data.get("clips", []):
            fname = clip["filename"]
            audio_id = local_filename_to_id.get(fname)
            if audio_id:
                # Calculate duration
                dur = get_audio_duration(audio_id_map[audio_id])
                loaded_clips.append({
                    "audioId": audio_id,
                    "filename": fname,
                    "duration": dur,
                    "trimStart": clip["trimStart"],
                    "trimEnd": clip["trimEnd"]
                })
                
        return {"success": True, "clips": loaded_clips, "folder": target_dir}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

import asyncio

@router.post("/api/audio-editor/browse-save-project")
async def browse_save_project():
    try:
        cmd = """osascript -e 'POSIX path of (choose file name with prompt "Chọn nơi lưu dự án:" default name "project.json")'"""
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

@router.post("/api/audio-editor/browse-load-project")
async def browse_load_project():
    try:
        cmd = """osascript -e 'POSIX path of (choose file with prompt "Chọn file dự án JSON để tải:" of type {"json"})'"""
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

@router.post("/api/audio-editor/open-temp-dir")
async def open_temp_dir():
    try:
        temp_dir = os.path.join(tempfile.gettempdir(), "genaudio_temp_exports")
        os.makedirs(temp_dir, exist_ok=True)
        # On macOS, use the open command to bring up Finder
        proc = await asyncio.create_subprocess_exec("open", temp_dir)
        await proc.wait()
        return {"success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
