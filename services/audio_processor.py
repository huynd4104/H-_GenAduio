import subprocess
import os

def get_audio_duration(file_path: str) -> float:
    """Gets the duration of an audio file using ffprobe (with pydub fallback)."""
    try:
        cmd = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", file_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        return float(result.stdout.strip())
    except Exception:
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(file_path)
            return audio.duration_seconds
        except Exception as e:
            print(f"Error getting duration for {file_path}: {e}")
            return 0.0

def merge_audio_clips(clips_data: list, output_wav: str, output_mp3: str):
    """Trims, standardizes, and merges multiple audio clips using FFmpeg."""
    if not clips_data:
        raise ValueError("No audio clips to merge.")
        
    inputs = []
    filter_parts = []
    
    for idx, clip in enumerate(clips_data):
        file_path = clip["filePath"]
        trim_start = clip["trimStart"]
        trim_end = clip["trimEnd"]
        duration = max(0.001, trim_end - trim_start) # avoid 0 or negative duration
        
        inputs.extend(["-ss", f"{trim_start:.3f}", "-t", f"{duration:.3f}", "-i", file_path])
        filter_parts.append(f"[{idx}:a]aresample=44100,aformat=sample_fmts=fltp:channel_layouts=stereo[a{idx}]")
        
    concat_inputs = "".join(f"[a{idx}]" for idx in range(len(clips_data)))
    concat_filter = f"{concat_inputs}concat=n={len(clips_data)}:v=0:a=1[outa]"
    
    filter_complex = "; ".join(filter_parts) + "; " + concat_filter
    
    # Run FFmpeg to output WAV
    cmd_wav = [
        "ffmpeg", "-y"
    ] + inputs + [
        "-filter_complex", filter_complex,
        "-map", "[outa]",
        output_wav
    ]
    
    result_wav = subprocess.run(cmd_wav, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result_wav.returncode != 0:
        raise RuntimeError(f"FFmpeg WAV merge failed: {result_wav.stderr}")
        
    # Transcode WAV to MP3
    cmd_mp3 = [
        "ffmpeg", "-y", "-i", output_wav,
        "-codec:a", "libmp3lame", "-qscale:a", "2",
        output_mp3
    ]
    result_mp3 = subprocess.run(cmd_mp3, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result_mp3.returncode != 0:
        raise RuntimeError(f"FFmpeg MP3 transcoding failed: {result_mp3.stderr}")
