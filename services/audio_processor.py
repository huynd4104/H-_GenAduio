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
    """Trims, standardizes, and merges multiple audio clips using FFmpeg mixing."""
    if not clips_data:
        raise ValueError("No audio clips to merge.")
        
    inputs = []
    filter_parts = []
    
    # Calculate total timeline duration of the export
    total_duration = 0.1
    for clip in clips_data:
        clip_end = clip["timelineStart"] + (clip["sourceEnd"] - clip["sourceStart"])
        if clip_end > total_duration:
            total_duration = clip_end

    for idx, clip in enumerate(clips_data):
        file_path = clip["filePath"]
        timeline_start = clip["timelineStart"]
        source_start = clip["sourceStart"]
        source_end = clip["sourceEnd"]
        volume = clip.get("volume", 1.0)
        
        play_dur = max(0.001, source_end - source_start)
        delay_ms = int(round(timeline_start * 1000))
        
        # Add input trimmed source clip
        inputs.extend(["-ss", f"{source_start:.3f}", "-t", f"{play_dur:.3f}", "-i", file_path])
        
        # Build filter chain: resample -> volume -> adelay
        filter_parts.append(
            f"[{idx}:a]aresample=44100,aformat=sample_fmts=fltp:channel_layouts=stereo,"
            f"volume={volume:.2f},adelay={delay_ms}|{delay_ms}[d{idx}]"
        )
        
    # Append silence padding as the last input
    silent_idx = len(clips_data)
    inputs.extend(["-f", "lavfi", "-t", f"{total_duration:.3f}", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100"])
    
    # Resample the silent input
    filter_parts.append(f"[{silent_idx}:a]aresample=44100,aformat=sample_fmts=fltp:channel_layouts=stereo[d{silent_idx}]")
    
    # Mix delayed streams and the silence padding
    mix_inputs = "".join(f"[d{i}]" for i in range(silent_idx + 1))
    filter_parts.append(f"{mix_inputs}amix=inputs={silent_idx + 1}:duration=longest:normalize=0[outa]")
    
    filter_complex = "; ".join(filter_parts)
    
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
