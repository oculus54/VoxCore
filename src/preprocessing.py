"""
src/preprocessing.py
====================
Audio preprocessing pipeline:
  download (yt-dlp) → WAV conversion (ffmpeg) → normalize → VAD silence removal (Silero) → transcribe (Faster-Whisper)
"""

from pathlib import Path

import ffmpeg
import torchaudio
import yt_dlp
from faster_whisper import WhisperModel
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps, collect_chunks

from src import config

# ── Lazy model cache ───────────────────────────────────────────────────────────
_whisper_model = None


def _get_whisper():
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = WhisperModel(
            config.WHISPER_MODEL,
            device=config.WHISPER_DEVICE,
            compute_type=config.WHISPER_COMPUTE_TYPE,
        )
    return _whisper_model


# ── Step 1: Download ───────────────────────────────────────────────────────────
def download_audio(youtube_url: str, output_dir: Path = config.AUDIO_DIR) -> tuple[Path, dict]:
    """Download best audio from YouTube URL via yt-dlp."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(output_dir / "%(id)s.%(ext)s"),
        "quiet": False,
        "noplaylist": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)

    downloaded = next(
        (f for f in output_dir.iterdir() if f.stem == info["id"]), None
    )
    return downloaded, info


# ── Step 2: Convert to 16 kHz mono WAV ────────────────────────────────────────
def convert_to_wav(input_file: Path, output_dir: Path = config.AUDIO_DIR) -> Path:
    output_file = Path(output_dir) / f"{input_file.stem}.wav"
    (
        ffmpeg
        .input(str(input_file))
        .output(str(output_file), ac=1, ar=config.SAMPLE_RATE, format="wav")
        .overwrite_output()
        .run(quiet=True)
    )
    return output_file


# ── Step 3: Loudness normalisation ────────────────────────────────────────────
def normalize_audio(wav_path: Path) -> Path:
    out = wav_path.with_name(wav_path.stem + "_norm.wav")
    (
        ffmpeg
        .input(str(wav_path))
        .output(str(out), af=config.LOUDNORM_FILTER)
        .overwrite_output()
        .run(quiet=True)
    )
    return out


# ── Step 4: Silence removal (Silero VAD) ──────────────────────────────────────
def remove_silence(wav_path: Path) -> tuple[Path, list]:
    out = wav_path.with_name(wav_path.stem + "_speech.wav")
    model = load_silero_vad()
    wav = read_audio(str(wav_path), sampling_rate=config.SAMPLE_RATE)
    timestamps = get_speech_timestamps(
        wav, model,
        sampling_rate=config.SAMPLE_RATE,
        threshold=config.VAD_THRESHOLD,
        min_speech_duration_ms=config.VAD_MIN_SPEECH_MS,
        min_silence_duration_ms=config.VAD_MIN_SILENCE_MS,
        speech_pad_ms=config.VAD_SPEECH_PAD_MS,
    )
    speech = collect_chunks(timestamps, wav)
    torchaudio.save(str(out), speech.unsqueeze(0), config.SAMPLE_RATE)
    return out, timestamps


# ── Step 5: Transcribe ─────────────────────────────────────────────────────────
def transcribe(audio_path: Path, language: str | None = None) -> tuple[str, list]:
    """Transcribe with Faster-Whisper; saves transcript to data/transcripts/."""
    model = _get_whisper()
    segments, info = model.transcribe(str(audio_path), beam_size=5, vad_filter=False, language=language)

    lines, segments_data = [], []
    for seg in segments:
        text = seg.text.strip()
        lines.append(text)
        segments_data.append({"start": round(seg.start, 2), "end": round(seg.end, 2), "text": text})

    transcript_text = "\n".join(lines)

    out_path = config.TRANSCRIPT_DIR / "transcript.txt"
    out_path.write_text(transcript_text, encoding="utf-8")
    print(f"[transcribe] lang={info.language} | duration={info.duration:.1f}s | saved → {out_path}")
    return transcript_text, segments_data


# ── Full pipeline ──────────────────────────────────────────────────────────────
def run_preprocessing(youtube_url: str, language: str | None = None) -> dict:
    """
    Download → WAV → normalize → VAD → transcribe.
    Returns a dict with title, clean_audio path, transcript text & path.
    """
    print("⬇  Downloading audio...")
    raw, meta = download_audio(youtube_url)

    print("🔄  Converting to WAV...")
    wav = convert_to_wav(raw)

    print("🔊  Normalizing loudness...")
    norm = normalize_audio(wav)

    print("🔇  Removing silence...")
    clean, timestamps = remove_silence(norm)

    print("📝  Transcribing...")
    transcript_text, segments = transcribe(clean, language=language)

    return {
        "title": meta.get("title", "unknown"),
        "duration": meta.get("duration"),
        "clean_audio": clean,
        "transcript_path": config.TRANSCRIPT_DIR / "transcript.txt",
        "transcript_text": transcript_text,
        "segments": segments,
    }
