import os
import yt_dlp
import ffmpeg
from pathlib import Path
OUTPUT_DIR = Path("downloads")
OUTPUT_DIR.mkdir(exist_ok=True)

def download_audio(youtube_url, output_dir="downloads"):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    output_template = str(output_dir / "%(id)s.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "quiet": False,
        "noplaylist": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)

    downloaded_file = None

    for file in output_dir.iterdir():
        if file.stem == info["id"]:
            downloaded_file = file
            break

    return downloaded_file, info

def convert_to_wav(input_file, output_dir="downloads"):
    output_dir = Path(output_dir)

    output_file = output_dir / f"{input_file.stem}.wav"

    (
        ffmpeg
        .input(str(input_file))
        .output(
            str(output_file),
            ac=1,          # mono
            ar=16000,      # 16 kHz
            format="wav"
        )
        .overwrite_output()
        .run(quiet=True)
    )

    return output_file

youtube_url = "https://youtu.be/keeqnciDVOo?si=Ca9bwY9pOcBW48fq"

audio_file, metadata = download_audio(youtube_url)

wav_file = convert_to_wav(audio_file)

print("Title :", metadata["title"])
print("Duration :", metadata["duration"], "seconds")
print("Saved :", wav_file)
