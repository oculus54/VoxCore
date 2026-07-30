import yt_dlp
from pathlib import Path

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
