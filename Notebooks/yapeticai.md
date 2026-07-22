# Yapetic-AI: Audio Processing & Transcription Pipeline

This script automates the extraction, optimization, and transcription of audio from YouTube videos. It utilizes a machine-learning stack to deliver clean, normalized audio and accurate text transcripts.

## Core Architecture

* **Extraction:** Pulls the best available audio stream from YouTube via `yt-dlp`.


* **Standardization:** Downmixes the source media into a 16kHz Mono WAV file using `ffmpeg`.


* **Normalization:** Balances audio loudness utilizing `ffmpeg`'s loudnorm filter parameters (I=-16:LRA=11:TP=-1.5).


* **Silence Removal:** Strips non-speech segments using the `silero-vad` model (threshold set to 0.5).


* **Transcription:** Executes speech-to-text conversion with `faster-whisper` using the `large-v3` model. It computes in `float16` and requires a CUDA-enabled GPU.



## Requirements

System-level FFmpeg and a CUDA-capable GPU are required.

Install the required Python packages:

```bash
apt-get install ffmpeg
pip install yt-dlp ffmpeg-python torch torchaudio silero-vad faster-whisper

```

## Usage

The execution flow is split into preprocessing and transcription:

```python
from pathlib import Path

# 1. Input the target URL
youtube_url = "https://youtu.be/keeqnciDVOo?si=Ca9bwY9pOcBW48fq"

# 2. Run the preprocessing pipeline (Download, Convert, Normalize, VAD)
result = preprocess_audio(youtube_url)

# 3. Transcribe the processed audio
transcript, segments = transcribe_audio(
    result["clean_audio"],
    language="en"
)

```

## Outputs

* **Audio files:** Saves all processed iterations into a `downloads` directory, terminating with a `_speech.wav` file containing only active voice.


* **Text file:** Dumps the final transcript into `downloads/transcript/transcript.txt`.


* **Metadata:** Returns internal dictionaries containing timestamps, duration, language, and original video title data.