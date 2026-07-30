from faster_whisper import WhisperModel
from pathlib import Path

def transcribe_audio(audio_path, output_dir="downloads/transcript", language=None, device="cuda"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Model initialized inside the function to keep this file strictly functional
    model = WhisperModel(
        "large-v3",
        device=device,
        compute_type="float16" if device == "cuda" else "int8"
    )

    segments, info = model.transcribe(
        str(audio_path),
        beam_size=5,
        vad_filter=False, 
        language=language
    )

    transcript = []
    segments_data = []

    print("=" * 70)
    print("TRANSCRIPT")
    print("=" * 70)

    for segment in segments:
        text = segment.text.strip()
        transcript.append(text)
        
        segments_data.append({
            "start": round(segment.start, 2),
            "end": round(segment.end, 2),
            "text": text
        })
        print(text)

    transcript_text = "\n".join(transcript)
    txt_file = output_dir / "transcript.txt"

    with open(txt_file, "w", encoding="utf-8") as f:
        f.write(transcript_text)

    print("\n")
    print("=" * 70)
    print("Language :", info.language)
    print("Duration :", round(info.duration, 2), "seconds")
    print("Saved to :", txt_file)

    return transcript_text, segments_data
