import torch
from download import download_audio
from convert import convert_to_wav
from normalize import normalize_audio
from vad import remove_silence_vad, timestamps_to_seconds
from transcribe import transcribe_audio

def preprocess_audio(youtube_url):
    print("=" * 60)
    print("Downloading audio...")
    audio_file, metadata = download_audio(youtube_url)

    print("Converting to WAV...")
    wav_file = convert_to_wav(audio_file)

    print("Normalizing audio...")
    normalized_audio = normalize_audio(wav_file)

    print("Removing silence...")
    clean_audio, timestamps = remove_silence_vad(normalized_audio)

    print("=" * 60)
    print("Audio preprocessing completed!")
    print("=" * 60)

    return {
        "title": metadata["title"],
        "duration": metadata["duration"],
        "clean_audio": clean_audio,
        "timestamps": timestamps_to_seconds(timestamps),
        "metadata": metadata
    }

def main():
    youtube_url = input("Enter The Youtube URL:")
    
    # GPU/CPU Detection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("CUDA Available:", torch.cuda.is_available())
    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("Running on CPU")

    # 1. Preprocess Pipeline
    result = preprocess_audio(youtube_url)
    
    print("\nTitle:", result["title"])
    print("Duration:", result["duration"], "seconds")
    print("Clean Audio Path:", result["clean_audio"])
    print("Sample Timestamps:", result["timestamps"][:5])

    # 2. Transcription Pipeline
    transcript, segments = transcribe_audio(
        result["clean_audio"],
        language="en",
        device=device
    )

if __name__ == "__main__":
    main()
