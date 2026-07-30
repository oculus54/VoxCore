import torch
import torchaudio
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps, collect_chunks
from pathlib import Path

def remove_silence_vad(input_audio):
    input_audio = Path(input_audio)
    output_audio = input_audio.with_name(
        input_audio.stem + "_speech.wav"
    )

    model = load_silero_vad()
    wav = read_audio(str(input_audio), sampling_rate=16000)

    speech_timestamps = get_speech_timestamps(
        wav,
        model,
        sampling_rate=16000,
        threshold=0.5,
        min_speech_duration_ms=250,
        min_silence_duration_ms=500,
        speech_pad_ms=200,
    )

    speech = collect_chunks(speech_timestamps, wav)

    torchaudio.save(
        str(output_audio),
        speech.unsqueeze(0),
        16000
    )

    return output_audio, speech_timestamps

def timestamps_to_seconds(timestamps, sample_rate=16000):
    return [
        {
            "start": round(t["start"] / sample_rate, 2),
            "end": round(t["end"] / sample_rate, 2)
        }
        for t in timestamps
    ]
