import torch
from pathlib import Path
import ffmpeg

print("CUDA Available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("Running on CPU")
    
def normalize_audio(input_audio):
    """
    Normalize audio loudness using FFmpeg's loudnorm filter.

    Args:
        input_audio (str or Path): Input WAV file.

    Returns:
        Path: Path to normalized WAV file.
    """

    input_audio = Path(input_audio)

    output_audio = input_audio.with_name(
        input_audio.stem + "_normalized.wav"
    )

    (
        ffmpeg
        .input(str(input_audio))
        .output(
            str(output_audio),
            af="loudnorm=I=-16:LRA=11:TP=-1.5"
        )
        .overwrite_output()
        .run(quiet=True)
    )

    return output_audio
    
normalized_audio = normalize_audio(wav_file)

print(normalized_audio)
