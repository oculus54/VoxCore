import ffmpeg
from pathlib import Path

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
