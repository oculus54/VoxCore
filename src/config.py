from pathlib import Path

# Directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
AUDIO_DIR = DATA_DIR / "audio"
TRANSCRIPT_DIR = DATA_DIR / "transcripts"

for d in (AUDIO_DIR, TRANSCRIPT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Audio preprocessing
SAMPLE_RATE = 16000
LOUDNORM_FILTER = "loudnorm=I=-16:LRA=11:TP=-1.5"

VAD_THRESHOLD = 0.5
VAD_MIN_SPEECH_MS = 250
VAD_MIN_SILENCE_MS = 500
VAD_SPEECH_PAD_MS = 200

# Whisper
WHISPER_MODEL = "large-v3"
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE_TYPE = "float16"
# Embedding
EMBED_MODEL = "BAAI/bge-base-en-v1.5"

# Chunking
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

# Retrieval
TOP_K = 10

# LLM (Hugging Face Inference API)
LLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS = 700