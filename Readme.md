# 🎙️ Audio-RAG-Summarizer

> **End-to-end pipeline that transcribes audio, builds a RAG index, and generates intelligent summaries using Llama.**

---

## Architecture

```
Audio File
   │
   ▼
Audio Pre-processing  (src/audio.py)
   │  – resample → 16 kHz mono WAV
   ▼
Faster-Whisper        (src/transcriber.py)
   │  – word-level timestamps, language detection
   ▼
Transcript Loader     (src/loader.py)
   │  – loads .txt / .json transcript
   ▼
Sentence Chunker      (src/chunker.py)
   │  – SentenceSplitter, configurable chunk/overlap size
   ▼
BGE Embedder          (src/embedder.py)
   │  – BAAI/bge-small-en-v1.5 (local)
   ▼
Vector Store          (src/vector_store.py)
   │  – LlamaIndex + FAISS index, persisted to data/index/
   ▼
Dense Retriever       (src/retriever.py)
   │  – top-k similarity search
   ▼
Prompt Builder        (src/prompt.py)
   │  – zero-shot summarization prompt
   ▼
Llama LLM             (src/llm.py)
   │  – OpenAI-compatible Llama API
   ▼
Summarizer            (src/summarizer.py)
   │  – orchestrates retrieval + generation
   ▼
Output                (outputs/summaries/)
```

---

## Quick Start

### 1. Clone & install

```bash
git clone <repo-url>
cd Audio-RAG-Summarizer
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env .env.local   # then fill in your keys
```

| Variable | Description |
|---|---|
| `LLAMA_API_KEY` | API key for Llama / Groq / OpenAI-compatible endpoint |
| `LLAMA_API_BASE` | Base URL of the API |
| `HF_TOKEN` | HuggingFace token (optional, for gated models) |

### 3. Run

```bash
# Full pipeline: transcribe → index → summarize
python main.py --audio data/audio/lecture.mp3

# Skip transcription (use existing transcript)
python main.py --transcript data/transcripts/lecture.txt

# Custom query
python main.py --audio data/audio/meeting.mp3 --query "What were the key decisions?"
```

---

## Project Structure

```
Audio-RAG-Summarizer/
├── main.py                  # CLI entry point
├── requirements.txt
├── .env                     # API keys (never commit)
├── data/
│   ├── audio/               # Input audio files
│   ├── transcripts/         # Whisper output (.txt / .json)
│   ├── embeddings/          # Cached embeddings
│   └── index/               # Persisted FAISS index
├── models/                  # Optional local model weights
├── src/
│   ├── config.py            # Central configuration
│   ├── audio.py             # Audio loading & preprocessing
│   ├── transcriber.py       # Faster-Whisper wrapper
│   ├── loader.py            # Transcript loader
│   ├── chunker.py           # SentenceSplitter
│   ├── embedder.py          # BGE embedding model
│   ├── vector_store.py      # FAISS VectorStoreIndex
│   ├── retriever.py         # Dense retriever
│   ├── prompt.py            # Prompt templates
│   ├── llm.py               # Llama API client
│   ├── summarizer.py        # Orchestration
│   ├── utils.py             # Helpers & logging
│   └── evaluation.py        # ROUGE / BERTScore metrics
├── outputs/
│   ├── summaries/           # Generated summaries (.md / .txt)
│   └── logs/                # Pipeline run logs
└── tests/
    ├── test_chunker.py
    ├── test_retriever.py
    └── test_llm.py
```

---

## Evaluation

```bash
python -m src.evaluation --summary outputs/summaries/out.txt --reference ref.txt
```

Outputs ROUGE-1/2/L and BERTScore F1.

---

## License

MIT
