"""
main.py  –  Audio-RAG-Summarizer entry point
Usage:
    python main.py <youtube_url> [query]
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

from src.preprocessing import run_preprocessing
from src.rag import run_rag
from src.evaluation import evaluate_run
from src.utils import get_logger

load_dotenv()
logger = get_logger("main")


def run_pipeline(youtube_url: str, query: str = "Summarize the entire video.") -> str:
    # ── 1. Preprocess: download → WAV → normalize → VAD → transcribe ──────────
    logger.info("Starting preprocessing...")
    result = run_preprocessing(youtube_url)
    logger.info(f"Preprocessed: '{result['title']}' ({result['duration']}s)")

    # ── 2. RAG: chunk → embed → index → retrieve → summarize ─────────────────
    logger.info("Running RAG pipeline...")
    summary, retrieved_nodes = run_rag(result["transcript_path"], query)

    # ── 3. Evaluate ───────────────────────────────────────────────────────────
    evaluate_run(query, retrieved_nodes, summary)

    # ── 4. Save & print ───────────────────────────────────────────────────────
    out_path = Path("outputs/summaries") / f"{result['title'][:50]}.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(summary, encoding="utf-8")
    logger.info(f"Summary saved → {out_path}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(summary)

    return summary


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <youtube_url> [query]")
        sys.exit(1)

    url = sys.argv[1]
    q = sys.argv[2] if len(sys.argv) > 2 else "Summarize the entire video."
    run_pipeline(url, q)
