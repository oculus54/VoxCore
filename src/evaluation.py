"""
Lightweight RAG evaluation — no external eval framework dependency.
Two things worth measuring here: retrieval quality and summary faithfulness.
"""

from src.utils import get_logger

logger = get_logger(__name__)


def retrieval_score_stats(retrieved_nodes):
    """Basic stats on retriever similarity scores — sanity check for retrieval quality."""
    scores = [node.score for node in retrieved_nodes if node.score is not None]
    if not scores:
        return {"count": 0, "min": None, "max": None, "avg": None}

    return {
        "count": len(scores),
        "min": round(min(scores), 4),
        "max": round(max(scores), 4),
        "avg": round(sum(scores) / len(scores), 4),
    }


def context_coverage(summary, retrieved_nodes, min_word_len=5):
    """
    Rough faithfulness proxy: what fraction of 'significant' words (len >= min_word_len)
    in the summary also appear somewhere in the retrieved context.
    Not a substitute for a real faithfulness eval (e.g. LLM-as-judge) — just a cheap sanity signal.
    """
    context_text = " ".join(node.text for node in retrieved_nodes).lower()
    context_words = set(w.strip(".,!?\"'") for w in context_text.split())

    summary_words = [w.strip(".,!?\"'") for w in summary.lower().split() if len(w) >= min_word_len]
    if not summary_words:
        return 0.0

    matched = sum(1 for w in summary_words if w in context_words)
    return round(matched / len(summary_words), 4)


def evaluate_run(query, retrieved_nodes, summary):
    """Run all lightweight checks and log a report."""
    stats = retrieval_score_stats(retrieved_nodes)
    coverage = context_coverage(summary, retrieved_nodes)

    logger.info(f"Query: {query}")
    logger.info(f"Retrieval scores: {stats}")
    logger.info(f"Context coverage: {coverage}")

    return {"query": query, "retrieval_stats": stats, "context_coverage": coverage}
