"""
src/rag.py
==========
Full RAG pipeline in one file:
  load transcript → chunk → embed → index → retrieve → prompt → LLM → summary
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from llama_index.core import SimpleDirectoryReader, Settings, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from src import config

load_dotenv()

# ── Lazy singletons ────────────────────────────────────────────────────────────
_embed_model = None
_llm_client = None

SUMMARY_PROMPT = """\
You are an expert AI assistant specialized in summarizing transcripts.
Use ONLY the information provided in the context below. Do not invent anything.

-------------------- CONTEXT --------------------
{context}
-------------------------------------------------

Task — provide a structured summary:
1. Main Topic
2. Key Concepts
3. Important Discussions
4. Conclusions
5. Key Takeaways

Keep it concise and under 300 words.
"""


# ── Embed model ────────────────────────────────────────────────────────────────
def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = HuggingFaceEmbedding(model_name=config.EMBED_MODEL)
        Settings.embed_model = _embed_model
        print("[rag] Embedding model loaded.")
    return _embed_model


# ── LLM client ────────────────────────────────────────────────────────────────
def _get_llm_client() -> InferenceClient:
    global _llm_client
    if _llm_client is None:
        token = os.getenv("HF_TOKEN")
        if not token:
            raise EnvironmentError("HF_TOKEN not set in .env")
        _llm_client = InferenceClient(api_key=token)
        print("[rag] HuggingFace client initialized.")
    return _llm_client


# ── RAG steps ─────────────────────────────────────────────────────────────────
def load_transcript(transcript_path: str | Path) -> list:
    docs = SimpleDirectoryReader(input_files=[str(transcript_path)]).load_data()
    print(f"[rag] Loaded {len(docs)} document(s).")
    return docs


def chunk_documents(documents: list) -> list:
    splitter = SentenceSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
    nodes = splitter.get_nodes_from_documents(documents)
    print(f"[rag] {len(nodes)} chunks created.")
    return nodes


def build_index(nodes: list) -> VectorStoreIndex:
    _get_embed_model()
    index = VectorStoreIndex(nodes)
    print(f"[rag] Index built with {len(nodes)} nodes.")
    return index


def retrieve(index: VectorStoreIndex, query: str, top_k: int = config.TOP_K) -> list:
    retriever = index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(query)
    print(f"[rag] Retrieved {len(nodes)} chunks for query: '{query}'")
    return nodes


def generate(prompt: str) -> str:
    client = _get_llm_client()
    response = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=config.LLM_TEMPERATURE,
        max_tokens=config.LLM_MAX_TOKENS,
    )
    return response.choices[0].message.content


# ── Full RAG pipeline ──────────────────────────────────────────────────────────
def run_rag(transcript_path: str | Path, query: str = "Summarize the entire video.") -> tuple[str, list]:
    """
    Load transcript → chunk → index → retrieve → prompt → generate.
    Returns (summary_text, retrieved_nodes).
    """
    documents = load_transcript(transcript_path)
    nodes = chunk_documents(documents)
    index = build_index(nodes)
    retrieved = retrieve(index, query)

    context = "\n\n".join(n.text for n in retrieved)
    prompt = SUMMARY_PROMPT.format(context=context)
    summary = generate(prompt)

    return summary, retrieved
