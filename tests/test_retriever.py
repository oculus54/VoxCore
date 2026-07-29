from llama_index.core import Document
from src.chunker import chunk_documents
from src.vector_store import build_index
from src.retriever import get_retriever, retrieve


def test_retrieve_returns_nodes():
    doc = Document(text="Networking is the practice of connecting computers. " * 20)
    nodes = chunk_documents([doc], chunk_size=50, chunk_overlap=10)
    index = build_index(nodes)
    retriever = get_retriever(index, top_k=3)

    results = retrieve(retriever, "What is networking?")

    assert len(results) <= 3
    assert all(hasattr(r, "score") for r in results)
