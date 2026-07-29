from llama_index.core import Document
from src.chunker import chunk_documents


def test_chunk_documents_basic():
    doc = Document(text="This is a test sentence. " * 100)
    nodes = chunk_documents([doc], chunk_size=50, chunk_overlap=10)

    assert len(nodes) > 1
    assert all(node.text for node in nodes)


def test_chunk_documents_empty():
    nodes = chunk_documents([], chunk_size=300, chunk_overlap=50)
    assert nodes == []
