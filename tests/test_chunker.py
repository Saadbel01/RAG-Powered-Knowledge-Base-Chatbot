from langchain.schema import Document

from rag_chatbot.config import get_settings
from rag_chatbot.ingestion.chunker import create_chunks


def make_doc(text: str) -> Document:

    return Document(
        page_content=text,
        metadata={
            "source": "test.txt",
            "page": 0,
        },
    )


def test_long_document_produces_multiple_chunks():

    long_text = "This is a sentence. " * 100
    docs = [make_doc(long_text)]

    chunks = create_chunks(docs)

    assert len(chunks) > 1, (
        "A long document should produce multiple chunks"
    )


def test_every_chunk_has_required_metadata():

    docs = [
        make_doc("Some content for testing metadata propagation.")
    ]

    chunks = create_chunks(docs)

    for chunk in chunks:
        assert "chunk_id" in chunk.metadata, "chunk_id is missing"
        assert "char_count" in chunk.metadata, "char_count is missing"
        assert "source" in chunk.metadata, "source is missing"


def test_chunk_size_is_not_wildly_exceeded():

    docs = [make_doc("word " * 500)]

    chunks = create_chunks(docs)
    cfg = get_settings()

    # Generous buffer for the splitting behavior
    max_allowed = (
        cfg.chunk_size
        + cfg.chunk_overlap
        + 50
    )

    for chunk in chunks:
        assert chunk.metadata["char_count"] <= max_allowed, (
            f"Chunk {chunk.metadata['chunk_id']} exceeds size limit: "
            f"{chunk.metadata['char_count']} > {max_allowed}"
        )