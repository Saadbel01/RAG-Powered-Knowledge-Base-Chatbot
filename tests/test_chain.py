from langchain.schema import Document

from rag_chatbot.generation.chain import format_docs


def test_format_docs_with_empty_list():
    """Empty input must return the fallback 'no documents' message."""
    result = format_docs([])

    assert "No relevant" in result


def test_format_docs_includes_source_and_content():
    """Output must contain the document label, source, and text."""
    doc = Document(
        page_content="The voltage limit is 48V.",
        metadata={
            "source": "manual.pdf",
            "page": 3,
        },
    )

    result = format_docs([doc])

    assert "Document 1" in result
    assert "manual.pdf" in result
    assert "The voltage limit is 48V." in result


def test_format_docs_separator_between_multiple_docs():
    """Multiple documents must be separated by the --- divider."""
    docs = [
        Document(
            page_content=f"Content {i}",
            metadata={"source": f"doc{i}.txt"},
        )
        for i in range(3)
    ]

    result = format_docs(docs)

    assert "Document 1" in result
    assert "Document 3" in result
    assert "---" in result


def test_format_docs_missing_page_uses_source_only():
    """
    When 'page' is absent from metadata, only the source
    name should be shown.
    """
    doc = Document(
        page_content="Some text.",
        metadata={
            "source": "notes.txt",
        },
    )

    result = format_docs([doc])

    assert "notes.txt" in result
    assert "p." not in result
