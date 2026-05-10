from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from rag_chatbot.config import get_settings
import structlog


log = structlog.get_logger(__name__)

def create_chunks(documents: list[Document]) -> list[Document]:
    cfg = get_settings()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
        add_start_index=True
    )
    chunks = splitter.split_documents(documents)
    
    if not chunks:
        log.erro("chunker.empty", reason="No chunks produced.")
        return []

    for idx ,chunk in enumerate(chunks):
        chunk.metadata.update({
            "chunk_id": idx,
            "char_count": len(chunk.page_content)
        })

    chunks_char_count = [chunk.metadata["char_count"] for chunk in chunks]
    log.info(
        "Chunk statistics",
        total_chunks=len(chunks),
        avg_character_length=round(sum(chunks_char_count) / len(chunks)),
        min_character_length=min(chunks_char_count),
        max_character_length=max(chunks_char_count)
    )
    return chunks
    
    