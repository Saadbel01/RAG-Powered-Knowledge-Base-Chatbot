import sys
import structlog
from rag_chatbot.config import get_settings
from rag_chatbot.ingestion.chunker import create_chunks
from rag_chatbot.ingestion.embedder import embed_and_store
from rag_chatbot.ingestion.loader import load_documents


logger = structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.dev.ConsoleRenderer(True)
    ]
)

log = structlog.get_logger(__name__) 

def run_ingestion() -> None:
    cfg = get_settings()
    log.info("=== RAG CHATTBOT - INGESTION PIPELINE ===")
    
    log.info("Start loading the documents ...")
    documents = load_documents(cfg.data_folder)
    if not documents:
        raise ValueError("Error: Enable to load the documents.")
    log.info("Documents loaded successfully")

    log.info("Start chunking the documents ...")
    chunks = create_chunks(documents)
    if not chunks:
        raise ValueError("Error: Enable to create chunks.")    
    log.info("Chunking done successfully")

    log.info("Start embedding and storing the chunks ...")
    embed_and_store(chunks)    
    log.info("Embedding and Storing the chunks done successfully")


if __name__ == "__main__":
    run_ingestion()
