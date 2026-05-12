import time
import structlog
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm
from rag_chatbot.config import get_settings
import os


cfg = get_settings()
os.environ["PINECONE_API_KEY"] = cfg.pinecone_api_key

log = structlog.get_logger(__name__)

def get_embedding_model() -> HuggingFaceEmbeddings:
    cfg = get_settings()

    return HuggingFaceEmbeddings(
        model_name=cfg.embed_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}   
    )

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def _upsert_batch(store: PineconeVectorStore,
                  batch: list[Document]) -> None:
    store.add_documents(batch)

def embed_and_store(chunks: list[Document]) -> PineconeVectorStore:
    cfg = get_settings()
    batch_size = 64
    log.info("embedder.start", total_chunk=len(chunks))
    embeded_model = get_embedding_model()
    first_batch = chunks[:64]

    store = PineconeVectorStore.from_documents(
        documents=first_batch,
        embedding=embeded_model,
        index_name=cfg.pinecone_index_name,
        pinecone_api_key=cfg.pinecone_api_key)
    for i in tqdm(range(64, len(chunks), batch_size),
                  total=(len(chunks) - 64 + batch_size - 1) // batch_size,
                  desc="Embedding batches"):
        chunk_batch = chunks[i: i + batch_size]
        _upsert_batch(store, chunk_batch)
        time.sleep(0.2)
    log.info("embedder.done", stored=len(chunks))
    return store


def load_existing_store() -> PineconeVectorStore:
    cfg = get_settings()
    return PineconeVectorStore(
        index_name=cfg.pinecone_index_name,
        embedding=get_embedding_model(),
        pinecone_api_key=cfg.pinecone_api_key
    )