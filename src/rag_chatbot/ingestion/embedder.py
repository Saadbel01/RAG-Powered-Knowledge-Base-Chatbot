import time
import structlog
from pinecone import Pinecone
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


@retry(stop=stop_after_attempt(3),
       wait=wait_exponential(multiplier=1, min=1, max=8))
def _upsert_batch(store: PineconeVectorStore,
                  batch: list[Document]) -> None:
    store.add_documents(batch)


def embed_and_store(chunks: list[Document]) -> PineconeVectorStore:
    cfg = get_settings()
    batch_size = 64
    embeddings = get_embedding_model()

    log.info("embedder.start", total_chunks=len(chunks), batch_size=batch_size)

    for chunk in chunks:
        chunk.metadata["text"] = chunk.page_content

    # The first batch creates the Pinecone connection
    first_batch = chunks[:batch_size]
    store = PineconeVectorStore.from_documents(
        documents=first_batch,
        embedding=embeddings,
        index_name=cfg.pinecone_index_name,
    )

    remaining = chunks[batch_size:]
    total_batches = (len(remaining) + batch_size - 1) // batch_size

    for i in tqdm(
        range(0, len(remaining), batch_size),
        total=total_batches,
        desc="Embedding batches",
    ):
        _upsert_batch(store, remaining[i: i + batch_size])
        time.sleep(0.2)

    log.info("embedder.done", stored=len(chunks))
    return store


def fetch_all_chunks_from_pinecone() -> list[Document]:
    """Fetch every stored chunk from Pinecone to populate BM25 in memory."""
    cfg = get_settings()
    pc = Pinecone(api_key=cfg.pinecone_api_key)
    idx = pc.Index(cfg.pinecone_index_name)

    documents: list[Document] = []
    # Page through vector IDs stored in the index
    for id_batch in idx.list(limit=100):
        response = idx.fetch(ids=id_batch)
        for vid, vec in response.vectors.items():
            meta = vec.metadata or {}
            documents.append(
                Document(
                    page_content=meta.get("text", ""),
                    metadata={k: v for k, v in meta.items() if k != "text"},
                )
            )
    log.info("corpus.loaded", total=len(documents))
    return documents


def load_existing_store() -> PineconeVectorStore:
    cfg = get_settings()
    return PineconeVectorStore(
        index_name=cfg.pinecone_index_name,
        embedding=get_embedding_model(),
        pinecone_api_key=cfg.pinecone_api_key
    )
