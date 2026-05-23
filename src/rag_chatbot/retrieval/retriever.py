from rag_chatbot.config import get_settings
from langchain_pinecone import PineconeVectorStore
from rank_bm25 import BM25Okapi
from rag_chatbot.ingestion.chunker import create_chunks
from rag_chatbot.ingestion.loader import load_documents
from rag_chatbot.ingestion.embedder import get_embedding_model
import numpy as np
from sentence_transformers import CrossEncoder
import structlog


log = structlog.get_logger(__name__)
_CROSS_ENCODER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def reciprocal_rank_fusion(dense: list, sparse: list, k=60) -> list:
    rrf_dict = {}
    dense_rrf = {}
    sparse_rrf = {}
    doc_map = {}
    for i, d in enumerate(dense, start=1):
        dense_rrf[d.metadata["chunk_id"]] = 1/(k + i)
        doc_map[d.metadata["chunk_id"]] = d
    for i, d in enumerate(sparse, start=1):
        sparse_rrf[d.metadata["chunk_id"]] = 1/(k + i)
        doc_map[d.metadata["chunk_id"]] = d

    unique_documents = set(
        [d.metadata["chunk_id"] for d in dense]
        ).union(set(
            [d.metadata["chunk_id"] for d in sparse])
    )
    for d in unique_documents:
        rrf_dict[d] = 0
        if d in dense_rrf.keys():
            rrf_dict[d] += dense_rrf[d]
        if d in sparse_rrf.keys():
            rrf_dict[d] += sparse_rrf[d]
    return [doc_map[d[0]] for d in sorted(rrf_dict.items(),
                                          key=lambda x: x[1], reverse=True)]


class HybridRetriever:

    def __init__(self, all_chunks=None):
        self._chunks = all_chunks
        self._cfg = get_settings()
        self._store = self._build_store()
        self._bm25 = None
        if self._chunks:
            corpus = [c.page_content.lower().split() for c in self._chunks]
            self._bm25 = BM25Okapi(corpus)

    def _build_store(self) -> PineconeVectorStore:

        return PineconeVectorStore(
            index_name=self._cfg.pinecone_index_name,
            embedding=get_embedding_model(),
            pinecone_api_key=self._cfg.pinecone_api_key
        )

    def _semantic_searh(self, query: str, k: int) -> list:
        return self._store.similarity_search(query, k)

    def _bm25_search(self, query: str, k: int) -> list:
        if not self._bm25:
            return []
        res = self._bm25.get_scores(query.lower().split())
        top_indices = np.argsort(res)[::-1]
        return [self._chunks[id] for id in top_indices[:k]]

    def retrieve(self, query, k=None):
        k = k or self._cfg.top_k
        fetch_k = k * 2
        dense = self._semantic_searh(query, fetch_k)
        sparse = self._bm25_search(query, fetch_k)
        fused = reciprocal_rank_fusion(dense, sparse)
        pool = fused[:k*3]
        if not pool:
            return dense[:k]
        pairs = [(query, chunk.page_content) for chunk in pool]
        scores = _CROSS_ENCODER.predict(pairs)
        ranked = sorted(zip(scores, pool), key=lambda x: x[0], reverse=True)
        result = [doc for _, doc in ranked[:k]]
        log.info("Retrieve Done", nb_chunks=k)
        return result

    def invoke(self, query):
        return self.retrieve(query)


if __name__ == "__main__":
    chunks = create_chunks(load_documents())
    retriver = HybridRetriever(chunks)
    res = retriver._bm25_search("what is agent", 3)
    for i, chunk in enumerate(res):
        print(f"{i}: {chunk}")
