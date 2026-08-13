import uuid
from contextlib import asynccontextmanager
from collections import defaultdict
from fastapi import FastAPI, HTTPException, Request
from rag_chatbot.config import get_settings
from rag_chatbot.retrieval.retriever import HybridRetriever
from rag_chatbot.generation.chain import create_rag_chain, ask_with_retry
from rag_chatbot.api.models import AskRequest, AskResponse
from rag_chatbot.api.models import HealthResponse, MetricsResponse
from rag_chatbot.api.middleware import rate_limiter
import structlog

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.dev.ConsoleRenderer(colors=True),
    ]
)

log = structlog.get_logger(__name__)

_metrics: dict[str, int | float] = defaultdict(int)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown logic.
    Everything between the start of this function and the 'yield' statement
    runs at startup, before any request is handled. Everything after 'yield'
    runs at shutdown.
    """
    log.info("startup.begin")
    cfg = get_settings()

    retriever = HybridRetriever()
    app.state.retriever = retriever

    app.state.chain = create_rag_chain(retriever)

    log.info(
        "startup.done",
        model=cfg.groq_model,
        index=cfg.pinecone_index_name,
    )
    yield

    log.info("shutdown.complete")

app = FastAPI(
    title="RAG Chatbot API",
    version="0.1.0",
    description="Answers questions about your documents using RAG.",
    lifespan=lifespan

)


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health(request: Request) -> HealthResponse:
    if hasattr(request.app.state, "chain"):
        return HealthResponse(status="ok", pinecone=True, groq=True)
    else:
        return HealthResponse(status="degraded", pinecone=False, groq=False)


@app.post("/ask", response_model=AskResponse, tags=["rag"])
def ask(ask_request: AskRequest, request: Request) -> AskResponse:
    request_id = str(uuid.uuid4())
    rate_limiter.check(ask_request.user_id)
    try:
        response = ask_with_retry(request.app.state.chain,
                                  ask_request.question)
    except Exception:
        raise HTTPException(status_code=503)

    _metrics["total_requests"] += 1
    _metrics["groq_calls"] += 1
    _metrics["total_latency"] += response["latency_ms"]

    return AskResponse(
        answer=response["answer"],
        latency_ms=response["latency_ms"],
        request_id=request_id
    )


@app.get("/metrics", response_model=MetricsResponse, tags=["ops"])
async def metrics() -> MetricsResponse:
    """Exposes simple request counters for monitoring dashboards."""
    total = _metrics["total_requests"]
    hits = _metrics["cache_hits"]
    groq = _metrics["groq_calls"]
    lat = _metrics["total_latency"]

    return MetricsResponse(
        total_requests=total,
        cache_hits=hits,
        cache_hit_rate=round(hits / total, 3) if total else 0.0,
        avg_latency_ms=round(lat / groq, 1) if groq else 0.0,
        groq_calls=groq
    )


def start() -> None:
    import uvicorn

    uvicorn.run(
        "rag_chatbot.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )
