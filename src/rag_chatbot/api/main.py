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

_metrics = defaultdict(int)

app = FastAPI()


@app.get("/health")
def get_health(request: Request) -> HealthResponse:
    if hasattr(request.app.state, "chain"):
        return HealthResponse(status="ok", pinecone=True, groq=True)
    else:
        return HealthResponse(status="degraded", pinecone=False, groq=False)


@app.post("/ask")
def ask_request(ask_request: AskRequest, request: Request) -> AskResponse:
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
