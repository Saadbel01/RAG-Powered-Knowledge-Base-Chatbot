from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=5, max_length=1000)
    user_id: str = "anonymous"


class AskResponse(BaseModel):
    answer: str
    latency_ms: int
    request_id: str


class HealthResponse(BaseModel):
    status: str
    pinecone: bool
    groq: bool


class MetricsResponse(BaseModel):
    total_requests: int
    cache_hits: int
    cache_hit_rate: float
    avg_latency_ms: float
    groq_calls: int
