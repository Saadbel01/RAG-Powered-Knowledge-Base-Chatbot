from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from rag_chatbot.api.main import app


@pytest.fixture
def client():
    """
    Create a TestClient with mocked app.state.

    By setting app.state.chain and app.state.retriever before yielding,
    we bypass the lifespan startup function and its external calls.

    The chain mock returns a fixed answer string so tests are deterministic.
    """
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "The voltage limit is 48V."

    with TestClient(app) as c:
        app.state.chain = mock_chain
        app.state.retriever = MagicMock()

        yield c


def test_health_endpoint_returns_ok(client):
    """GET /health must return HTTP 200 with status 'ok'."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ask_returns_expected_shape(client):
    """
    POST /ask must return HTTP 200 with answer,
    latency_ms, and request_id.
    """
    response = client.post(
        "/ask",
        json={
            "question": "What is the voltage limit?",
            "user_id": "test",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "answer" in data
    assert "latency_ms" in data
    assert "request_id" in data
    assert isinstance(data["latency_ms"], int)


def test_ask_rejects_short_question(client):
    """Questions shorter than 5 characters must return HTTP 422."""
    response = client.post(
        "/ask",
        json={
            "question": "Hi",
            "user_id": "test",
        },
    )

    # Pydantic's min_length=5 triggers 422 Unprocessable Entity
    assert response.status_code == 422


def test_metrics_endpoint_returns_counters(client):
    """GET /metrics must return HTTP 200 with total_requests field."""
    response = client.get("/metrics")

    assert response.status_code == 200
    assert "total_requests" in response.json()
