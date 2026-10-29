"""Tests for FastAPI server endpoints."""

import pytest
from fastapi.testclient import TestClient

from open_context_circulator.server import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app with a mock relay."""
    from open_context_circulator.server import set_relay

    class MockRelay:
        async def forward_chat_completion(self, messages, model=None, **kwargs):
            return {
                "id": "chatcmpl-mock",
                "object": "chat.completion",
                "created": 123456789,
                "model": model or "mock-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Context received. Mock upstream assistant response.",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"total_tokens": 42},
            }

    mock = MockRelay()
    set_relay(mock)
    test_client = TestClient(app)
    yield test_client
    set_relay(None)


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_check(self, client):
        """Test health check returns healthy status."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data

    def test_health_response_model(self, client):
        """Test health endpoint returns correct response model."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert set(data.keys()) == {"status", "timestamp"}


class TestChatCompletionsEndpoint:
    """Tests for /v1/chat/completions endpoint."""

    def test_chat_completions_basic(self, client):
        """Test basic chat completion request."""
        request_data = {
            "messages": [
                {"role": "user", "content": "Hello"}
            ]
        }
        
        response = client.post("/v1/chat/completions", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert "messages" in data
        assert "context_info" in data
        assert len(data["messages"]) == 1
        assert data["messages"][0]["role"] == "assistant"

    def test_chat_completions_with_session_id(self, client):
        """Test chat completion with provided session ID."""
        session_id = "test-session-123"
        request_data = {
            "session_id": session_id,
            "messages": [
                {"role": "user", "content": "Test message"}
            ]
        }
        
        response = client.post("/v1/chat/completions", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == session_id

    def test_chat_completions_multiple_messages(self, client):
        """Test chat completion with multiple messages."""
        request_data = {
            "messages": [
                {"role": "user", "content": "First"},
                {"role": "assistant", "content": "Second"},
                {"role": "user", "content": "Third"}
            ]
        }
        
        response = client.post("/v1/chat/completions", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert "context_info" in data
        assert "total_tokens" in data["context_info"]

    def test_chat_completions_context_info(self, client):
        """Test that context_info contains expected fields."""
        request_data = {
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
            "max_context_tokens": 4000
        }
        
        response = client.post("/v1/chat/completions", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        context_info = data["context_info"]
        assert "total_tokens" in context_info
        assert "window_start" in context_info
        assert "window_end" in context_info
        assert "rolled_turns" in context_info
        assert "utilization" in context_info

    def test_chat_completions_invalid_request(self, client):
        """Test that empty messages list is rejected."""
        request_data = {
            "messages": []
        }
        
        response = client.post("/v1/chat/completions", json=request_data)
        
        # Should fail validation (min_length=1)
        assert response.status_code == 422
