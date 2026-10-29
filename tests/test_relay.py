"""Unit and integration tests for upstream LLM relay and response persistence."""

import json
import os
import sqlite3
import pytest
import httpx
from fastapi.testclient import TestClient

from open_context_circulator.engine import ContextEngine
from open_context_circulator.relay import (
    RelayConfig,
    RelayError,
    UpstreamRelay,
    UpstreamResponseError,
    UpstreamTimeoutError,
    UpstreamUnavailableError,
    extract_assistant_message,
)
from open_context_circulator.server import (
    app,
    set_engine,
    set_relay,
    set_store,
)
from open_context_circulator.store import MessageStore


class TestRelayErrors:
    """Tests for RelayError hierarchy and attributes."""

    def test_relay_error_base(self):
        """Test base RelayError attributes and defaults."""
        err = RelayError("Generic relay error", status_code=502, code="custom_code", details={"foo": "bar"})
        assert str(err) == "Generic relay error"
        assert err.message == "Generic relay error"
        assert err.status_code == 502
        assert err.code == "custom_code"
        assert err.details == {"foo": "bar"}

    def test_upstream_error_subclasses(self):
        """Test that specialized errors inherit from RelayError and set proper codes."""
        unavail = UpstreamUnavailableError("Unavailable", details={"retry": False})
        assert isinstance(unavail, RelayError)
        assert unavail.status_code == 502
        assert unavail.code == "upstream_unavailable"

        timeout = UpstreamTimeoutError("Timeout")
        assert isinstance(timeout, RelayError)
        assert timeout.status_code == 504
        assert timeout.code == "upstream_timeout"

        resp_err = UpstreamResponseError("Bad gateway", status_code=502)
        assert isinstance(resp_err, RelayError)
        assert resp_err.status_code == 502
        assert resp_err.code == "upstream_response_error"


class TestRelayConfig:
    """Tests for RelayConfig loading and defaults."""

    def test_default_config(self):
        """Test default relay configuration values."""
        config = RelayConfig()
        assert config.upstream_url == "http://127.0.0.1:8045/v1"
        assert config.api_key is None
        assert config.timeout == 30.0
        assert config.default_model == "default"

    def test_from_env_defaults(self, monkeypatch):
        """Test from_env when no environment variables are set."""
        monkeypatch.delenv("UPSTREAM_URL", raising=False)
        monkeypatch.delenv("UPSTREAM_API_KEY", raising=False)
        monkeypatch.delenv("UPSTREAM_TIMEOUT", raising=False)
        monkeypatch.delenv("UPSTREAM_MODEL", raising=False)

        config = RelayConfig.from_env()
        assert config.upstream_url == "http://127.0.0.1:8045/v1"
        assert config.api_key is None
        assert config.timeout == 30.0
        assert config.default_model == "default"

    def test_from_env_custom(self, monkeypatch):
        """Test from_env with custom environment variables."""
        monkeypatch.setenv("UPSTREAM_URL", "https://api.openai.com/v1")
        monkeypatch.setenv("UPSTREAM_API_KEY", "sk-custom-test-key")
        monkeypatch.setenv("UPSTREAM_TIMEOUT", "45.5")
        monkeypatch.setenv("UPSTREAM_MODEL", "gpt-4o")

        config = RelayConfig.from_env()
        assert config.upstream_url == "https://api.openai.com/v1"
        assert config.api_key == "sk-custom-test-key"
        assert config.timeout == 45.5
        assert config.default_model == "gpt-4o"

    def test_custom_instantiation(self):
        """Test direct initialization of RelayConfig."""
        config = RelayConfig(
            upstream_url="http://custom-host:9000/v1",
            api_key="direct-key",
            timeout=15.0,
            default_model="claude-3-haiku",
        )
        assert config.upstream_url == "http://custom-host:9000/v1"
        assert config.api_key == "direct-key"
        assert config.timeout == 15.0
        assert config.default_model == "claude-3-haiku"


class TestExtractAssistantMessage:
    """Tests for extract_assistant_message helper."""

    def test_extract_standard_openai_format(self):
        """Test extracting message from OpenAI-style choices array."""
        payload = {
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello from upstream!",
                    },
                    "finish_reason": "stop",
                }
            ]
        }
        extracted = extract_assistant_message(payload)
        assert extracted == {
            "role": "assistant",
            "content": "Hello from upstream!",
        }

    def test_extract_legacy_choices_text(self):
        """Test extracting message from legacy text choice."""
        payload = {
            "choices": [
                {
                    "index": 0,
                    "text": "Legacy completion response",
                }
            ]
        }
        extracted = extract_assistant_message(payload)
        assert extracted == {
            "role": "assistant",
            "content": "Legacy completion response",
        }

    def test_extract_occ_messages_format(self):
        """Test extracting message from OCC messages array format."""
        payload = {
            "messages": [
                {"role": "assistant", "content": "OCC format message"}
            ]
        }
        extracted = extract_assistant_message(payload)
        assert extracted == {
            "role": "assistant",
            "content": "OCC format message",
        }

    def test_extract_direct_message(self):
        """Test extracting message from top-level message field."""
        payload = {
            "message": {"role": "assistant", "content": "Direct message"}
        }
        extracted = extract_assistant_message(payload)
        assert extracted == {
            "role": "assistant",
            "content": "Direct message",
        }

    def test_extract_direct_content(self):
        """Test extracting message from top-level content field."""
        payload = {"content": "Direct string content"}
        extracted = extract_assistant_message(payload)
        assert extracted == {
            "role": "assistant",
            "content": "Direct string content",
        }


class TestUpstreamRelayUnit:
    """Unit tests for UpstreamRelay client logic."""

    def test_target_url_resolution(self):
        """Test resolving target URL across variations."""
        r1 = UpstreamRelay(RelayConfig(upstream_url="http://127.0.0.1:8045/v1"))
        assert r1.get_target_url() == "http://127.0.0.1:8045/v1/chat/completions"

        r2 = UpstreamRelay(RelayConfig(upstream_url="http://127.0.0.1:8045/v1/"))
        assert r2.get_target_url() == "http://127.0.0.1:8045/v1/chat/completions"

        r3 = UpstreamRelay(RelayConfig(upstream_url="http://127.0.0.1:8045/v1/chat/completions"))
        assert r3.get_target_url() == "http://127.0.0.1:8045/v1/chat/completions"

    def test_headers_with_and_without_api_key(self):
        """Test header construction with and without auth token."""
        no_key = UpstreamRelay(RelayConfig(api_key=None))
        assert no_key.get_headers() == {"Content-Type": "application/json"}

        with_key = UpstreamRelay(RelayConfig(api_key="secret-token-123"))
        assert with_key.get_headers() == {
            "Content-Type": "application/json",
            "Authorization": "Bearer secret-token-123",
        }

    @pytest.mark.anyio
    async def test_forward_chat_completion_success(self):
        """Test successful forward call with mock transport."""
        captured_request = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured_request["url"] = str(request.url)
            captured_request["headers"] = dict(request.headers)
            captured_request["body"] = json.loads(request.read())
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-unit-1",
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "Unit test response",
                            },
                        }
                    ],
                },
            )

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            relay = UpstreamRelay(
                config=RelayConfig(
                    upstream_url="http://test-upstream/v1",
                    api_key="my-api-key",
                    default_model="test-default-model",
                ),
                client=client,
            )

            res = await relay.forward_chat_completion(
                messages=[{"role": "user", "content": "Ping"}],
                temperature=0.7,
                extra_params={"stream": False},
            )

            assert res["id"] == "chatcmpl-unit-1"
            assert captured_request["url"] == "http://test-upstream/v1/chat/completions"
            assert captured_request["headers"]["authorization"] == "Bearer my-api-key"
            assert captured_request["body"]["model"] == "test-default-model"
            assert captured_request["body"]["temperature"] == 0.7
            assert captured_request["body"]["stream"] is False
            assert captured_request["body"]["messages"] == [{"role": "user", "content": "Ping"}]

    @pytest.mark.anyio
    async def test_forward_chat_completion_connection_error(self):
        """Test that httpx.ConnectError is mapped to UpstreamUnavailableError."""
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("Connection refused by target")

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            relay = UpstreamRelay(client=client)
            with pytest.raises(UpstreamUnavailableError) as exc_info:
                await relay.forward_chat_completion(messages=[{"role": "user", "content": "Hi"}])
            assert exc_info.value.status_code == 502
            assert exc_info.value.code == "upstream_unavailable"

    @pytest.mark.anyio
    async def test_forward_chat_completion_timeout(self):
        """Test that httpx.TimeoutException is mapped to UpstreamTimeoutError."""
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("Read timed out")

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            relay = UpstreamRelay(client=client)
            with pytest.raises(UpstreamTimeoutError) as exc_info:
                await relay.forward_chat_completion(messages=[{"role": "user", "content": "Hi"}])
            assert exc_info.value.status_code == 504
            assert exc_info.value.code == "upstream_timeout"

    @pytest.mark.anyio
    async def test_forward_chat_completion_upstream_http_error(self):
        """Test that upstream HTTP 500 status raises UpstreamResponseError."""
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="Internal Server Error from upstream LLM")

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            relay = UpstreamRelay(client=client)
            with pytest.raises(UpstreamResponseError) as exc_info:
                await relay.forward_chat_completion(messages=[{"role": "user", "content": "Hi"}])
            assert exc_info.value.status_code == 502
            assert exc_info.value.code == "upstream_response_error"
            assert "500" in exc_info.value.message


class TestIntegrationRelayAndPersistence:
    """End-to-end integration tests verifying forwarding, response persistence in SQLite, and error handling."""

    @pytest.fixture
    def test_db(self, tmp_path):
        """Provide isolated SQLite database."""
        db_file = tmp_path / "test_occ.db"
        store = MessageStore(str(db_file))
        set_store(store)
        set_engine(ContextEngine(max_tokens=4000))
        yield store, db_file
        set_store(None)
        set_engine(None)
        set_relay(None)

    def test_end_to_end_forwarding_and_response_persistence(self, test_db):
        """Verify chat completion forwards virtualized context and persists assistant response in SQLite."""
        store, db_file = test_db
        session_id = "test-session-persist"

        # Mock upstream handler returning an OpenAI completion response
        captured_payloads = []

        def upstream_handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.read())
            captured_payloads.append(body)
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-persisted-123",
                    "object": "chat.completion",
                    "created": 1700000000,
                    "model": "upstream-mock-model",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "Hello! I received your virtualized context.",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 8,
                        "total_tokens": 18,
                    },
                },
            )

        mock_client = httpx.AsyncClient(transport=httpx.MockTransport(upstream_handler))
        relay = UpstreamRelay(
            config=RelayConfig(
                upstream_url="http://127.0.0.1:8045/v1",
                api_key="secret-key-occ",
                default_model="upstream-mock-model",
            ),
            client=mock_client,
        )
        set_relay(relay)

        client = TestClient(app)

        # 1. Send chat completion request
        request_body = {
            "session_id": session_id,
            "messages": [
                {"role": "user", "content": "Hello from client turn 1"}
            ],
            "max_context_tokens": 4000,
        }

        response = client.post("/v1/chat/completions", json=request_body)

        # 2. Check HTTP response
        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == session_id
        assert len(data["messages"]) == 1
        assert data["messages"][0]["role"] == "assistant"
        assert data["messages"][0]["content"] == "Hello! I received your virtualized context."

        # Verify passthrough fields
        assert data["id"] == "chatcmpl-persisted-123"
        assert data["choices"][0]["message"]["content"] == "Hello! I received your virtualized context."
        assert data["usage"]["total_tokens"] == 18
        assert "context_info" in data
        assert data["context_info"]["total_tokens"] > 0

        # 3. Check what was sent to upstream
        assert len(captured_payloads) == 1
        assert captured_payloads[0]["messages"] == [
            {"role": "user", "content": "Hello from client turn 1"}
        ]

        # 4. Verify SQLite MessageStore persistence
        turns = store.get_turns(session_id)
        assert len(turns) == 2
        # Turn 1: user
        assert turns[0]["role"] == "user"
        assert turns[0]["content"] == "Hello from client turn 1"
        # Turn 2: assistant (recorded back from upstream)
        assert turns[1]["role"] == "assistant"
        assert turns[1]["content"] == "Hello! I received your virtualized context."
        assert turns[1]["token_count"] > 0

        # Check session table
        session = store.get_session(session_id)
        assert session is not None
        assert session["total_tokens"] > 0

    def test_multi_turn_loopback_context_persistence(self, test_db):
        """Verify that consecutive turns build up history in SQLite and forward complete history."""
        store, _ = test_db
        session_id = "test-session-multiturn"
        captured_messages = []

        turn_count = 0

        def upstream_handler(request: httpx.Request) -> httpx.Response:
            nonlocal turn_count
            turn_count += 1
            body = json.loads(request.read())
            captured_messages.append(body["messages"])
            return httpx.Response(
                200,
                json={
                    "id": f"chatcmpl-turn-{turn_count}",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": f"Response to turn {turn_count}",
                            },
                        }
                    ],
                },
            )

        mock_client = httpx.AsyncClient(transport=httpx.MockTransport(upstream_handler))
        relay = UpstreamRelay(client=mock_client)
        set_relay(relay)

        client = TestClient(app)

        # First turn
        res1 = client.post(
            "/v1/chat/completions",
            json={
                "session_id": session_id,
                "messages": [{"role": "user", "content": "Message 1"}],
            },
        )
        assert res1.status_code == 200

        # Second turn
        res2 = client.post(
            "/v1/chat/completions",
            json={
                "session_id": session_id,
                "messages": [{"role": "user", "content": "Message 2"}],
            },
        )
        assert res2.status_code == 200

        # Check SQLite contains 4 turns: user1, assistant1, user2, assistant2
        turns = store.get_turns(session_id)
        assert len(turns) == 4
        assert [t["role"] for t in turns] == ["user", "assistant", "user", "assistant"]
        assert [t["content"] for t in turns] == [
            "Message 1",
            "Response to turn 1",
            "Message 2",
            "Response to turn 2",
        ]

        # Verify what was sent to upstream in second turn includes earlier history
        assert len(captured_messages[1]) == 3  # user1, assistant1, user2
        assert captured_messages[1][0] == {"role": "user", "content": "Message 1"}
        assert captured_messages[1][1] == {"role": "assistant", "content": "Response to turn 1"}
        assert captured_messages[1][2] == {"role": "user", "content": "Message 2"}

    def test_graceful_error_handling_upstream_unavailable(self, test_db):
        """Verify HTTP 502 response and contract error shape when upstream connection fails."""
        store, _ = test_db
        session_id = "test-session-unavailable"

        def failing_handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("Failed to establish a new connection: Connection refused")

        mock_client = httpx.AsyncClient(transport=httpx.MockTransport(failing_handler))
        relay = UpstreamRelay(client=mock_client)
        set_relay(relay)

        client = TestClient(app)

        response = client.post(
            "/v1/chat/completions",
            json={
                "session_id": session_id,
                "messages": [{"role": "user", "content": "Can you hear me?"}],
            },
        )

        assert response.status_code == 502
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == "upstream_unavailable"
        assert "Connection refused" in data["error"]["message"]
        assert "details" in data["error"]

        # Verify no assistant turn was erroneously stored
        turns = store.get_turns(session_id)
        assert len(turns) == 1
        assert turns[0]["role"] == "user"

    def test_graceful_error_handling_upstream_timeout(self, test_db):
        """Verify HTTP 504 response when upstream call times out."""
        store, _ = test_db
        session_id = "test-session-timeout"

        def timeout_handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("Upstream socket read timed out")

        mock_client = httpx.AsyncClient(transport=httpx.MockTransport(timeout_handler))
        relay = UpstreamRelay(client=mock_client)
        set_relay(relay)

        client = TestClient(app)

        response = client.post(
            "/v1/chat/completions",
            json={
                "session_id": session_id,
                "messages": [{"role": "user", "content": "Wait for me"}],
            },
        )

        assert response.status_code == 504
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == "upstream_timeout"
        assert "details" in data["error"]

    def test_graceful_error_handling_upstream_500_response(self, test_db):
        """Verify HTTP 502 response when upstream returns 500 error."""
        store, _ = test_db
        session_id = "test-session-500"

        def error_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={"error": "LLM cluster overload"})

        mock_client = httpx.AsyncClient(transport=httpx.MockTransport(error_handler))
        relay = UpstreamRelay(client=mock_client)
        set_relay(relay)

        client = TestClient(app)

        response = client.post(
            "/v1/chat/completions",
            json={
                "session_id": session_id,
                "messages": [{"role": "user", "content": "Trigger 500"}],
            },
        )

        assert response.status_code == 502
        data = response.json()
        assert "error" in data
        assert data["error"]["code"] == "upstream_response_error"
        assert "500" in data["error"]["message"]

    def test_configurable_upstream_env_vars(self, test_db, monkeypatch):
        """Verify that relay picks up UPSTREAM_URL and UPSTREAM_API_KEY from environment."""
        monkeypatch.setenv("UPSTREAM_URL", "http://custom-proxy.internal:9999/v1")
        monkeypatch.setenv("UPSTREAM_API_KEY", "env-api-token-987")

        captured_request = {}

        def custom_handler(request: httpx.Request) -> httpx.Response:
            captured_request["url"] = str(request.url)
            captured_request["auth"] = request.headers.get("authorization")
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {"message": {"role": "assistant", "content": "Configured via env"}}
                    ]
                },
            )

        mock_client = httpx.AsyncClient(transport=httpx.MockTransport(custom_handler))
        # Create relay loading from env
        relay = UpstreamRelay(client=mock_client)
        set_relay(relay)

        client = TestClient(app)
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello env"}],
            },
        )

        assert response.status_code == 200
        assert captured_request["url"] == "http://custom-proxy.internal:9999/v1/chat/completions"
        assert captured_request["auth"] == "Bearer env-api-token-987"
        assert response.json()["messages"][0]["content"] == "Configured via env"
