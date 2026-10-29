"""FastAPI proxy server for context virtualization and upstream LLM forwarding."""

import os
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from .engine import ContextEngine
from .relay import (
    RelayConfig,
    RelayError,
    UpstreamRelay,
    extract_assistant_message,
)
from .store import MessageStore


def _get_default_max_context_tokens() -> int:
    val = os.environ.get("OCC_MAX_CONTEXT_TOKENS")
    if val:
        try:
            return int(val)
        except ValueError:
            pass
    return 4000


class ChatRequest(BaseModel):
    """Chat completion request model supporting standard OpenAI and OCC fields."""
    model_config = ConfigDict(extra="allow")

    session_id: Optional[str] = None
    messages: list[dict] = Field(..., min_length=1)
    max_context_tokens: int = Field(default_factory=_get_default_max_context_tokens)
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None

    def get_extra_params(self) -> dict[str, Any]:
        """Extract additional parameters to pass through to upstream."""
        standard_keys = {
            "session_id",
            "messages",
            "max_context_tokens",
            "model",
            "temperature",
            "max_tokens",
        }
        if self.__pydantic_extra__:
            return {
                k: v for k, v in self.__pydantic_extra__.items()
                if k not in standard_keys
            }
        return {}


class ChatResponse(BaseModel):
    """Chat completion response model with context metadata and upstream passthrough."""
    model_config = ConfigDict(extra="allow")

    session_id: str
    messages: list[dict]
    context_info: dict
    id: Optional[str] = None
    object: Optional[str] = None
    created: Optional[int] = None
    model: Optional[str] = None
    choices: Optional[list[dict]] = None
    usage: Optional[dict] = None
    raw_response: Optional[dict] = None


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: str


app = FastAPI(
    title="Open Context Circulator",
    description="Context virtualization relay for AI agents",
    version="0.2.0"
)


# Global state (in production, use dependency injection)
_store: Optional[MessageStore] = None
_engine: Optional[ContextEngine] = None
_relay: Optional[UpstreamRelay] = None


def get_store() -> MessageStore:
    """Get or create message store."""
    global _store
    if _store is None:
        db_path = os.environ.get("OCC_DB_PATH", "occ.db")
        _store = MessageStore(db_path)
    return _store


def set_store(store: Optional[MessageStore]) -> None:
    """Override message store (primarily for tests)."""
    global _store
    _store = store


def get_engine(max_tokens: Optional[int] = None) -> ContextEngine:
    """Get or create context engine."""
    global _engine
    if _engine is None:
        effective_max = max_tokens if max_tokens is not None else _get_default_max_context_tokens()
        _engine = ContextEngine(max_tokens=effective_max)
    return _engine


def set_engine(engine: Optional[ContextEngine]) -> None:
    """Override context engine (primarily for tests)."""
    global _engine
    _engine = engine


def get_relay() -> UpstreamRelay:
    """Get or create upstream relay."""
    global _relay
    if _relay is None:
        _relay = UpstreamRelay()
    return _relay


def set_relay(relay: Optional[Any]) -> None:
    """Override upstream relay (primarily for tests)."""
    global _relay
    _relay = relay


@app.exception_handler(RelayError)
async def relay_error_handler(request: Request, exc: RelayError) -> JSONResponse:
    """Graceful error handler for upstream relay failures adhering to API contract."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.code,
                "message": exc.message,
                "details": exc.details,
            }
        },
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc).isoformat()
    )


@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(request: ChatRequest):
    """Process chat completion with context virtualization and upstream forwarding.
    
    1. Retrieves or creates active session.
    2. Records incoming message turns in SQLite MessageStore.
    3. Calculates virtualized context window within token bounds.
    4. Forwards virtualized messages to upstream LLM endpoint.
    5. Records received assistant response into SQLite MessageStore.
    6. Returns virtualized context metadata and assistant response / passthrough.
    """
    store = get_store()
    engine = get_engine(request.max_context_tokens)
    relay = get_relay()

    # Generate session ID if not provided
    session_id = request.session_id or str(uuid.uuid4())

    # Get or create session
    session = store.get_session(session_id)
    if session is None:
        session = store.create_session(session_id)

    # Store incoming messages as turns
    incoming_tokens = 0
    for msg in request.messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        token_count = engine.estimate_message_tokens(msg)
        incoming_tokens += token_count
        store.add_turn(session_id, role, content, token_count)

    # Get all turns for this session
    turns = store.get_turns(session_id)

    # Calculate context window
    window = engine.calculate_window(turns, request.max_context_tokens)

    # Forward virtualized context to upstream LLM
    upstream_response = await relay.forward_chat_completion(
        messages=window.messages,
        model=request.model,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        extra_params=request.get_extra_params(),
    )

    # Extract assistant message from upstream response
    assistant_msg = extract_assistant_message(upstream_response)

    # Store assistant response into SQLite MessageStore
    response_tokens = engine.estimate_message_tokens(assistant_msg)
    store.add_turn(
        session_id=session_id,
        role=assistant_msg.get("role", "assistant"),
        content=assistant_msg.get("content", ""),
        token_count=response_tokens,
    )

    # Update session activity with total accumulated tokens
    all_turns = store.get_turns(session_id)
    total_session_tokens = sum(t["token_count"] for t in all_turns)
    store.update_session_activity(session_id, total_session_tokens)

    # Format choices for direct passthrough
    choices = upstream_response.get("choices")
    if not choices:
        choices = [{"index": 0, "message": assistant_msg, "finish_reason": "stop"}]

    return ChatResponse(
        session_id=session_id,
        messages=[assistant_msg],
        context_info=engine.get_context_info(window),
        id=upstream_response.get("id"),
        object=upstream_response.get("object"),
        created=upstream_response.get("created"),
        model=upstream_response.get("model") or request.model,
        choices=choices,
        usage=upstream_response.get("usage"),
        raw_response=upstream_response,
    )


# For running directly with uvicorn
if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("OCC_HOST", "0.0.0.0")
    port = int(os.environ.get("OCC_PORT") or os.environ.get("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
