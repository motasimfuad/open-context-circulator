"""FastAPI proxy server for context virtualization."""

import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .engine import ContextEngine
from .store import MessageStore


class ChatRequest(BaseModel):
    """Chat completion request model."""
    session_id: Optional[str] = None
    messages: list[dict] = Field(..., min_length=1)
    max_context_tokens: int = 4000


class ChatResponse(BaseModel):
    """Chat completion response model."""
    session_id: str
    messages: list[dict]
    context_info: dict


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: str


app = FastAPI(
    title="Open Context Circulator",
    description="Context virtualization relay for AI agents",
    version="0.1.0"
)

# Global state (in production, use dependency injection)
_store: Optional[MessageStore] = None
_engine: Optional[ContextEngine] = None


def get_store() -> MessageStore:
    """Get or create message store."""
    global _store
    if _store is None:
        import os
        db_path = os.environ.get("OCC_DB_PATH", "occ.db")
        _store = MessageStore(db_path)
    return _store


def get_engine(max_tokens: int = 4000) -> ContextEngine:
    """Get or create context engine."""
    global _engine
    if _engine is None:
        _engine = ContextEngine(max_tokens=max_tokens)
    return _engine


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc).isoformat()
    )


@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(request: ChatRequest):
    """Process chat completion with context virtualization.
    
    This endpoint:
    1. Creates or retrieves a session
    2. Stores incoming messages
    3. Calculates context window with token bounds
    4. Returns windowed context with metadata
    """
    store = get_store()
    engine = get_engine(request.max_context_tokens)
    
    # Generate session ID if not provided
    session_id = request.session_id or str(uuid.uuid4())
    
    # Get or create session
    session = store.get_session(session_id)
    if session is None:
        session = store.create_session(session_id)
    
    # Store incoming messages as turns
    total_tokens = 0
    for msg in request.messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        token_count = engine.estimate_message_tokens(msg)
        total_tokens += token_count
        
        store.add_turn(session_id, role, content, token_count)
    
    # Update session activity
    store.update_session_activity(session_id, total_tokens)
    
    # Get all turns for this session
    turns = store.get_turns(session_id)
    
    # Calculate context window
    window = engine.calculate_window(turns, request.max_context_tokens)
    
    # For the walking skeleton, we echo back the last user message
    # as a simple "assistant" response
    # In production, this would call the actual LLM
    last_user_msg = None
    for msg in reversed(request.messages):
        if msg.get("role") == "user":
            last_user_msg = msg.get("content", "")
            break
    
    assistant_response = {
        "role": "assistant",
        "content": f"Context received. Window: {window.total_tokens}/{request.max_context_tokens} tokens, {window.rolled_turns} turns rolled."
    }
    
    # Store assistant response
    response_tokens = engine.estimate_message_tokens(assistant_response)
    store.add_turn(session_id, "assistant", assistant_response["content"], response_tokens)
    
    return ChatResponse(
        session_id=session_id,
        messages=[assistant_response],
        context_info=engine.get_context_info(window)
    )


# For running directly with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)