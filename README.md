# Open Context Circulator (OCC)

> **Cross-platform context virtualization & token bounding relay for AI agents.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Tests Passing](https://img.shields.io/badge/tests-49%20passed-brightgreen.svg)]()

---

## Overview

AI coding agents and autonomous workflows executing multi-step tasks routinely face context window limits, token exhaustion, and runaway API costs.

**Open Context Circulator (OCC)** is an ultra-lightweight, OpenAI-compatible proxy server designed to sit between AI agent runtimes (such as **Hermes Agent**) and upstream LLM providers (such as **Antigravity Manager**, local models, or commercial APIs).

### Key Features
- **Context Virtualization**: Dynamically bounds inference context within a configurable token budget while retaining the canonical conversation history.
- **SQLite Persistence**: Automatically logs every turn, session, and token count to a local, zero-maintenance SQLite database (`occ.db`).
- **Rolling Context Window**: Automatically shifts the context window to prioritize system directives and recent turns, cleanly rolling older turns out of active LLM prompts.
- **Drop-in OpenAI Compatibility**: Exposes standard `/v1/chat/completions` and `/health` endpoints with full passthrough of upstream response metadata, usage stats, and choices.
- **Cross-Platform**: First-class support for both Windows and Linux developer environments.

---

## Architecture Overview

```text
+-------------------+
|  AI Agent Client  |  (e.g., Hermes Agent / CLI)
+---------+---------+
          |
          | HTTP POST /v1/chat/completions (OpenAI format)
          v
+-----------------------------------------------------------+
|               Open Context Circulator (OCC)               |
|                                                           |
|   +-----------------------+     +---------------------+   |
|   |    FastAPI Router     |     |   SQLite Database   |   |
|   |  - /health            | <-> |  (turns, sessions,  |   |
|   |  - /v1/chat/completions     |   token metrics)    |   |
|   +-----------+-----------+     +---------------------+   |
|               |                                           |
|               v                                           |
|   +-----------------------+                               |
|   | Context Window Engine | (Bounds context to budget;    |
|   | & Token Estimator     |  retains system prompts &     |
|   +-----------+-----------+  most recent turns)           |
|               |                                           |
|               v                                           |
|   +-----------------------+                               |
|   |     Upstream Relay    |                               |
|   +-----------+-----------+                               |
+---------------+-------------------------------------------+
                |
                | Forward bounded context payload
                v
+-------------------------------+
|         Upstream LLM          |  (e.g., Antigravity Proxy :8045,
+-------------------------------+   Ollama, vLLM, OpenAI, Claude)
```

### Context Virtualization Mechanics
1. **Incoming Request**: The client sends a chat completion request containing messages and an optional `session_id`.
2. **Session & Turn Storage**: Incoming messages are estimated for token count and persisted into SQLite (`occ.db`).
3. **Window Calculation**: The `ContextEngine` builds an optimal context window respecting `OCC_MAX_CONTEXT_TOKENS`. System messages are preserved, and turns are rolled forward from oldest to newest.
4. **Relay Forwarding**: Only the virtualized context window is dispatched upstream to the LLM backend.
5. **Response Archiving**: The upstream assistant response is captured, persisted to SQLite, and returned to the client alongside detailed virtualization metadata (`context_info`).

---

## Quickstart

### Prerequisites
- Python 3.10 or higher
- Package manager: `uv` (recommended) or `pip`

---

### Windows Quickstart

#### Option A: Using `uv` (Recommended)
```powershell
# Clone the repository
git clone https://github.com/motasimfuad/open-context-circulator.git
cd open-context-circulator

# Create virtual environment and install dependencies
uv venv
.venv\Scripts\activate
uv pip install -e .

# Start the server
python -m open_context_circulator.cli --port 8000
```

#### Option B: Using standard `pip`
```powershell
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -e .

# Start the server
python main.py --port 8000
```

---

### Linux Quickstart

#### Option A: Using `uv` (Recommended)
```bash
# Clone the repository
git clone https://github.com/motasimfuad/open-context-circulator.git
cd open-context-circulator

# Create virtual environment and install
uv venv
source .venv/bin/activate
uv pip install -e .

# Start the server
python -m open_context_circulator.cli --port 8000
```

#### Option B: Using standard `pip`
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# Start the server
python3 main.py --port 8000
```

---

## CLI Runner

OCC provides multiple convenient entry points:

```bash
# Using installed console script
open-context-circulator --help
occ --help

# Using module runner
python -m open_context_circulator.cli --help
python -m open_context_circulator --help

# Using repo root main.py
python main.py --help
```

### CLI Arguments & Options

| Flag | Default | Environment Variable | Description |
|---|---|---|---|
| `--host` | `0.0.0.0` | `OCC_HOST` | Host interface to bind server |
| `--port` | `8000` | `OCC_PORT` | Port number to listen on |
| `--upstream-url` | `http://127.0.0.1:8045/v1` | `OCC_UPSTREAM_URL` | Upstream OpenAI-compatible LLM endpoint |
| `--db-path` | `occ.db` | `OCC_DB_PATH` | Path to SQLite turns database |
| `--max-context-tokens` | `4000` | `OCC_MAX_CONTEXT_TOKENS` | Maximum virtualized token window |
| `--reload` | `False` | — | Enable uvicorn auto-reload for local dev |
| `--log-level` | `info` | — | Log level (`info`, `debug`, `warning`, `error`) |

---

## Configuration Options

Configure OCC via environment variables in `.env` or your shell:

```bash
# Upstream LLM Endpoint (e.g., Antigravity proxy or OpenAI)
export OCC_UPSTREAM_URL="http://127.0.0.1:8045/v1"

# Upstream API Authentication (optional)
export UPSTREAM_API_KEY="sk-..."

# Server Port and Host
export OCC_PORT=8000
export OCC_HOST="0.0.0.0"

# Context Window Boundary
export OCC_MAX_CONTEXT_TOKENS=4000

# SQLite Database Location
export OCC_DB_PATH="occ.db"
```

---

## Integration Guide

### 1. Integrating with Hermes Agent & Antigravity Local Proxy

When operating **Hermes Agent** alongside **Antigravity Manager / Local Proxy**, OCC operates as an intelligent buffer between the agent and Antigravity:

```text
[Hermes Agent] ---> (Port 8000) [ OCC Proxy ] ---> (Port 8045) [ Antigravity Local Proxy ]
```

#### Step 1: Start OCC pointing to Antigravity
```bash
export OCC_UPSTREAM_URL="http://127.0.0.1:8045/v1"
export OCC_PORT=8000
python -m open_context_circulator.cli
```

#### Step 2: Configure Hermes Agent Profile
In your Hermes profile configuration (`~/.hermes/config.yaml` or profile `config.yaml`):

```yaml
model: gemini-3.8-flash-high
provider: custom
custom_providers:
  occ_relay:
    base_url: http://127.0.0.1:8000/v1
    api_key: not-needed
```

Or set the environment variable before invoking Hermes:
```bash
export OPENAI_BASE_URL="http://127.0.0.1:8000/v1"
```

---

### 2. Manual Testing / Verification with `curl`

#### Check Health Status
```bash
curl http://localhost:8000/health
```
```json
{
  "status": "healthy",
  "timestamp": "2026-10-29T07:30:00.000000+00:00"
}
```

#### Send Chat Completion Request
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test-session-001",
    "messages": [
      {"role": "system", "content": "You are a concise engineering assistant."},
      {"role": "user", "content": "Explain context virtualization in one sentence."}
    ],
    "max_context_tokens": 4000
  }'
```

The response includes standard OpenAI `choices` plus OCC's virtualization metadata:
```json
{
  "session_id": "test-session-001",
  "messages": [
    {
      "role": "assistant",
      "content": "Context virtualization decouples conversational memory from model inference limits by dynamically serving a bounded window over persisted turns."
    }
  ],
  "context_info": {
    "total_tokens": 32,
    "window_start": 0,
    "window_end": 2,
    "rolled_turns": 0,
    "utilization": 0.008
  }
}
```

---

## Testing Guide

OCC features comprehensive unit and integration test coverage across all subsystems (storage, context engine, upstream relay, error handlers, and FastAPI server endpoints).

### Run Test Suite
```bash
# Run all tests
pytest

# Run tests with verbose output
pytest -v

# Run a specific test module
pytest tests/test_relay.py
pytest tests/test_engine.py
pytest tests/test_server.py
```

All 49 tests run isolated against in-memory SQLite stores and mock HTTP transports, executing cleanly with zero external network dependencies.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2026 Motasim Fuad.
