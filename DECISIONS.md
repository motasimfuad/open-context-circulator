# Decisions

## Requirements
- Open-source context virtualization relay for Windows/Hermes agents
- Build incrementally in slices
- Target: Claude's open-source prize submission
- Profile building as open-source contributor

## Stack Options
Evaluated:
1. **Python (FastAPI + SQLite + pytest)**: Popular, well-documented, excellent AI agent support, cross-platform, low hosting cost, simple deployment
2. **Node.js (Express + SQLite + Jest)**: Popular, good documentation, cross-platform, slightly more complex async patterns
3. **Hybrid (Python backend + Node.js frontend)**: More complex, harder to change later, overkill for MVP

## Chosen Stack
**Python (FastAPI + SQLite + pytest)** - Approved by Fuad
- **Why**: Best documented for AI agents, simple SQLite storage (no external DB), cross-platform (Windows/Linux), minimal dependencies, fast iteration
- **Hard to change**: SQLite schema design, async patterns in FastAPI
- **Hosting**: Can run anywhere Python runs, no external services required

## Decisions Log
- 2026-10-29: Project initialized with PROJECT-KIT scaffold
- 2026-10-29: Stack chosen - Python (FastAPI + SQLite + pytest) for Slice 1 walking skeleton

## Decisions Log
- 2026-10-29: Project initialized with PROJECT-KIT scaffold
