# API Contract

## Status: FROZEN (Slice 1)
*Frozen after stack decision - changes require version bump*

## Base URL
`http://localhost:8000` (configurable via PORT env var)

## Endpoints

### GET /health
Health check endpoint.

**Response**: `200 OK`
```json
{"status": "healthy", "timestamp": "2026-10-29T12:00:00Z"}
```

### POST /v1/chat/completions
Chat completion endpoint with context virtualization.

**Request**:
```json
{
  "session_id": "string (optional, generates UUID if missing)",
  "messages": [
    {"role": "user|assistant|system", "content": "string"}
  ],
  "max_context_tokens": 4000 (optional, default 4000)
}
```

**Response**: `200 OK`
```json
{
  "session_id": "string",
  "messages": [
    {"role": "assistant", "content": "string"}
  ],
  "context_info": {
    "total_tokens": 1234,
    "window_start": 0,
    "window_end": 5,
    "rolled_turns": 0
  }
}
```

**Error Response**: `4xx/5xx`
```json
{
  "error": {
    "code": "string",
    "message": "string",
    "details": {}
  }
}
```

## Data Shapes

### Message
- `role`: "user" | "assistant" | "system"
- `content`: string (required)

### Turn (internal, stored in SQLite)
- `id`: integer (auto-increment)
- `session_id`: string (indexed)
- `role`: string
- `content`: string
- `token_count`: integer
- `created_at`: timestamp

### Session (internal, stored in SQLite)
- `id`: string (primary key)
- `created_at`: timestamp
- `last_activity`: timestamp
- `total_tokens`: integer

## Error Format
Standard HTTP status codes:
- `400 Bad Request`: Invalid input format
- `404 Not Found`: Session not found (if querying specific session)
- `500 Internal Server Error`: Server error

## Authentication
None for Slice 1 (local-only). To be added in later slices.

## Rate Limits
None for Slice 1 (local-only). To be added in later slices.
