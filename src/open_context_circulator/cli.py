"""Command line interface runner for Open Context Circulator."""

import argparse
import os
import sys
import uvicorn


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for the OCC CLI runner."""
    parser = argparse.ArgumentParser(
        prog="open-context-circulator",
        description="Cross-platform context virtualization & token bounding relay for AI agents.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=os.environ.get("OCC_HOST", "0.0.0.0"),
        help="Host address to bind server (default: 0.0.0.0, env: OCC_HOST)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("OCC_PORT") or os.environ.get("PORT") or 8000),
        help="Port number to listen on (default: 8000, env: OCC_PORT)",
    )
    parser.add_argument(
        "--upstream-url",
        type=str,
        default=os.environ.get("OCC_UPSTREAM_URL") or os.environ.get("UPSTREAM_URL", "http://127.0.0.1:8045/v1"),
        help="Upstream OpenAI-compatible LLM endpoint URL (default: http://127.0.0.1:8045/v1, env: OCC_UPSTREAM_URL)",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=os.environ.get("OCC_DB_PATH", "occ.db"),
        help="Path to SQLite storage database (default: occ.db, env: OCC_DB_PATH)",
    )
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=int(os.environ.get("OCC_MAX_CONTEXT_TOKENS", 4000)),
        help="Default max virtual context window in tokens (default: 4000, env: OCC_MAX_CONTEXT_TOKENS)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        default=False,
        help="Enable uvicorn hot reloading for development",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        help="Logging verbosity level (default: info)",
    )
    return parser


def main(args: list[str] | None = None) -> None:
    """Run Open Context Circulator server."""
    parser = build_parser()
    parsed_args = parser.parse_args(args)

    # Sync parsed flags to environment variables so FastAPI dependencies pick them up
    os.environ["OCC_HOST"] = parsed_args.host
    os.environ["OCC_PORT"] = str(parsed_args.port)
    os.environ["OCC_UPSTREAM_URL"] = parsed_args.upstream_url
    os.environ["OCC_DB_PATH"] = parsed_args.db_path
    os.environ["OCC_MAX_CONTEXT_TOKENS"] = str(parsed_args.max_context_tokens)

    uvicorn.run(
        "open_context_circulator.server:app",
        host=parsed_args.host,
        port=parsed_args.port,
        reload=parsed_args.reload,
        log_level=parsed_args.log_level,
    )


if __name__ == "__main__":
    main()
