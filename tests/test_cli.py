"""Unit tests for Open Context Circulator CLI entrypoint."""

import os
from unittest.mock import patch
import pytest

from open_context_circulator.cli import build_parser, main


def test_build_parser_defaults():
    parser = build_parser()
    args = parser.parse_args([])
    assert args.host == "0.0.0.0"
    assert args.port == 8000
    assert args.upstream_url == "http://127.0.0.1:8045/v1"
    assert args.db_path == "occ.db"
    assert args.max_context_tokens == 4000
    assert args.reload is False
    assert args.log_level == "info"


def test_build_parser_custom_args():
    parser = build_parser()
    custom_argv = [
        "--host", "127.0.0.1",
        "--port", "9000",
        "--upstream-url", "http://localhost:11434/v1",
        "--db-path", "custom.db",
        "--max-context-tokens", "8192",
        "--reload",
        "--log-level", "debug",
    ]
    args = parser.parse_args(custom_argv)
    assert args.host == "127.0.0.1"
    assert args.port == 9000
    assert args.upstream_url == "http://localhost:11434/v1"
    assert args.db_path == "custom.db"
    assert args.max_context_tokens == 8192
    assert args.reload is True
    assert args.log_level == "debug"


@patch("uvicorn.run")
def test_main_invokes_uvicorn(mock_uvicorn_run):
    original_env = os.environ.copy()
    try:
        main(["--port", "8888", "--host", "127.0.0.1"])
        mock_uvicorn_run.assert_called_once_with(
            "open_context_circulator.server:app",
            host="127.0.0.1",
            port=8888,
            reload=False,
            log_level="info",
        )
    finally:
        os.environ.clear()
        os.environ.update(original_env)
