"""Tests for context engine and SQLite storage."""

import os
import tempfile
import pytest
from pathlib import Path

from open_context_circulator.store import MessageStore
from open_context_circulator.engine import ContextEngine, ContextWindow


class TestMessageStore:
    """Tests for SQLite message storage."""

    @pytest.fixture
    def store(self):
        """Create a temporary database for each test."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = MessageStore(str(db_path))
            yield store

    def test_create_session(self, store):
        """Test session creation."""
        session = store.create_session("test-session-1")
        
        assert session["id"] == "test-session-1"
        assert "created_at" in session
        assert "last_activity" in session
        assert session["total_tokens"] == 0

    def test_get_session(self, store):
        """Test retrieving a session."""
        # Create session
        store.create_session("test-session-2")
        
        # Retrieve it
        session = store.get_session("test-session-2")
        
        assert session is not None
        assert session["id"] == "test-session-2"
        
        # Non-existent session
        missing = store.get_session("non-existent")
        assert missing is None

    def test_add_turn(self, store):
        """Test adding conversation turns."""
        store.create_session("test-session-3")
        
        turn_id = store.add_turn(
            "test-session-3",
            role="user",
            content="Hello, world!",
            token_count=5
        )
        
        assert turn_id > 0

    def test_get_turns(self, store):
        """Test retrieving turns in order."""
        store.create_session("test-session-4")
        
        # Add turns
        store.add_turn("test-session-4", "user", "First", 3)
        store.add_turn("test-session-4", "assistant", "Second", 4)
        store.add_turn("test-session-4", "user", "Third", 3)
        
        turns = store.get_turns("test-session-4")
        
        assert len(turns) == 3
        assert turns[0]["content"] == "First"
        assert turns[1]["content"] == "Second"
        assert turns[2]["content"] == "Third"

    def test_get_turns_with_limit(self, store):
        """Test retrieving limited turns."""
        store.create_session("test-session-5")
        
        for i in range(10):
            store.add_turn("test-session-5", "user", f"Message {i}", 5)
        
        # Get only first 5
        turns = store.get_turns("test-session-5", limit=5)
        
        assert len(turns) == 5
        assert turns[0]["content"] == "Message 0"
        assert turns[4]["content"] == "Message 4"

    def test_count_turns(self, store):
        """Test counting turns."""
        store.create_session("test-session-6")
        
        assert store.count_turns("test-session-6") == 0
        
        store.add_turn("test-session-6", "user", "One", 3)
        store.add_turn("test-session-6", "assistant", "Two", 4)
        
        assert store.count_turns("test-session-6") == 2

    def test_update_session_activity(self, store):
        """Test updating session activity."""
        store.create_session("test-session-7")
        
        store.update_session_activity("test-session-7", 100)
        
        session = store.get_session("test-session-7")
        assert session["total_tokens"] == 100

    def test_delete_turns_before(self, store):
        """Test deleting old turns for rolling window."""
        store.create_session("test-session-8")
        
        # Add 10 turns
        for i in range(10):
            store.add_turn("test-session-8", "user", f"Message {i}", 5)
        
        # Delete first 5 (keep from index 5)
        deleted = store.delete_turns_before("test-session-8", 5)
        
        assert deleted == 5
        
        # Should have 5 remaining
        remaining = store.get_turns("test-session-8")
        assert len(remaining) == 5
        assert remaining[0]["content"] == "Message 5"


class TestContextEngine:
    """Tests for context window management."""

    def test_estimate_tokens(self):
        """Test token estimation."""
        engine = ContextEngine()
        
        # Empty string
        assert engine.estimate_tokens("") == 0
        
        # Short string
        tokens = engine.estimate_tokens("Hello")
        assert tokens >= 1
        
        # Longer string - approximately len/4
        text = "a" * 100
        tokens = engine.estimate_tokens(text)
        assert 20 <= tokens <= 30  # Around 25 tokens

    def test_calculate_window_empty(self):
        """Test window calculation with no turns."""
        engine = ContextEngine(max_tokens=100)
        window = engine.calculate_window([])
        
        assert window.messages == []
        assert window.total_tokens == 0
        assert window.rolled_turns == 0

    def test_calculate_window_fits(self):
        """Test window when all turns fit."""
        engine = ContextEngine(max_tokens=1000)
        
        turns = [
            {"role": "user", "content": "Hello", "token_count": 3},
            {"role": "assistant", "content": "Hi there", "token_count": 4},
        ]
        
        window = engine.calculate_window(turns)
        
        assert len(window.messages) == 2
        assert window.rolled_turns == 0
        assert window.window_start == 0
        assert window.window_end == 2

    def test_calculate_window_rolls(self):
        """Test window rolls older turns when over limit."""
        engine = ContextEngine(max_tokens=50)
        
        # Create turns that exceed limit
        turns = [
            {"role": "user", "content": "a" * 100, "token_count": 30},  # ~30 tokens
            {"role": "assistant", "content": "b" * 100, "token_count": 30},  # ~30 tokens
            {"role": "user", "content": "c" * 100, "token_count": 30},  # ~30 tokens
        ]
        
        window = engine.calculate_window(turns)
        
        # Should roll older turns to fit
        assert window.rolled_turns > 0
        assert window.total_tokens <= 50
        assert len(window.messages) < 3

    def test_add_system_message(self):
        """Test adding system message to window."""
        engine = ContextEngine(max_tokens=100)
        
        window = ContextWindow(
            messages=[{"role": "user", "content": "Hello"}],
            total_tokens=10,
            window_start=0,
            window_end=1,
            rolled_turns=0,
            max_tokens=100
        )
        
        new_window = engine.add_system_message(window, "You are helpful")
        
        assert len(new_window.messages) == 2
        assert new_window.messages[0]["role"] == "system"

    def test_should_compact(self):
        """Test compaction threshold check."""
        engine = ContextEngine(max_tokens=100)
        
        # Under threshold (80%)
        window_under = ContextWindow(
            messages=[],
            total_tokens=70,
            window_start=0,
            window_end=0,
            rolled_turns=0,
            max_tokens=100
        )
        assert not engine.should_compact(window_under, threshold=0.8)
        
        # Over threshold
        window_over = ContextWindow(
            messages=[],
            total_tokens=85,
            window_start=0,
            window_end=0,
            rolled_turns=0,
            max_tokens=100
        )
        assert engine.should_compact(window_over, threshold=0.8)

    def test_get_context_info(self):
        """Test context info dict generation."""
        engine = ContextEngine(max_tokens=100)
        
        window = ContextWindow(
            messages=[],
            total_tokens=50,
            window_start=0,
            window_end=5,
            rolled_turns=2,
            max_tokens=100
        )
        
        info = engine.get_context_info(window)
        
        assert info["total_tokens"] == 50
        assert info["window_start"] == 0
        assert info["window_end"] == 5
        assert info["rolled_turns"] == 2
        assert info["utilization"] == 0.5


class TestIntegration:
    """Integration tests for store + engine."""

    def test_full_workflow(self):
        """Test complete workflow: store turns, calculate window."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            store = MessageStore(str(db_path))
            engine = ContextEngine(max_tokens=200)
            
            # Create session
            store.create_session("integration-test")
            
            # Add several turns
            for i in range(5):
                store.add_turn(
                    "integration-test",
                    role="user" if i % 2 == 0 else "assistant",
                    content=f"Message {i}",
                    token_count=10
                )
            
            # Get turns and calculate window
            turns = store.get_turns("integration-test")
            window = engine.calculate_window(turns)
            
            assert len(turns) == 5
            assert len(window.messages) > 0
            assert window.total_tokens > 0