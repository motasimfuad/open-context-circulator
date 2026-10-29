"""SQLite storage for canonical turns and session state."""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class MessageStore:
    """SQLite-backed storage for conversation turns and sessions."""

    def __init__(self, db_path: str):
        """Initialize the message store.
        
        Args:
            db_path: Path to SQLite database file.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with row factory."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            
            # Sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    created_at TIMESTAMP NOT NULL,
                    last_activity TIMESTAMP NOT NULL,
                    total_tokens INTEGER DEFAULT 0
                )
            """)
            
            # Turns table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    token_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                )
            """)
            
            # Index for fast session lookups
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_turns_session_id 
                ON turns(session_id)
            """)
            
            conn.commit()
        finally:
            conn.close()

    def create_session(self, session_id: str) -> dict:
        """Create a new session.
        
        Args:
            session_id: Unique session identifier.
            
        Returns:
            Session dict with id, created_at, last_activity, total_tokens.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            now = datetime.now(timezone.utc).isoformat()
            
            cursor.execute("""
                INSERT INTO sessions (id, created_at, last_activity, total_tokens)
                VALUES (?, ?, ?, 0)
            """, (session_id, now, now))
            
            conn.commit()
            
            return {
                "id": session_id,
                "created_at": now,
                "last_activity": now,
                "total_tokens": 0
            }
        finally:
            conn.close()

    def get_session(self, session_id: str) -> Optional[dict]:
        """Get session by ID.
        
        Args:
            session_id: Session identifier.
            
        Returns:
            Session dict or None if not found.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
            row = cursor.fetchone()
            
            if row is None:
                return None
            
            return dict(row)
        finally:
            conn.close()

    def update_session_activity(self, session_id: str, total_tokens: int) -> None:
        """Update session's last activity timestamp and token count.
        
        Args:
            session_id: Session identifier.
            total_tokens: New total token count.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            now = datetime.now(timezone.utc).isoformat()
            
            cursor.execute("""
                UPDATE sessions 
                SET last_activity = ?, total_tokens = ?
                WHERE id = ?
            """, (now, total_tokens, session_id))
            
            conn.commit()
        finally:
            conn.close()

    def add_turn(self, session_id: str, role: str, content: str, token_count: int) -> int:
        """Add a conversation turn.
        
        Args:
            session_id: Session identifier.
            role: Message role (user, assistant, system).
            content: Message content.
            token_count: Estimated token count.
            
        Returns:
            Turn ID.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            now = datetime.now(timezone.utc).isoformat()
            
            cursor.execute("""
                INSERT INTO turns (session_id, role, content, token_count, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (session_id, role, content, token_count, now))
            
            conn.commit()
            return cursor.lastrowid
        finally:
            conn.close()

    def get_turns(self, session_id: str, limit: Optional[int] = None) -> list[dict]:
        """Get turns for a session.
        
        Args:
            session_id: Session identifier.
            limit: Optional limit on number of turns.
            
        Returns:
            List of turn dicts ordered by creation time.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            
            if limit:
                cursor.execute("""
                    SELECT * FROM turns 
                    WHERE session_id = ?
                    ORDER BY created_at ASC
                    LIMIT ?
                """, (session_id, limit))
            else:
                cursor.execute("""
                    SELECT * FROM turns 
                    WHERE session_id = ?
                    ORDER BY created_at ASC
                """, (session_id,))
            
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def get_turns_in_window(
        self, 
        session_id: str, 
        start_idx: int, 
        end_idx: int
    ) -> list[dict]:
        """Get turns within a window (for rolling context).
        
        Args:
            session_id: Session identifier.
            start_idx: Start index (inclusive).
            end_idx: End index (exclusive).
            
        Returns:
            List of turn dicts in the window.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM turns 
                WHERE session_id = ?
                ORDER BY created_at ASC
                LIMIT ? OFFSET ?
            """, (session_id, end_idx - start_idx, start_idx))
            
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def count_turns(self, session_id: str) -> int:
        """Count total turns in a session.
        
        Args:
            session_id: Session identifier.
            
        Returns:
            Number of turns.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) FROM turns WHERE session_id = ?
            """, (session_id,))
            result = cursor.fetchone()[0]
            return result if result is not None else 0
        finally:
            conn.close()

    def delete_turns_before(self, session_id: str, keep_from_idx: int) -> int:
        """Delete old turns to maintain rolling window.
        
        Args:
            session_id: Session identifier.
            keep_from_idx: Keep turns from this index onward.
            
        Returns:
            Number of turns deleted.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            
            # Get IDs of turns to delete
            cursor.execute("""
                SELECT id FROM turns 
                WHERE session_id = ?
                ORDER BY created_at ASC
                LIMIT ?
            """, (session_id, keep_from_idx))
            
            ids_to_delete = [row[0] for row in cursor.fetchall()]
            
            if not ids_to_delete:
                return 0
            
            # Delete them
            placeholders = ",".join("?" * len(ids_to_delete))
            cursor.execute(f"""
                DELETE FROM turns WHERE id IN ({placeholders})
            """, ids_to_delete)
            
            conn.commit()
            return len(ids_to_delete)
        finally:
            conn.close()