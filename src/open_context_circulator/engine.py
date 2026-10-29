"""Context window manager for token bounds and rolling turns."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ContextWindow:
    """Represents a context window with bounds information."""
    messages: list[dict]
    total_tokens: int
    window_start: int
    window_end: int
    rolled_turns: int
    max_tokens: int


@dataclass
class TokenEstimate:
    """Token estimation result."""
    tokens: int
    method: str


class ContextEngine:
    """Manages context window with token bounding and rolling.
    
    Implements a simple token estimation strategy and rolling window
    to keep context within bounds while preserving recent turns.
    """

    # Simple token estimation: ~4 chars per token (conservative)
    CHARS_PER_TOKEN = 4

    def __init__(self, max_tokens: int = 4000):
        """Initialize the context engine.
        
        Args:
            max_tokens: Maximum tokens to keep in context window.
        """
        self.max_tokens = max_tokens

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text.
        
        Uses a simple character-based estimation.
        For production, use tiktoken or similar.
        
        Args:
            text: Text to estimate.
            
        Returns:
            Estimated token count.
        """
        if not text:
            return 0
        return max(1, len(text) // self.CHARS_PER_TOKEN)

    def estimate_message_tokens(self, message: dict) -> int:
        """Estimate tokens for a message.
        
        Args:
            message: Dict with 'role' and 'content'.
            
        Returns:
            Estimated token count including role overhead.
        """
        # Role overhead (~4 tokens for "user: " or "assistant: ")
        role_tokens = 4
        content_tokens = self.estimate_tokens(message.get("content", ""))
        return role_tokens + content_tokens

    def calculate_window(
        self,
        turns: list[dict],
        max_tokens: Optional[int] = None
    ) -> ContextWindow:
        """Calculate the context window that fits within token bounds.
        
        Keeps the most recent turns that fit within the token limit.
        Older turns are "rolled" (excluded from the window).
        
        Args:
            turns: List of turn dicts with 'role', 'content', 'token_count'.
            max_tokens: Override default max_tokens.
            
        Returns:
            ContextWindow with messages and bounds info.
        """
        max_tokens = max_tokens or self.max_tokens
        
        if not turns:
            return ContextWindow(
                messages=[],
                total_tokens=0,
                window_start=0,
                window_end=0,
                rolled_turns=0,
                max_tokens=max_tokens
            )

        # Convert turns to messages and calculate tokens
        messages = []
        message_tokens = []
        
        for turn in turns:
            msg = {"role": turn["role"], "content": turn["content"]}
            tokens = self.estimate_message_tokens(msg)
            messages.append(msg)
            message_tokens.append(tokens)

        # Work backwards to find the window that fits
        total_tokens = sum(message_tokens)
        rolled_turns = 0
        window_start = 0
        
        # If we're over the limit, roll older turns
        while total_tokens > max_tokens and window_start < len(messages):
            total_tokens -= message_tokens[window_start]
            window_start += 1
            rolled_turns += 1

        window_end = len(messages)
        
        # Extract the window
        window_messages = messages[window_start:window_end]
        
        return ContextWindow(
            messages=window_messages,
            total_tokens=total_tokens,
            window_start=window_start,
            window_end=window_end,
            rolled_turns=rolled_turns,
            max_tokens=max_tokens
        )

    def add_system_message(
        self,
        window: ContextWindow,
        system_content: str
    ) -> ContextWindow:
        """Add a system message to the window.
        
        Args:
            window: Current context window.
            system_content: System message content.
            
        Returns:
            New ContextWindow with system message prepended.
        """
        if not system_content:
            return window

        system_msg = {"role": "system", "content": system_content}
        system_tokens = self.estimate_message_tokens(system_msg)
        
        # Prepend system message
        new_messages = [system_msg] + window.messages
        new_total = window.total_tokens + system_tokens
        
        # If we're still within bounds, return as-is
        if new_total <= self.max_tokens:
            return ContextWindow(
                messages=new_messages,
                total_tokens=new_total,
                window_start=0,
                window_end=len(new_messages),
                rolled_turns=window.rolled_turns,
                max_tokens=self.max_tokens
            )

        # Need to remove more turns to fit system message
        # Keep system message + as many recent turns as possible
        available_tokens = self.max_tokens - system_tokens
        kept_messages = []
        kept_tokens = 0
        
        # Add from most recent backwards
        for msg in reversed(window.messages):
            msg_tokens = self.estimate_message_tokens(msg)
            if kept_tokens + msg_tokens <= available_tokens:
                kept_messages.insert(0, msg)
                kept_tokens += msg_tokens
            else:
                break

        new_rolled = window.rolled_turns + (len(window.messages) - len(kept_messages))
        
        return ContextWindow(
            messages=[system_msg] + kept_messages,
            total_tokens=system_tokens + kept_tokens,
            window_start=0,
            window_end=len(kept_messages) + 1,
            rolled_turns=new_rolled,
            max_tokens=self.max_tokens
        )

    def should_compact(self, window: ContextWindow, threshold: float = 0.8) -> bool:
        """Check if context should be compacted.
        
        Args:
            window: Current context window.
            threshold: Ratio at which to trigger compaction.
            
        Returns:
            True if compaction is recommended.
        """
        if window.max_tokens == 0:
            return False
        return (window.total_tokens / window.max_tokens) >= threshold

    def get_context_info(self, window: ContextWindow) -> dict:
        """Get context info dict for API response.
        
        Args:
            window: Context window.
            
        Returns:
            Dict with context metadata.
        """
        return {
            "total_tokens": window.total_tokens,
            "window_start": window.window_start,
            "window_end": window.window_end,
            "rolled_turns": window.rolled_turns,
            "max_tokens": window.max_tokens,
            "utilization": round(window.total_tokens / window.max_tokens, 2) if window.max_tokens > 0 else 0
        }