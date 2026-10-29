"""Upstream LLM proxy forwarding and relay client."""

import os
from typing import Any, Optional
import httpx
from pydantic import BaseModel, Field


class RelayConfig(BaseModel):
    """Configuration for upstream LLM relay."""
    upstream_url: str = Field(default="http://127.0.0.1:8045/v1")
    api_key: Optional[str] = None
    timeout: float = 30.0
    default_model: str = "default"

    @classmethod
    def from_env(cls) -> "RelayConfig":
        """Load configuration from environment variables."""
        return cls(
            upstream_url=os.environ.get("OCC_UPSTREAM_URL") or os.environ.get("UPSTREAM_URL", "http://127.0.0.1:8045/v1"),
            api_key=os.environ.get("OCC_UPSTREAM_API_KEY") or os.environ.get("UPSTREAM_API_KEY"),
            timeout=float(os.environ.get("OCC_UPSTREAM_TIMEOUT") or os.environ.get("UPSTREAM_TIMEOUT", "30.0")),
            default_model=os.environ.get("OCC_UPSTREAM_MODEL") or os.environ.get("UPSTREAM_MODEL", "default"),
        )


class RelayError(Exception):
    """Base exception for upstream relay errors."""

    def __init__(
        self,
        message: str,
        status_code: int = 502,
        code: str = "upstream_error",
        details: Optional[dict] = None,
    ):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code
        self.details = details or {}


class UpstreamUnavailableError(RelayError):
    """Raised when upstream LLM service is unreachable or connection fails."""

    def __init__(
        self,
        message: str = "Upstream LLM service is unavailable",
        details: Optional[dict] = None,
    ):
        super().__init__(
            message=message,
            status_code=502,
            code="upstream_unavailable",
            details=details,
        )


class UpstreamTimeoutError(RelayError):
    """Raised when request to upstream LLM service times out."""

    def __init__(
        self,
        message: str = "Upstream LLM service request timed out",
        details: Optional[dict] = None,
    ):
        super().__init__(
            message=message,
            status_code=504,
            code="upstream_timeout",
            details=details,
        )


class UpstreamResponseError(RelayError):
    """Raised when upstream returns an HTTP error status (4xx/5xx)."""

    def __init__(
        self,
        message: str,
        status_code: int = 502,
        details: Optional[dict] = None,
    ):
        super().__init__(
            message=message,
            status_code=status_code,
            code="upstream_response_error",
            details=details,
        )


def extract_assistant_message(response_data: dict) -> dict:
    """Extract assistant message dict from upstream response payload.
    
    Supports OpenAI standard format (choices[0].message), legacy text format,
    OCC format (messages[0]), or direct dictionary payload.
    """
    if not isinstance(response_data, dict):
        return {"role": "assistant", "content": str(response_data)}

    # Standard OpenAI format: choices[0].message
    choices = response_data.get("choices")
    if choices and isinstance(choices, list) and len(choices) > 0:
        first_choice = choices[0]
        if isinstance(first_choice, dict):
            if "message" in first_choice and isinstance(first_choice["message"], dict):
                msg = first_choice["message"]
                return {
                    "role": msg.get("role", "assistant"),
                    "content": msg.get("content", "") or "",
                }
            if "text" in first_choice:
                return {
                    "role": "assistant",
                    "content": first_choice.get("text", "") or "",
                }

    # OCC format: messages[0]
    messages = response_data.get("messages")
    if messages and isinstance(messages, list) and len(messages) > 0:
        first_msg = messages[0]
        if isinstance(first_msg, dict):
            return {
                "role": first_msg.get("role", "assistant"),
                "content": first_msg.get("content", "") or "",
            }

    # Direct message field
    if "message" in response_data and isinstance(response_data["message"], dict):
        msg = response_data["message"]
        return {
            "role": msg.get("role", "assistant"),
            "content": msg.get("content", "") or "",
        }

    # Direct content field
    if "content" in response_data:
        return {"role": "assistant", "content": str(response_data["content"])}

    return {"role": "assistant", "content": str(response_data)}


class UpstreamRelay:
    """Relay client for forwarding virtualized context to upstream OpenAI-compatible LLM."""

    def __init__(
        self,
        config: Optional[RelayConfig] = None,
        client: Optional[httpx.AsyncClient] = None,
    ):
        """Initialize upstream relay.
        
        Args:
            config: Optional relay configuration. If not provided, loads from env.
            client: Optional httpx.AsyncClient instance (useful for mocking/testing).
        """
        self.config = config or RelayConfig.from_env()
        self._client = client

    def get_target_url(self) -> str:
        """Resolve full target URL for chat completions endpoint."""
        base = self.config.upstream_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        return f"{base}/chat/completions"

    def get_headers(self) -> dict[str, str]:
        """Build HTTP headers including optional Bearer auth."""
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    async def forward_chat_completion(
        self,
        messages: list[dict],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        extra_params: Optional[dict] = None,
    ) -> dict[str, Any]:
        """Forward virtualized messages to upstream LLM endpoint.
        
        Args:
            messages: List of message dicts (virtualized context).
            model: Optional model identifier.
            temperature: Optional sampling temperature.
            max_tokens: Optional max tokens to generate.
            extra_params: Additional OpenAI-compatible request parameters.
            
        Returns:
            Raw response dict from upstream LLM.
            
        Raises:
            UpstreamUnavailableError: If connection to upstream fails.
            UpstreamTimeoutError: If upstream times out.
            UpstreamResponseError: If upstream returns 4xx/5xx status.
        """
        target_url = self.get_target_url()
        headers = self.get_headers()

        payload: dict[str, Any] = {
            "messages": messages,
            "model": model or self.config.default_model,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if extra_params:
            payload.update(extra_params)

        try:
            if self._client is not None:
                response = await self._client.post(
                    target_url,
                    json=payload,
                    headers=headers,
                    timeout=self.config.timeout,
                )
            else:
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        target_url,
                        json=payload,
                        headers=headers,
                        timeout=self.config.timeout,
                    )

            if response.status_code >= 400:
                mapped_status = 502 if response.status_code >= 500 else response.status_code
                raise UpstreamResponseError(
                    f"Upstream returned HTTP {response.status_code}: {response.text}",
                    status_code=mapped_status,
                    details={
                        "upstream_status": response.status_code,
                        "upstream_body": response.text,
                    },
                )

            return response.json()

        except httpx.ConnectError as e:
            raise UpstreamUnavailableError(
                f"Cannot connect to upstream LLM at {target_url}: {e}",
                details={"target_url": target_url, "error": str(e)},
            ) from e
        except httpx.ConnectTimeout as e:
            raise UpstreamTimeoutError(
                f"Connection to upstream LLM at {target_url} timed out: {e}",
                details={"target_url": target_url, "error": str(e)},
            ) from e
        except httpx.TimeoutException as e:
            raise UpstreamTimeoutError(
                f"Upstream LLM at {target_url} timed out after {self.config.timeout}s: {e}",
                details={"target_url": target_url, "error": str(e)},
            ) from e
        except httpx.RequestError as e:
            raise UpstreamUnavailableError(
                f"Error communicating with upstream LLM at {target_url}: {e}",
                details={"target_url": target_url, "error": str(e)},
            ) from e
