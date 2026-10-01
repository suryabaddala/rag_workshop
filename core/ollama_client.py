"""
Ollama Client module for Local RAG Workshop.
Provides lightweight HTTP interaction with local Ollama runtime without external SDKs.
Supports connection health checks, model discovery, and prompt generation.
"""

from dataclasses import dataclass
import json
import logging
from typing import Any, Dict, Generator, List, Optional, Union
import requests

logger = logging.getLogger(__name__)


class OllamaClientError(Exception):
    """Base exception for Ollama client interactions."""
    pass


class OllamaConnectionError(OllamaClientError):
    """Raised when Ollama daemon is unreachable."""
    pass


class OllamaModelNotFoundError(OllamaClientError):
    """Raised when the specified model is not installed locally."""
    pass


@dataclass
class OllamaModelInfo:
    """Represents an installed Ollama model."""
    name: str
    size_bytes: int
    modified_at: str

    @property
    def size_gb(self) -> float:
        return round(self.size_bytes / (1024 ** 3), 2)


class OllamaClient:
    """
    Direct HTTP client for the local Ollama API.
    Zero external heavy dependencies; uses Python requests.
    """

    DEFAULT_BASE_URL = "http://127.0.0.1:11434"

    def __init__(self, base_url: str = DEFAULT_BASE_URL, timeout: int = 90):
        """
        Initialize Ollama client.

        Args:
            base_url: Ollama API endpoint (default: http://127.0.0.1:11434).
            timeout: Maximum response timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def check_connection(self) -> bool:
        """
        Pings Ollama server to verify that the local daemon is running.

        Returns:
            True if reachable, False otherwise.
        """
        try:
            resp = requests.get(f"{self.base_url}/api/version", timeout=3)
            return resp.status_code == 200
        except Exception:
            return False

    def list_installed_models(self) -> List[OllamaModelInfo]:
        """
        Retrieves list of models currently downloaded and available in Ollama.

        Returns:
            List of OllamaModelInfo objects.

        Raises:
            OllamaConnectionError: If Ollama service cannot be reached.
        """
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if resp.status_code != 200:
                raise OllamaClientError(f"Ollama returned HTTP status {resp.status_code}: {resp.text}")

            data = resp.json()
            raw_models = data.get("models", [])
            models: List[OllamaModelInfo] = []

            for m in raw_models:
                name = m.get("name") or m.get("model", "")
                size = m.get("size", 0)
                modified = m.get("modified_at", "")
                models.append(OllamaModelInfo(name=name, size_bytes=size, modified_at=modified))

            return models

        except requests.exceptions.ConnectionError as e:
            raise OllamaConnectionError(
                f"Cannot connect to Ollama at '{self.base_url}'. "
                "Ensure the Ollama application is running (e.g., check system tray or run 'ollama serve')."
            ) from e
        except Exception as e:
            if isinstance(e, (OllamaClientError, OllamaConnectionError)):
                raise
            raise OllamaClientError(f"Error querying Ollama models: {str(e)}") from e

    def generate(
        self,
        prompt: str,
        model: str,
        system: Optional[str] = None,
        temperature: float = 0.2,
        stream: bool = False,
    ) -> Union[str, Generator[str, None, None]]:
        """
        Sends generation request to local Ollama runtime.

        Args:
            prompt: Formatted user query with retrieved document context.
            model: Name of installed local model (e.g. 'qwen2.5:3b' or 'llama3.2:latest').
            system: System instructions enforcing grounded behavior.
            temperature: Sampling temperature (0.0 - 1.0, low = deterministic/grounded).
            stream: If True, yields string token chunks as they arrive.

        Returns:
            Complete response string if stream=False, or a Generator of tokens if stream=True.
        """
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature,
            },
        }
        if system:
            payload["system"] = system

        endpoint = f"{self.base_url}/api/generate"

        if not stream:
            try:
                resp = requests.post(endpoint, json=payload, timeout=self.timeout)
                if resp.status_code == 404:
                    raise OllamaModelNotFoundError(
                        f"Model '{model}' is not installed in Ollama. "
                        f"Run: ollama pull {model}"
                    )
                if resp.status_code != 200:
                    raise OllamaClientError(f"Ollama generation failed (HTTP {resp.status_code}): {resp.text}")

                data = resp.json()
                return data.get("response", "").strip()

            except requests.exceptions.ConnectionError as e:
                raise OllamaConnectionError(
                    f"Lost connection to Ollama at '{self.base_url}' during generation."
                ) from e
            except requests.exceptions.Timeout as e:
                raise OllamaClientError(
                    f"Ollama generation timed out after {self.timeout} seconds. "
                    "Try using a smaller model (e.g., qwen2.5:1.5b) or decreasing Top-K."
                ) from e
        else:
            return self._stream_generator(endpoint, payload)

    def _stream_generator(self, endpoint: str, payload: Dict[str, Any]) -> Generator[str, None, None]:
        """Generator yielding tokens from streamed Ollama HTTP response."""
        try:
            with requests.post(endpoint, json=payload, stream=True, timeout=self.timeout) as resp:
                if resp.status_code == 404:
                    raise OllamaModelNotFoundError(f"Model '{payload.get('model')}' is not installed.")
                if resp.status_code != 200:
                    raise OllamaClientError(f"Ollama stream failed with HTTP {resp.status_code}.")

                for line in resp.iter_lines(decode_unicode=True):
                    if line:
                        chunk_json = json.loads(line)
                        token = chunk_json.get("response", "")
                        yield token
                        if chunk_json.get("done", False):
                            break
        except requests.exceptions.ConnectionError as e:
            raise OllamaConnectionError("Connection to Ollama was interrupted during streaming.") from e
