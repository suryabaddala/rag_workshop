"""
Model Selector and Fallback module for Local RAG Workshop.
Implements the workshop model resolution policy:
1. Search locally installed models for Qwen family (Primary).
2. If Qwen missing, search for Llama family (Fallback).
3. If neither found, provide clear terminal instructions without crashing.
"""

from dataclasses import dataclass, field
import logging
import re
from typing import List, Optional

from .ollama_client import OllamaClient, OllamaConnectionError

logger = logging.getLogger(__name__)


@dataclass
class ModelStatus:
    """Represents the outcome of the model detection and fallback resolution."""
    status: str  # 'active', 'no_models', 'offline', 'other_only'
    selected_model: Optional[str]
    family: Optional[str]  # 'Qwen', 'Llama', 'Other', None
    is_fallback: bool
    all_installed: List[str] = field(default_factory=list)
    qwen_models: List[str] = field(default_factory=list)
    llama_models: List[str] = field(default_factory=list)
    badge_text: str = ""
    message: str = ""
    setup_instructions: str = ""


class ModelSelector:
    """
    Detects locally installed Ollama models at runtime and enforces
    the Qwen (Primary) -> Llama (Fallback) resolution logic.
    """

    @classmethod
    def resolve_model(
        cls,
        client: Optional[OllamaClient] = None,
        preferred_model_override: Optional[str] = None,
    ) -> ModelStatus:
        """
        Inspects Ollama runtime and resolves the active model based on workshop rules.

        Args:
            client: Optional OllamaClient instance.
            preferred_model_override: Optional specific model name chosen by the user in UI.

        Returns:
            ModelStatus with resolved model, family, fallback indicator, and help text.
        """
        if client is None:
            client = OllamaClient()

        # Step 1: Check if Ollama service is reachable
        try:
            installed_models = client.list_installed_models()
        except OllamaConnectionError:
            return ModelStatus(
                status="offline",
                selected_model=None,
                family=None,
                is_fallback=False,
                badge_text="🔴 Ollama Offline",
                message="Cannot connect to local Ollama service at http://127.0.0.1:11434.",
                setup_instructions=(
                    "1. Install Ollama from https://ollama.com\n"
                    "2. Start the Ollama application or run 'ollama serve' in terminal.\n"
                    "3. Pull the recommended model: ollama pull qwen2.5:1.5b"
                ),
            )
        except Exception as e:
            return ModelStatus(
                status="offline",
                selected_model=None,
                family=None,
                is_fallback=False,
                badge_text="⚠️ Connection Error",
                message=f"Failed to query Ollama: {str(e)}",
                setup_instructions="Check Ollama service status and verify it is running.",
            )

        model_names = [m.name for m in installed_models]

        # Categorize models
        qwen_models = [m for m in model_names if cls._is_qwen(m)]
        llama_models = [m for m in model_names if cls._is_llama(m)]

        # If user explicitly selected an installed model from UI dropdown
        if (
            preferred_model_override
            and preferred_model_override in model_names
            and (cls._is_qwen(preferred_model_override) or cls._is_llama(preferred_model_override))
        ):
            is_q = cls._is_qwen(preferred_model_override)
            is_l = cls._is_llama(preferred_model_override)
            family = "Qwen" if is_q else ("Llama" if is_l else "Other")
            is_fb = not is_q
            badge = f"⚡ {preferred_model_override} (Qwen)" if is_q else (
                f"🔄 {preferred_model_override} (Llama Fallback)" if is_l else f"📦 {preferred_model_override}"
            )
            return ModelStatus(
                status="active",
                selected_model=preferred_model_override,
                family=family,
                is_fallback=is_fb,
                all_installed=model_names,
                qwen_models=qwen_models,
                llama_models=llama_models,
                badge_text=badge,
                message=f"Using selected model '{preferred_model_override}'.",
            )

        # Step 2: Priority 1 - Detect Qwen model
        if qwen_models:
            chosen = cls._pick_best_variant(qwen_models)
            return ModelStatus(
                status="active",
                selected_model=chosen,
                family="Qwen",
                is_fallback=False,
                all_installed=model_names,
                qwen_models=qwen_models,
                llama_models=llama_models,
                badge_text=f"⚡ {chosen} (Primary: Qwen)",
                message=f"Primary model '{chosen}' detected and active.",
            )

        # Step 3: Priority 2 - Detect Llama model fallback
        if llama_models:
            chosen = cls._pick_best_variant(llama_models)
            return ModelStatus(
                status="active",
                selected_model=chosen,
                family="Llama",
                is_fallback=True,
                all_installed=model_names,
                qwen_models=qwen_models,
                llama_models=llama_models,
                badge_text=f"🔄 {chosen} (Fallback: Llama)",
                message=f"Qwen not found. Automatically fell back to Llama model '{chosen}'.",
            )

        # Step 4: Other models installed, but neither supported family is available.
        if model_names:
            chosen = model_names[0]
            return ModelStatus(
                status="other_only",
                selected_model=None,
                family=None,
                is_fallback=False,
                all_installed=model_names,
                qwen_models=[],
                llama_models=[],
                badge_text="⚠️ No Supported Model",
                message=(
                    f"Neither Qwen nor Llama was found. Ollama has other model(s), including '{chosen}', "
                    "but this workshop only uses Qwen or Llama."
                ),
                setup_instructions=(
                    "Install a supported workshop model in your terminal:\n\n"
                    "  Primary (Recommended): ollama pull qwen2.5:1.5b\n"
                    "  Fallback: ollama pull llama3.2:1b"
                ),
            )

        # Step 5: No models installed at all
        return ModelStatus(
            status="no_models",
            selected_model=None,
            family=None,
            is_fallback=False,
            all_installed=[],
            qwen_models=[],
            llama_models=[],
            badge_text="⚠️ No Models Installed",
            message="Ollama is running, but no models are installed.",
            setup_instructions=(
                "Open your terminal and install the recommended workshop model:\n\n"
                "  Primary (Recommended):\n"
                "    ollama pull qwen2.5:1.5b\n\n"
                "  Fallback Option:\n"
                "    ollama pull llama3.2:1b"
            ),
        )

    @staticmethod
    def _is_qwen(model_name: str) -> bool:
        """Checks if a model name belongs to the Qwen family."""
        lower = model_name.lower()
        return bool(re.search(r"\bqwen", lower) or "qwen" in lower)

    @staticmethod
    def _is_llama(model_name: str) -> bool:
        """Checks if a model name belongs to the Llama family."""
        lower = model_name.lower()
        return bool(re.search(r"\bllama", lower) or "llama" in lower)

    @classmethod
    def _pick_best_variant(cls, models: List[str]) -> str:
        """
        Heuristic to choose the most appropriate lightweight model when multiple exist.
        Prefers smaller laptop-friendly sizes (1.5b, 3b, 1b, 2b) or 'latest'.
        """
        if not models:
            return ""

        # Priority keyword search in order of laptop friendliness
        priority_patterns = [
            r":1\.5b",
            r":3b",
            r":1b",
            r":2b",
            r":latest",
            r":7b",
            r":8b",
        ]

        for pat in priority_patterns:
            for m in models:
                if re.search(pat, m.lower()):
                    return m

        # Default to first available
        return models[0]
