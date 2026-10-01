"""
Unit tests for ModelSelector module.
Validates the workshop model resolution policy:
1. Primary: Qwen detection
2. Fallback: Llama detection
3. No models handling
4. Offline Ollama handling
5. Explicit user override
"""

from unittest.mock import MagicMock
import pytest

from core.model_selector import ModelSelector, ModelStatus
from core.ollama_client import OllamaClient, OllamaModelInfo, OllamaConnectionError


class TestModelSelector:
    def test_qwen_primary_selection(self):
        mock_client = MagicMock(spec=OllamaClient)
        mock_client.list_installed_models.return_value = [
            OllamaModelInfo(name="llama3.2:latest", size_bytes=2_000_000_000, modified_at="2026-09-28"),
            OllamaModelInfo(name="qwen2.5:3b", size_bytes=1_900_000_000, modified_at="2026-09-30"),
        ]

        status = ModelSelector.resolve_model(mock_client)
        assert status.status == "active"
        assert status.selected_model == "qwen2.5:3b"
        assert status.family == "Qwen"
        assert status.is_fallback is False
        assert "Primary" in status.badge_text

    def test_llama_fallback_selection(self):
        mock_client = MagicMock(spec=OllamaClient)
        # Only Llama is installed
        mock_client.list_installed_models.return_value = [
            OllamaModelInfo(name="llama3.2:latest", size_bytes=2_000_000_000, modified_at="2026-09-28"),
        ]

        status = ModelSelector.resolve_model(mock_client)
        assert status.status == "active"
        assert status.selected_model == "llama3.2:latest"
        assert status.family == "Llama"
        assert status.is_fallback is True
        assert "Fallback" in status.badge_text

    def test_no_models_installed_scenario(self):
        mock_client = MagicMock(spec=OllamaClient)
        mock_client.list_installed_models.return_value = []

        status = ModelSelector.resolve_model(mock_client)
        assert status.status == "no_models"
        assert status.selected_model is None
        assert "ollama pull qwen2.5:1.5b" in status.setup_instructions

    def test_other_models_do_not_bypass_workshop_policy(self):
        mock_client = MagicMock(spec=OllamaClient)
        mock_client.list_installed_models.return_value = [
            OllamaModelInfo(name="mistral:latest", size_bytes=2_000_000_000, modified_at="2026-09-28"),
        ]

        status = ModelSelector.resolve_model(mock_client)
        assert status.status == "other_only"
        assert status.selected_model is None
        assert status.family is None
        assert "ollama pull qwen2.5:1.5b" in status.setup_instructions

        overridden = ModelSelector.resolve_model(
            mock_client,
            preferred_model_override="mistral:latest",
        )
        assert overridden.status == "other_only"
        assert overridden.selected_model is None

    def test_ollama_daemon_offline_scenario(self):
        mock_client = MagicMock(spec=OllamaClient)
        mock_client.list_installed_models.side_effect = OllamaConnectionError("Connection refused")

        status = ModelSelector.resolve_model(mock_client)
        assert status.status == "offline"
        assert status.selected_model is None
        assert "Ollama Offline" in status.badge_text
        assert "https://ollama.com" in status.setup_instructions

    def test_user_override_selection(self):
        mock_client = MagicMock(spec=OllamaClient)
        mock_client.list_installed_models.return_value = [
            OllamaModelInfo(name="qwen2.5:3b", size_bytes=1_900_000_000, modified_at="2026-09-30"),
            OllamaModelInfo(name="llama3.2:latest", size_bytes=2_000_000_000, modified_at="2026-09-28"),
        ]

        # Explicitly override to llama
        status = ModelSelector.resolve_model(mock_client, preferred_model_override="llama3.2:latest")
        assert status.selected_model == "llama3.2:latest"
        assert status.family == "Llama"
        assert status.is_fallback is True
