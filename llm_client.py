from __future__ import annotations

from typing import Any

from google import genai
from google.genai.errors import APIError
from config import DEFAULT_MODEL, GEMINI_API_KEY


class LLMClientError(Exception):
    pass


class GeminiClient:
    """Google Gemini LLM client wrapper."""

    def __init__(self, model: str | None = None) -> None:
        if not GEMINI_API_KEY:
            raise LLMClientError("Gemini API key is not configured. Set GEMINI_API_KEY in environment variables.")
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.default_model = model or DEFAULT_MODEL

    def generate_response(
        self,
        prompt_text: str,
        model: str,
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> dict[str, Any]:
        try:
            chat = self.client.chats.create(model=model)
            response = chat.send_message(
                prompt_text,
                config=genai.types.GenerateContentConfig(
                    maxOutputTokens=max_tokens,
                    temperature=temperature,
                ),
            )
            output = response.text.strip() if response.text else ""
            token_count = response.usage_metadata.output_tokens if hasattr(response, "usage_metadata") else 0
            return {
                "output": output,
                "usage": {"total_tokens": token_count},
            }
        except APIError as exc:
            raise LLMClientError(f"Gemini API error: {exc}") from exc
        except Exception as exc:
            raise LLMClientError(f"Unexpected LLM client error: {exc}") from exc
