"""Centralized chat model factory supporting multi-provider LLMs."""

import os
from functools import lru_cache

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel

from src.config import settings


@lru_cache(maxsize=16)
def get_chat_model(model: str, temperature: float = 0.0) -> BaseChatModel:
    """Instantiate and cache a provider-agnostic LangChain ChatModel.

    Supports prefixes such as 'google_genai:gemini-3.5-flash-lite',
    'anthropic:claude-3-5-haiku-latest', or 'groq:llama-3.3-70b-versatile'.
    """
    os.environ.setdefault("GOOGLE_API_KEY", settings.GEMINI_API_KEY.get_secret_value())
    return init_chat_model(model=model, temperature=temperature)
