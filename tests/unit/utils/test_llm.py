from unittest.mock import MagicMock, patch

from src.utils.llm import get_chat_model


def setup_function() -> None:
    """Clear lru_cache before each test to ensure isolation."""
    get_chat_model.cache_clear()


@patch("src.utils.llm.init_chat_model")
def test_get_chat_model_instantiates_and_caches_by_params(
    mock_init_chat_model: MagicMock,
) -> None:
    fake_llm = MagicMock()
    mock_init_chat_model.return_value = fake_llm

    first = get_chat_model("google_genai:gemini-3.5-flash-lite", temperature=0.0)
    second = get_chat_model("google_genai:gemini-3.5-flash-lite", temperature=0.0)

    assert first is fake_llm
    assert second is fake_llm
    mock_init_chat_model.assert_called_once_with(
        model="google_genai:gemini-3.5-flash-lite",
        temperature=0.0,
    )


@patch("src.utils.llm.init_chat_model")
def test_get_chat_model_separates_cache_for_different_models_or_temperatures(
    mock_init_chat_model: MagicMock,
) -> None:
    llm_cold = MagicMock()
    llm_warm = MagicMock()
    llm_other_provider = MagicMock()
    mock_init_chat_model.side_effect = [llm_cold, llm_warm, llm_other_provider]

    res_1 = get_chat_model("google_genai:gemini-3.5-flash-lite", temperature=0.0)
    res_2 = get_chat_model("google_genai:gemini-3.5-flash-lite", temperature=0.2)
    res_3 = get_chat_model("anthropic:claude-3-5-haiku-latest", temperature=0.0)

    assert res_1 is llm_cold
    assert res_2 is llm_warm
    assert res_3 is llm_other_provider
    assert mock_init_chat_model.call_count == 3
