from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from agrivibe.config import Settings


def create_chat_model(settings: Settings) -> BaseChatModel:
    """Creates the chat model named by `CHAT_MODEL`.

    Args:
        settings: Application settings with the model name and OpenAI key.

    Returns:
        The chat model the agents use.
    """
    return init_chat_model(
        settings.chat_model,
        api_key=settings.openai_api_key.get_secret_value(),
        use_responses_api=True,
    )
