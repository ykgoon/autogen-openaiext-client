from autogen_openaiext_client import OpenAIExtChatCompletionClient
from autogen_openaiext_client.info import GeminiInfo
from dotenv import load_dotenv
import os
import asyncio
from autogen_core.components.models import UserMessage


def test_gemini():
    load_dotenv()
    assert "GEMINI_API_KEY" in os.environ.keys()

    client = OpenAIExtChatCompletionClient(
        model="gemini-1.5-flash",
        api_key=os.environ["GEMINI_API_KEY"],
        model_info=GeminiInfo,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    result = asyncio.run(
        client.create(
            [UserMessage(content="What is the capital of France?", source="user")]
        )
    )
    assert str(result.content).strip() == "Paris", result.content
