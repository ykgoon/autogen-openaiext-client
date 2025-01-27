import os
import asyncio
from autogen_core.models import UserMessage
from autogen_openaiext_client import GeminiChatCompletionClient
from autogen_openaiext_client.info import GeminiInfo
from dotenv import load_dotenv
from unittest import TestCase


class GeminiTestCase(TestCase):
    def test_gemini(self):
        load_dotenv()
        assert "GEMINI_API_KEY" in os.environ.keys()

        client = GeminiChatCompletionClient(
            model="gemini-1.5-flash", api_key=os.environ["GEMINI_API_KEY"]
        )

        result = asyncio.run(
            client.create(
                [UserMessage(content="What is the capital of France?", source="user")]
            )
        )
        self.assertIn("Paris", str(result.content).strip(), result.content)

        client = GeminiChatCompletionClient(
            model="gemini-2.0-flash-exp", api_key=os.environ["GEMINI_API_KEY"]
        )
        result = asyncio.run(
            client.create(
                [UserMessage(content="What is the capital of Japan?", source="user")]
            )
        )
        self.assertIn("Tokyo", str(result.content).strip(), result.content)
