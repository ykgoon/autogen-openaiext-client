import os
import asyncio
from autogen_core.models import UserMessage
from autogen_openaiext_client import GeminiChatCompletionClient
from autogen_openaiext_client.info import GeminiInfo
from dotenv import load_dotenv
from unittest import TestCase


class GeminiTestCase(TestCase):
    def setUp(self):
        load_dotenv()
        assert "GEMINI_API_KEY" in os.environ.keys()

    def test_gemini(self):
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

    def test_gemini_tools(self):
        # Test tool-calling
        # https://github.com/vballoli/autogen-openaiext-client/issues/4

        client = GeminiChatCompletionClient(
            model="gemini-2.0-flash-exp", api_key=os.environ["GEMINI_API_KEY"]
        )

        weather_tool_schema = {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        }
        result = asyncio.run(
            client.create(
                tools=[weather_tool_schema],
                messages=[
                    UserMessage(
                        content="What is the weather like in San Francisco?", source="user"
                    )
                ],
            )
        )
        self.assertIsInstance(result.content, list)
        self.assertEqual(result.content[0].name, "get_current_weather")
