# autogen-openaiext-client

This Autogen client is to help interface with non-OpenAI LLMs through the OpenAI API.

# Usage

```python
from autogen_openaiext_client import GeminiChatCompletionClient
import asyncio

# Initialize the client
client = GeminiChatCompletionClient(model="gemini-1.5-flash", api_key=os.environ["GEMINI_API_KEY"])

# use the client like any other autogen client. For example:
result = asyncio.run(
    client.create(
        [UserMessage(content="What is the capital of France?", source="user")]
    )
)
print(result.content)
# Paris
```

Currently, `Gemini`, `TogetherAI` and `Groq` clients are supported through the `GeminiChatCompletionClient`, `TogetherAIChatCompletionClient` and `GroqChatCompletionClient` respectively.

Install [Magentic-One](https://github.com/microsoft/autogen/tree/main/python/packages/autogen-magentic-one) and run `python examples/magentic_coder_example.py` for a sample usage with other autogen-based frameworks.



# Known Issues

1. Tool calling in Gemini through OpenAI API runs into issues.

# Disclaimer

This is a community project for Autogen. Feel free to contribute via issues and PRs and I will try my best to get to it every 3 days.

