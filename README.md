# autogen-openaiext-client

This Autogen client is to help interface *quickly* with non-OpenAI LLMs through the OpenAI API.

See [here](https://microsoft.github.io/autogen/stable/reference/python/autogen_ext.models.openai.html#autogen_ext.models.openai.OpenAIChatCompletionClient) for more information on using with custom LLMs.

> This repository simply include clients you can use to initialize your LLMs easily - since the Autogen >v0.4 supports the non-OpenAI LLMs within the `autogen_ext` package itself with a really nice and clean changes from [jackgerrits](https://github.com/jackgerrits) [here](https://github.com/microsoft/autogen/pull/4856).


=======
# Install

`pip install autogen-openaiext-client`

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

# Contributing

1. Adding a new model to existing external providers
   1. For example, adding a new model to `GeminiChatCompletionClient` includes modifying the `GeminiInfo` class in `info.py` and adding the new model to `_MODEL_CAPABILITIES` and `_MODEL_TOKEN_LIMITS` dictionaries.
2. Adding a new external provider
   1. Add a new client class in `client.py`, relevant `ProviderInfo` class in `info.py` and add it to `__init__.py` for easy import.

# Disclaimer

This is a community project for Autogen. Feel free to contribute via issues and PRs and I will try my best to get to it every 3 days.

