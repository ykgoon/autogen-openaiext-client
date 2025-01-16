from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_openaiext_client.info import ModelInfo, GeminiInfo, GroqInfo, TogetherAIInfo
from autogen_core.models import ModelInfo

from typing import Optional


class GeminiChatCompletionClient(OpenAIChatCompletionClient):

    def __init__(self, **kwargs):
        assert "model" in kwargs, "model is required"
        model = kwargs["model"]
    
        if not "model_capabilities" in kwargs:
            model_capabilities = GeminiInfo.get_capabilities(model)
            kwargs["model_capabilities"] = model_capabilities

        if not "model_info" in kwargs:
            model_info = GeminiInfo.get_info(model)
            kwargs["model_info"] = ModelInfo(**model_info)

        model_info: Optional[ModelInfo] = None
        if "model_info" in kwargs:
            model_info = kwargs["model_info"]
        

        super().__init__(**kwargs)


class GroqChatCompletionClient(OpenAIChatCompletionClient):

    def __init__(self, **kwargs):
        assert "model" in kwargs, "model is required"
        model = kwargs["model"]
    
        if not "model_capabilities" in kwargs:
            model_capabilities = GroqInfo.get_capabilities(model)
            kwargs["model_capabilities"] = model_capabilities

        if not "model_info" in kwargs:
            model_info = GroqInfo.get_info(model)
            kwargs["model_info"] = ModelInfo(**model_info)

        model_info: Optional[ModelInfo] = None
        if "model_info" in kwargs:
            model_info = kwargs["model_info"]
        

        super().__init__(**kwargs)


class TogetherAIChatCompletionClient(OpenAIChatCompletionClient):

    def __init__(self, **kwargs):
        assert "model" in kwargs, "model is required"
        model = kwargs["model"]
    
        if not "model_capabilities" in kwargs:
            model_capabilities = TogetherAIInfo.get_capabilities(model)
            kwargs["model_capabilities"] = model_capabilities

        if not "model_info" in kwargs:
            model_info = TogetherAIInfo.get_info(model)
            kwargs["model_info"] = ModelInfo(**model_info)

        model_info: Optional[ModelInfo] = None
        if "model_info" in kwargs:
            model_info = kwargs["model_info"]
        

        super().__init__(**kwargs)
        