from typing import Dict

from autogen_core.models import ModelInfo


class ExtInfo:
    _MODEL_POINTERS = {}

    _MODEL_CAPABILITIES: Dict[str, ModelInfo] = {}

    _MODEL_TOKEN_LIMITS: Dict[str, int] = {}

    BASE_URL: str = ""

    @staticmethod
    def resolve_model(model: str) -> str:
        if model in GeminiInfo._MODEL_POINTERS:
            return GeminiInfo._MODEL_POINTERS[model]
        return model

    @staticmethod
    def get_capabilities(model: str) -> ModelInfo:
        resolved_model = GeminiInfo.resolve_model(model)
        return GeminiInfo._MODEL_CAPABILITIES[resolved_model]

    @staticmethod
    def get_token_limit(model: str) -> int:
        resolved_model = GeminiInfo.resolve_model(model)
        return GeminiInfo._MODEL_TOKEN_LIMITS[resolved_model]

    def _add_model(
        self, model_name: str, model_capabilities: ModelInfo, token_limit: int
    ):
        self._MODEL_CAPABILITIES[model_name] = model_capabilities
        self._MODEL_TOKEN_LIMITS[model_name] = token_limit

    @classmethod
    def add_model(
        cls, model_name: str, model_capabilities: ModelInfo, token_limit: int
    ):
        cls._add_model(model_name, model_capabilities, token_limit)

    @classmethod
    def get_info(cls, model: str) -> ModelInfo:
        resolved_model = cls.resolve_model(model)
        return cls._MODEL_CAPABILITIES[resolved_model]


class GeminiInfo(ExtInfo):
    _MODEL_POINTERS = {"gemini-1.5-flash": "gemini-1.5-flash"}

    _MODEL_CAPABILITIES: Dict[str, ModelInfo] = {
        "gemini-1.5-flash": {
            "vision": True,
            "function_calling": True,
            "json_output": True,
            "family": "gemini",
        },
        "gemini-1.5-pro": {
            "vision": True,
            "function_calling": True,
            "json_output": True,
            "family": "gemini",
        },
    }

    _MODEL_TOKEN_LIMITS: Dict[str, int] = {
        "gemini-1.5-flash": 1048576,
        "gemini-1.5-pro": 2097152,
    }

    BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta/openai/"


class TogetherAIInfo(ExtInfo):
    _MODEL_POINTERS = {
        "llama-3.1-8b-instruct-turbo": "llama-3.1-8b-instruct-turbo",
        "llama-3.1-70b-instruct-turbo": "llama-3.1-70b-instruct-turbo",
        "llama-3.1-405b-instruct-turbo": "llama-3.1-405b-instruct-turbo",
        "llama-3-8b-instruct-turbo": "llama-3-8b-instruct-turbo",
        "llama-3-70b-instruct-turbo": "llama-3-70b-instruct-turbo",
        "llama-3.2-3b-instruct-turbo": "llama-3.2-3b-instruct-turbo",
        "llama-3-8b-instruct-lite": "llama-3-8b-instruct-lite",
        "llama-3-70b-instruct-lite": "llama-3-70b-instruct-lite",
        "llama-3-8b-instruct-reference": "llama-3-8b-instruct-reference",
        "llama-3-70b-instruct-reference": "llama-3-70b-instruct-reference",
        "llama-3.1-nemotron-70b": "llama-3.1-nemotron-70b",
        "qwen-2.5-coder-32b-instruct": "qwen-2.5-coder-32b-instruct",
        "qwen-32b-preview": "qwen-32b-preview",
        "wizardlm-2-8x22b": "wizardlm-2-8x22b",
        "gemma-2-27b": "gemma-2-27b",
        "gemma-2-9b": "gemma-2-9b",
        "dbrx-instruct": "dbrx-instruct",
        "deepseek-llm-67b-chat": "deepseek-llm-67b-chat",
        "gemma-2b-it": "gemma-2b-it",
        "llama-2-13b-chat-hf": "llama-2-13b-chat-hf",
        "mistral-7b-instruct": "mistral-7b-instruct",
        "mistral-7b-instruct-v0.2": "mistral-7b-instruct-v0.2",
        "mistral-7b-instruct-v0.3": "mistral-7b-instruct-v0.3",
        "mixtral-8x7b-instruct": "mixtral-8x7b-instruct",
        "mixtral-8x22b-instruct": "mixtral-8x22b-instruct",
        "qwen-2.5-7b-instruct-turbo": "qwen-2.5-7b-instruct-turbo",
        "qwen-2.5-72b-instruct-turbo": "qwen-2.5-72b-instruct-turbo",
        "qwen-2-instruct": "qwen-2-instruct",
        "llama-vision-free": "llama-vision-free",
        "llama-3.2-11b-vision-instruct-turbo": "llama-3.2-11b-vision-instruct-turbo",
        "llama-3.2-90b-vision-instruct-turbo": "llama-3.2-90b-vision-instruct-turbo",
        "llama-2-70b-hf": "llama-2-70b-hf",
        "mistral-7b": "mistral-7b",
        "mixtral-8x7b": "mixtral-8x7b",
        "stable-diffusion-xl-base-1.0": "stable-diffusion-xl-base-1.0",
    }

    _MODEL_CAPABILITIES: Dict[str, ModelInfo] = {
        "llama-3.1-8b-instruct-turbo": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "family": "meta-llama3.1",
        },
        "llama-3.1-70b-instruct-turbo": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "family": "meta-llama3.1",
        },
        "llama-3.1-405b-instruct-turbo": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "family": "meta-llama3.1",
        },
        "llama-3-8b-instruct-turbo": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "family": "meta-llama3",
        },
        "llama-3-70b-instruct-turbo": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "family": "meta-llama3",
        },
        "llama-3.2-3b-instruct-turbo": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "family": "meta-llama3.2",
        },
        "llama-3-8b-instruct-lite": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "family": "meta-llama3",
        },
        "llama-3-70b-instruct-lite": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "family": "meta-llama3",
        },
        "llama-3-8b-instruct-reference": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "family": "meta-llama3",
        },
        "llama-3-70b-instruct-reference": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "family": "meta-llama3",
        },
        "llama-3.1-nemotron-70b": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "family": "meta-llama3.1",
        },
        "qwen-2.5-coder-32b-instruct": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "family": "alibaba-qwen",
        },
        "qwen-32b-preview": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "family": "alibaba-qwen",
        },
        "gemma-2-27b": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "family": "google-gemma",
        },
        "gemma-2-9b": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "family": "google-gemma",
        },
        "dbrx-instruct": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "family": "dbrx",
        },
        "deepseek-llm-67b-chat": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "family": "deepseek",
        },
        "gemma-2b-it": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "family": "google-gemma",
        },
        "llama-2-13b-chat-hf": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "family": "meta-llama2",
        },
        "mistral-7b-instruct": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "family": "mistral",
        },
        "mistral-7b-instruct-v0.2": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "family": "mistral",
        },
        "mistral-7b-instruct-v0.3": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "family": "mistral",
        },
        "mixtral-8x7b-instruct": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "family": "mixtral",
        },
        "mixtral-8x22b-instruct": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "family": "mixtral",
        },
        "qwen-2.5-7b-instruct-turbo": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "family": "alibaba-qwen",
        },
        "qwen-2.5-72b-instruct-turbo": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "family": "alibaba-qwen",
        },
        "qwen-2-instruct": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "family": "alibaba-qwen",
        },
        "llama-vision-free": {
            "vision": True,
            "function_calling": False,
            "json_output": False,
            "family": "meta-llama",
        },
        "llama-3.2-11b-vision-instruct-turbo": {
            "vision": True,
            "function_calling": False,
            "json_output": False,
            "family": "meta-llama3.2",
        },
        "llama-3.2-90b-vision-instruct-turbo": {
            "vision": True,
            "function_calling": False,
            "json_output": False,
            "family": "meta-llama3.2",
        },
        "llama-2-70b-hf": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "family": "meta-llama2",
        },
        "mistral-7b": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "family": "mistral",
        },
        "mixtral-8x7b": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "family": "mistral-mixtral",
        },
        "stable-diffusion-xl-base-1.0": {
            "vision": True,
            "function_calling": False,
            "json_output": False,
        },
    }

    _MODEL_TOKEN_LIMITS: Dict[str, int] = {
        "llama-3.1-8b-instruct-turbo": 131072,
        "llama-3.1-70b-instruct-turbo": 131072,
        "llama-3.1-405b-instruct-turbo": 130815,
        "llama-3-8b-instruct-turbo": 8192,
        "llama-3-70b-instruct-turbo": 8192,
        "llama-3.2-3b-instruct-turbo": 131072,
        "llama-3-8b-instruct-lite": 8192,
        "llama-3-70b-instruct-lite": 8192,
        "llama-3-8b-instruct-reference": 8192,
        "llama-3-70b-instruct-reference": 8192,
        "llama-3.1-nemotron-70b": 32768,
        "qwen-2.5-coder-32b-instruct": 32768,
        "qwen-32b-preview": 32768,
        "wizardlm-2-8x22b": 65536,
        "gemma-2-27b": 8192,
        "gemma-2-9b": 8192,
        "dbrx-instruct": 32768,
        "deepseek-llm-67b-chat": 4096,
        "gemma-2b-it": 8192,
        "mythomax-l2-13b": 4096,
        "llama-2-13b-chat-hf": 4096,
        "mistral-7b-instruct": 8192,
        "mistral-7b-instruct-v0.2": 32768,
        "mistral-7b-instruct-v0.3": 32768,
        "mixtral-8x7b-instruct": 32768,
        "mixtral-8x22b-instruct": 65536,
        "nous-hermes-2-mixtral-8x7b-dpo": 32768,
        "qwen-2.5-7b-instruct-turbo": 32768,
        "qwen-2.5-72b-instruct-turbo": 32768,
        "qwen-2-instruct": 32768,
        "upstage-solar-instruct-v1": 4096,
        "llama-vision-free": 131072,
        "llama-3.2-11b-vision-instruct-turbo": 131072,
        "llama-3.2-90b-vision-instruct-turbo": 131072,
        "llama-2-70b-hf": 4096,
        "mistral-7b": 8192,
        "mixtral-8x7b": 32768,
        "flux-1-schnell-free": 0,
        "flux-1-schnell": 4,
        "flux-1-dev": 28,
        "flux-1-canny": 28,
        "flux-1-depth": 28,
        "flux-1-redux": 28,
        "flux-1.1-pro": 0,
        "flux-1-pro": 0,
        "stable-diffusion-xl-base-1.0": 0,
    }

    BASE_URL: str = "https://api.together.xyz/v1"


class GroqInfo(ExtInfo):
    _MODEL_POINTERS = {
        "distil-whisper-large-v3-en": "distil-whisper-large-v3-en",
        "gemma2-9b-it": "gemma2-9b-it",
        "llama3-groq-70b-8192-tool-use-preview": "llama3-groq-70b-8192-tool-use-preview",
        "llama3-groq-8b-8192-tool-use-preview": "llama3-groq-8b-8192-tool-use-preview",
        "llama-3.1-70b-versatile": "llama-3.1-70b-versatile",
        "llama-3.1-70b-specdec": "llama-3.1-70b-specdec",
        "llama-3.1-8b-instant": "llama-3.1-8b-instant",
        "llama-3.2-1b-preview": "llama-3.2-1b-preview",
        "llama-3.2-3b-preview": "llama-3.2-3b-preview",
        "llama-3.2-11b-vision-preview": "llama-3.2-11b-vision-preview",
        "llama-3.2-90b-vision-preview": "llama-3.2-90b-vision-preview",
        "llama-guard-3-8b": "llama-guard-3-8b",
        "llama3-70b-8192": "llama3-70b-8192",
        "llama3-8b-8192": "llama3-8b-8192",
        "mixtral-8x7b-32768": "mixtral-8x7b-32768",
    }

    _MODEL_CAPABILITIES: Dict[str, ModelInfo] = {
        "distil-whisper-large-v3-en": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "family": "openai-whisper",
        },
        "gemma2-9b-it": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "family": "google-gemma",
        },
        "llama3-groq-70b-8192-tool-use-preview": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "family": "meta-llama3",
        },
        "llama3-groq-8b-8192-tool-use-preview": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "family": "meta-llama3",
        },
        "llama-3.1-70b-versatile": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "family": "meta-llama3.1",
        },
        "llama-3.1-70b-specdec": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "family": "meta-llama3.1",
        },
        "llama-3.1-8b-instant": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "family": "meta-llama3.1",
        },
        "llama-3.2-1b-preview": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "family": "meta-llama3.2",
        },
        "llama-3.2-3b-preview": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "family": "meta-llama3.2",
        },
        "llama-3.2-11b-vision-preview": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "family": "meta-llama3.2",
        },
        "llama-3.2-90b-vision-preview": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "family": "meta-llama3.2",
        },
        "llama-guard-3-8b": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "family": "meta-llama3",
        },
        "llama3-70b-8192": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "family": "meta-llama3",
        },
        "llama3-8b-8192": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "family": "meta-llama3",
        },
        "mixtral-8x7b-32768": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "family": "misitral-mixtral",
        },
    }

    _MODEL_TOKEN_LIMITS: Dict[str, int] = {
        "gemmi2-9b-it": 8192,
        "llama3-groq-70b-8192-tool-use-preview": 8192,
        "llama3-groq-8b-8192-tool-use-preview": 8192,
        "llama-3.1-70b-versatile": 32768,
        "llama-3.1-70b-specdec": 8192,
        "llama-3.1-8b-instant": 8192,
        "llama-3.2-1b-preview": 8192,
        "llama-3.2-3b-preview": 8192,
        "llama-3.2-11b-vision-preview": 8192,
        "llama-3.2-90b-vision-preview": 8192,
        "llama-guard-3-8b": 8192,
        "llama3-70b-8192": 8192,
        "llama3-8b-8192": 8192,
        "mixtral-8x7b-32768": 32768,
    }

    BASE_URL: str = "https://api.groq.com/openai/v1"
