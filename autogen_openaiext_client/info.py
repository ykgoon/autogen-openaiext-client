from typing import Dict

from autogen_core.components.models import ModelCapabilities


class ExtInfo:
    _MODEL_POINTERS = {}

    _MODEL_CAPABILITIES: Dict[str, ModelCapabilities] = {}

    _MODEL_TOKEN_LIMITS: Dict[str, int] = {}

    BASE_URL: str = ""

    @staticmethod
    def resolve_model(model: str) -> str:
        if model in GeminiInfo._MODEL_POINTERS:
            return GeminiInfo._MODEL_POINTERS[model]
        return model

    @staticmethod
    def get_capabilities(model: str) -> ModelCapabilities:
        resolved_model = GeminiInfo.resolve_model(model)
        return GeminiInfo._MODEL_CAPABILITIES[resolved_model]

    @staticmethod
    def get_token_limit(model: str) -> int:
        resolved_model = GeminiInfo.resolve_model(model)
        return GeminiInfo._MODEL_TOKEN_LIMITS[resolved_model]

    def _add_model(
        self, model_name: str, model_capabilities: ModelCapabilities, token_limit: int
    ):
        self._MODEL_CAPABILITIES[model_name] = model_capabilities
        self._MODEL_TOKEN_LIMITS[model_name] = token_limit

    @classmethod
    def add_model(
        cls, model_name: str, model_capabilities: ModelCapabilities, token_limit: int
    ):
        cls._add_model(model_name, model_capabilities, token_limit)


class GeminiInfo(ExtInfo):
    _MODEL_POINTERS = {"gemini-1.5-flash": "gemini-1.5-flash"}

    _MODEL_CAPABILITIES: Dict[str, ModelCapabilities] = {
        "gemini-1.5-flash": {
            "vision": True,
            "function_calling": True,
            "json_output": True,
        },
        "gemini-1.5-pro": {
            "vision": True,
            "function_calling": True,
            "json_output": True,
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
        "mythomax-l2-13b": "mythom",
        "llama-2-13b-chat-hf": "llama-2-13b-chat-hf",
        "mistral-7b-instruct": "mistral-7b-instruct",
        "mistral-7b-instruct-v0.2": "mistral-7b-instruct-v0.2",
        "mistral-7b-instruct-v0.3": "mistral-7b-instruct-v0.3",
        "mixtral-8x7b-instruct": "mixtral-8x7b-instruct",
        "mixtral-8x22b-instruct": "mixtral-8x22b-instruct",
        "nous-hermes-2-mixtral-8x7b-dpo": "nous-hermes-2-mixtral-8x7b-dpo",
        "qwen-2.5-7b-instruct-turbo": "qwen-2.5-7b-instruct-turbo",
        "qwen-2.5-72b-instruct-turbo": "qwen-2.5-72b-instruct-turbo",
        "qwen-2-instruct": "qwen-2-instruct",
        "upstage-solar-instruct-v1": "upstage-solar-instruct-v1",
        "llama-vision-free": "llama-vision-free",
        "llama-3.2-11b-vision-instruct-turbo": "llama-3.2-11b-vision-instruct-turbo",
        "llama-3.2-90b-vision-instruct-turbo": "llama-3.2-90b-vision-instruct-turbo",
        "llama-2-70b-hf": "llama-2-70b-hf",
        "mistral-7b": "mistral-7b",
        "mixtral-8x7b": "mixtral-8x7b",
        "flux-1-schnell-free": "flux-1-schnell-free",
        "flux-1-schnell": "flux-1-schnell",
        "flux-1-dev": "flux-1-dev",
        "flux-1-canny": "flux-1-canny",
        "flux-1-depth": "flux-1-depth",
        "flux-1-redux": "flux-1-redux",
        "flux-1.1-pro": "flux-1.1-pro",
        "flux-1-pro": "flux-1-pro",
        "stable-diffusion-xl-base-1.0": "stable-diffusion-xl-base-1.0",
    }

    _MODEL_CAPABILITIES: Dict[str, ModelCapabilities] = {
        "llama-3.1-8b-instruct-turbo": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
        },
        "llama-3.1-70b-instruct-turbo": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
        },
        "llama-3.1-405b-instruct-turbo": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
        },
        "llama-3-8b-instruct-turbo": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
        },
        "llama-3-70b-instruct-turbo": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
        },
        "llama-3.2-3b-instruct-turbo": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
        },
        "llama-3-8b-instruct-lite": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
        },
        "llama-3-70b-instruct-lite": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
        },
        "llama-3-8b-instruct-reference": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
        },
        "llama-3-70b-instruct-reference": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
        },
        "llama-3.1-nemotron-70b": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
        },
        "qwen-2.5-coder-32b-instruct": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
        },
        "qwen-32b-preview": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
        },
        "wizardlm-2-8x22b": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
        },
        "gemma-2-27b": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
        },
        "gemma-2-9b": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
        },
        "dbrx-instruct": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
        },
        "deepseek-llm-67b-chat": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
        },
        "gemma-2b-it": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
        },
        "mythomax-l2-13b": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
        },
        "llama-2-13b-chat-hf": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
        },
        "mistral-7b-instruct": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
        },
        "mistral-7b-instruct-v0.2": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
        },
        "mistral-7b-instruct-v0.3": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
        },
        "mixtral-8x7b-instruct": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
        },
        "mixtral-8x22b-instruct": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
        },
        "nous-hermes-2-mixtral-8x7b-dpo": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
        },
        "qwen-2.5-7b-instruct-turbo": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
        },
        "qwen-2.5-72b-instruct-turbo": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
        },
        "qwen-2-instruct": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
        },
        "upstage-solar-instruct-v1": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
        },
        "llama-vision-free": {
            "vision": True,
            "function_calling": False,
            "json_output": False,
        },
        "llama-3.2-11b-vision-instruct-turbo": {
            "vision": True,
            "function_calling": False,
            "json_output": False,
        },
        "llama-3.2-90b-vision-instruct-turbo": {
            "vision": True,
            "function_calling": False,
            "json_output": False,
        },
        "llama-2-70b-hf": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
        },
        "mistral-7b": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
        },
        "mixtral-8x7b": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
        },
        "flux-1-schnell-free": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
        },
        "flux-1-schnell": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
        },
        "flux-1-dev": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
        },
        "flux-1-canny": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
        },
        "flux-1-depth": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
        },
        "flux-1-redux": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
        },
        "flux-1.1-pro": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
        },
        "flux-1-pro": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
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
        "gemma-7b-it": "gemma-7b-it",
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

    _MODEL_CAPABILITIES: Dict[str, ModelCapabilities] = {
        "distil-whisper-large-v3-en": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
        },
        "gemma2-9b-it": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
        },
        "gemma-7b-it": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
        },
        "llama3-groq-70b-8192-tool-use-preview": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
        },
        "llama3-groq-8b-8192-tool-use-preview": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
        },
        "llama-3.1-70b-versatile": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
        },
        "llama-3.1-70b-specdec": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
        },
        "llama-3.1-8b-instant": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
        },
        "llama-3.2-1b-preview": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
        },
        "llama-3.2-3b-preview": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
        },
        "llama-3.2-11b-vision-preview": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
        },
        "llama-3.2-90b-vision-preview": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
        },
        "llama-guard-3-8b": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
        },
        "llama3-70b-8192": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
        },
        "llama3-8b-8192": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
        },
        "mixtral-8x7b-32768": {
            "vision": False,
            "function_calling": False,
            "json_output": False,
        },
    }

    _MODEL_TOKEN_LIMITS: Dict[str, int] = {
        "gemmi2-9b-it": 8192,
        "gemma-7b-it": 8192,
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
