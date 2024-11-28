from typing import Optional, Dict, List, Union

from typing_extensions import TypedDict
from autogen_core.components.models import ModelCapabilities
from autogen_ext.models._openai.config import CreateArguments


class OpenAIExtClientConfiguration(CreateArguments, total=False):
    model: str
    api_key: str
    timeout: Union[float, None]
    max_retries: int
    model_capabilities: ModelCapabilities
    base_url: str
