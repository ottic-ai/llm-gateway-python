from typing import Any, Dict, Optional, Generator
from openai import AzureOpenAI
from ..openai.gateway import OpenAIGateway
from ...types import EnumLLMProvider, ChatCompletionParams

class AzureGateway(OpenAIGateway):
    def __init__(self, api_key: str, endpoint: str, api_version: Optional[str] = None, **kwargs):
        super().__init__(EnumLLMProvider.AZUREOPENAI)
        self.client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            api_version=api_version or "2023-05-15"
        )
