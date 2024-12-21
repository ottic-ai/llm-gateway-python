from enum import Enum
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass

class EnumLLMProvider(str, Enum):
    OPENAI = 'openai'
    AZUREOPENAI = 'azureopenai'
    ANTHROPIC = 'anthropic'

@dataclass
class LLMGatewayParams:
    provider: EnumLLMProvider
    api_key: Optional[str] = None
    endpoint: Optional[str] = None
    deployment: Optional[str] = None
    api_version: Optional[str] = None

@dataclass
class LLMGatewayConfig:
    retries: Optional[int] = None
    timeout: Optional[int] = None
    fallbacks: Optional[Dict[str, Any]] = None

@dataclass
class Message:
    role: str
    content: Union[str, List[Dict[str, Any]]]
    name: Optional[str] = None

@dataclass
class ChatCompletionParams:
    model: str
    messages: List[Message]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Dict[str, Any]] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    system: Optional[Union[str, List[Dict[str, Any]]]] = None
    top_k: Optional[int] = None
    stop_sequences: Optional[List[str]] = None

class LLMProvider:
    def __init__(self, name: EnumLLMProvider):
        self.name = name

    def chat_completion(self, params: ChatCompletionParams) -> Any:
        raise NotImplementedError

    def chat_completion_stream(self, params: ChatCompletionParams) -> Any:
        raise NotImplementedError
