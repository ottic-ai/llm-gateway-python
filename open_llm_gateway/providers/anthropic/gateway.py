from typing import Any, Dict, Optional, Generator
from anthropic import NOT_GIVEN, Anthropic
from ...types import EnumLLMProvider, LLMProvider, ChatCompletionParams

class AnthropicGateway(LLMProvider):
    def __init__(self, api_key: str, **kwargs):
        super().__init__(EnumLLMProvider.ANTHROPIC)
        self.client = Anthropic(api_key=api_key)

    def chat_completion(self, params: ChatCompletionParams) -> Dict[str, Any]:
        messages = [{key: value for key, value in msg.__dict__.items() if key != 'name'} for msg in params.messages]
        response = self.client.messages.create(
            model=params.model,
            messages=messages,
            max_tokens=params.max_tokens,
            temperature=params.temperature or NOT_GIVEN,
            top_p=params.top_p or NOT_GIVEN,
            system=params.system or NOT_GIVEN,
            tools=params.tools or NOT_GIVEN
        )

        if response.stop_reason == 'tool_use':
            tool_use_block = response.content[-1]
            response.llm_gateway_output = [{
                'type': 'tool_calls',
                'tool_name': tool_use_block.name,
                'arguments': tool_use_block.input
            }]
        else:
            response.llm_gateway_output = [{
                'type': 'text',
                'content': block.text
            } for block in response.content if block.type == 'text']
        return response

    def chat_completion_stream(self, params: ChatCompletionParams) -> Generator[Dict[str, Any], None, None]:
        messages = [{key: value for key, value in msg.__dict__.items() if key != 'name'} for msg in params.messages]
        stream = self.client.messages.create(
            model=params.model,
            messages=messages,
            max_tokens=params.max_tokens,
            temperature=params.temperature or NOT_GIVEN,
            top_p=params.top_p or NOT_GIVEN,
            system=params.system or NOT_GIVEN,
            tools=params.tools or NOT_GIVEN,
            stream=True
        )
        for chunk in stream:
            yield chunk
