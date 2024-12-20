from typing import Any, Dict, Optional, Generator
from anthropic import Anthropic
from ...types import EnumLLMProvider, LLMProvider, ChatCompletionParams

class AnthropicGateway(LLMProvider):
    def __init__(self, api_key: str, **kwargs):
        super().__init__(EnumLLMProvider.ANTHROPIC)
        self.client = Anthropic(api_key=api_key)

    def chat_completion(self, params: ChatCompletionParams) -> Dict[str, Any]:
        response = self.client.messages.create(
            model=params.model,
            messages=[msg.__dict__ for msg in params.messages],
            temperature=params.temperature,
            max_tokens=params.max_tokens,
            top_p=params.top_p,
            system=params.system,
            tools=params.tools
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
        stream = self.client.messages.create(
            model=params.model,
            messages=[msg.__dict__ for msg in params.messages],
            temperature=params.temperature,
            max_tokens=params.max_tokens,
            top_p=params.top_p,
            system=params.system,
            tools=params.tools,
            stream=True
        )

        for chunk in stream:
            if hasattr(chunk, 'type'):
                if chunk.type == 'content_block_start':
                    chunk.llm_gateway_output = [{
                        'type': 'text',
                        'content': chunk.content_block.text
                    }]
                elif chunk.type == 'content_block_delta':
                    chunk.llm_gateway_output = [{
                        'type': 'text',
                        'content': chunk.delta.text
                    }]
                elif chunk.type == 'tool_use':
                    chunk.llm_gateway_output = [{
                        'type': 'tool_calls',
                        'tool_name': chunk.tool_calls[0].name,
                        'arguments': chunk.tool_calls[0].input
                    }]
            yield chunk
