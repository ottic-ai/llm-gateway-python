from typing import Any, Dict, Optional, Generator
from openai import OpenAI
from ...types import EnumLLMProvider, LLMProvider, ChatCompletionParams

class OpenAIGateway(LLMProvider):
    def __init__(self, api_key: str, **kwargs):
        super().__init__(EnumLLMProvider.OPENAI)
        self.client = OpenAI(api_key=api_key)

    def chat_completion(self, params: ChatCompletionParams) -> Dict[str, Any]:
        response = self.client.chat.completions.create(
            model=params.model,
            messages=[msg.__dict__ for msg in params.messages],
            temperature=params.temperature,
            max_tokens=params.max_tokens,
            top_p=params.top_p,
            tools=params.tools,
            tool_choice=params.tool_choice
        )

        completion = response.choices[0]
        if completion.finish_reason == 'tool_calls':
            response.llm_gateway_output = [{
                'type': 'tool_calls',
                'tool_name': completion.message.tool_calls[0].function.name,
                'arguments': completion.message.tool_calls[0].function.arguments
            }]
        else:
            response.llm_gateway_output = [{
                'type': 'text',
                'content': completion.message.content
            }]
        return response

    def chat_completion_stream(self, params: ChatCompletionParams) -> Generator[Dict[str, Any], None, None]:
        stream = self.client.chat.completions.create(
            model=params.model,
            messages=[msg.__dict__ for msg in params.messages],
            temperature=params.temperature,
            max_tokens=params.max_tokens,
            top_p=params.top_p,
            tools=params.tools,
            tool_choice=params.tool_choice,
            stream=True
        )

        for chunk in stream:
            completion = chunk.choices[0]
            if completion.finish_reason == 'tool_calls':
                chunk.llm_gateway_output = [{
                    'type': 'tool_calls',
                    'tool_name': completion.message.tool_calls[0].function.name,
                    'arguments': completion.message.tool_calls[0].function.arguments
                }]
            else:
                chunk.llm_gateway_output = [{
                    'type': 'text',
                    'content': completion.delta.content if completion.delta.content else ''
                }]
            yield chunk
