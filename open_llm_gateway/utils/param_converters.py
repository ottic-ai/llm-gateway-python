from typing import Dict, Any, List, Optional, Union

from src.types import ChatCompletionParams, Message

def is_openai_format(params: Dict[str, Any]) -> bool:
    return any(key in params for key in ['n', 'tools', 'tool_choice', 'response_format', 'max_completion_tokens'])

def is_anthropic_format(params: Dict[str, Any]) -> bool:
    return any(key in params for key in ['system', 'stop_sequences', 'top_k'])

def convert_tool_choice_to_anthropic(tool_choice: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(tool_choice, str):
        if tool_choice == 'auto':
            return {'type': 'auto'}
        elif tool_choice == 'required':
            return {'type': 'any'}
        return None
    elif isinstance(tool_choice, dict) and tool_choice.get('type') == 'function':
        return {
            'type': 'tool',
            'name': tool_choice.get('function', {}).get('name')
        }
    return None

def convert_tool_choice_to_openai(tool_choice: Dict[str, Any]) -> Union[str, Dict[str, Any]]:
    tool_type = tool_choice.get('type')
    if tool_type == 'auto':
        return 'auto'
    elif tool_type == 'any':
        return 'required'
    elif tool_type == 'tool' and 'name' in tool_choice:
        return {
            'type': 'function',
            'function': {'name': tool_choice['name']}
        }
    return None
def convert_openai_to_anthropic(params: ChatCompletionParams) -> ChatCompletionParams:
    anthropic_params = ChatCompletionParams(messages=[], model=params.model)

    # Extract system message
    messages = params.messages
    system_message = next((msg for msg in messages if msg.role == 'system'), None)
    if system_message:
        anthropic_params.system = str(system_message.content)

    # Convert messages format
    anthropic_params.messages = [
        Message(
            role=msg.role,
            content=msg.content
        )
        for msg in messages
        if msg.role != 'system'
    ]

    # Convert tools
    if params.tools:
        anthropic_params.tools = [{
            'name': tool.function.name,
            'description': tool.function.description,
            'input_schema': tool.function.parameters
        } for tool in params.tools]

        # Handle tool_choice
        if params.tool_choice:
            anthropic_params.tool_choice = convert_tool_choice_to_anthropic(params.tool_choice)

    # Convert max_tokens
    if params.max_completion_tokens:
        anthropic_params.max_tokens = params.max_completion_tokens
    else:
        anthropic_params.max_tokens = 1000
    return anthropic_params


def convert_anthropic_to_openai(params: ChatCompletionParams) -> ChatCompletionParams:
    openai_params = ChatCompletionParams(messages=[], model=params.model)

    # Convert system to message
    if params.system:
        openai_params.messages.append(Message(role='system', content=params.system))

    # Add other messages
    openai_params.messages.extend([
        Message(
            role=msg.role,
            content=msg.content
        )
        for msg in params.messages
    ])

    # Convert tools
    if params.tools:
        openai_params.tools = [{
            'type': 'function',
            'function': {
                'name': tool['name'],
                'description': tool.get('description'),
                'parameters': tool.get('input_schema')
            }
        } for tool in params.tools]

        # Handle tool_choice
        if params.tool_choice:
            openai_params.tool_choice = convert_tool_choice_to_openai(params.tool_choice)

    # Convert max_tokens
    if params.max_tokens:
        openai_params.max_completion_tokens = params.max_tokens


    return openai_params
