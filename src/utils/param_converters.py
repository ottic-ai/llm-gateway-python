from typing import Dict, Any, List

def is_openai_format(message: Dict[str, Any]) -> bool:
    return isinstance(message.get('content'), str) and 'role' in message

def is_anthropic_format(message: Dict[str, Any]) -> bool:
    return isinstance(message.get('content'), list) and 'role' in message

def convert_openai_to_anthropic(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    converted_messages = []
    for message in messages:
        converted_message = {
            'role': message['role'],
            'content': [{'type': 'text', 'text': message['content']}] if isinstance(message['content'], str) else message['content']
        }
        if 'name' in message:
            converted_message['name'] = message['name']
        converted_messages.append(converted_message)
    return converted_messages

def convert_anthropic_to_openai(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    converted_messages = []
    for message in messages:
        content = ''
        if isinstance(message['content'], list):
            for block in message['content']:
                if block['type'] == 'text':
                    content += block['text']
        else:
            content = message['content']
            
        converted_message = {
            'role': message['role'],
            'content': content
        }
        if 'name' in message:
            converted_message['name'] = message['name']
        converted_messages.append(converted_message)
    return converted_messages
