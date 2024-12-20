import uuid
import time
from typing import Any, Dict, Optional, Type, Generator
from .types import EnumLLMProvider, LLMGatewayParams, LLMGatewayConfig, ChatCompletionParams, LLMProvider
from .providers.openai.gateway import OpenAIGateway
from .providers.anthropic.gateway import AnthropicGateway
from .providers.azure.gateway import AzureGateway
from .utils.logger import log_debug, log_error, log_info
from .utils.param_converters import (
    convert_openai_to_anthropic,
    convert_anthropic_to_openai,
    is_anthropic_format,
    is_openai_format
)

class LLMGateway:
    def __init__(self, params: LLMGatewayParams, config: Optional[LLMGatewayConfig] = None):
        self.config = config or LLMGatewayConfig()
        log_debug('Initializing LLMGateway', {'provider': params.provider})
        self.provider = self._init_provider(params)
        
        if config and config.fallbacks:
            log_info('Configuring fallback provider', {
                'provider': config.fallbacks.get('fallback_provider', {}).get('provider'),
                'model': config.fallbacks.get('fallback_model')
            })
            self.fallback_provider = self._init_provider(config.fallbacks['fallback_provider'])
        else:
            self.fallback_provider = None

    def _init_provider(self, params: LLMGatewayParams) -> LLMProvider:
        provider_map = {
            EnumLLMProvider.OPENAI: OpenAIGateway,
            EnumLLMProvider.AZUREOPENAI: AzureGateway,
            EnumLLMProvider.ANTHROPIC: AnthropicGateway
        }
        
        provider_class = provider_map.get(params.provider)
        if not provider_class:
            raise ValueError(f'Unsupported provider: {params.provider}')
            
        return provider_class(
            api_key=params.api_key,
            endpoint=params.endpoint,
            api_version=params.api_version
        )

    def _retry_with_backoff(self, operation: Any, params: ChatCompletionParams) -> Any:
        attempts = 0
        max_retries = self.config.retries or 0
        
        while True:
            request_id = str(uuid.uuid4())
            try:
                log_debug('Starting operation attempt', {
                    'request_id': request_id,
                    'attempt': attempts + 1,
                    'model': params.model
                })
                
                response = operation()
                
                log_info('Operation successful', {
                    'request_id': request_id,
                    'model': params.model,
                    'attempts': attempts + 1
                })
                
                return response
                
            except Exception as error:
                attempts += 1
                remaining_retries = max_retries - attempts
                
                log_error('Operation failed', error, {
                    'request_id': request_id,
                    'attempt': attempts,
                    'remaining_retries': remaining_retries
                })
                
                if attempts > max_retries or not self._should_retry(error):
                    raise error
                
                delay = self._calculate_backoff(attempts)
                log_debug('Retrying after delay', {
                    'request_id': request_id,
                    'delay_ms': delay,
                    'attempt': attempts + 1
                })
                
                time.sleep(delay / 1000)

    def _should_retry(self, error: Exception) -> bool:
        error_str = str(error).lower()
        if 'rate limit' in error_str or '429' in error_str:
            return True
        if 'timeout' in error_str or 'connection' in error_str:
            return True
        return False

    def _calculate_backoff(self, attempt: int) -> float:
        base = min(1000 * (2 ** (attempt - 1)), 10000)
        jitter = (time.time() * 1000) % 1000
        return base + jitter

    def _convert_params_for_provider(self, params: ChatCompletionParams) -> ChatCompletionParams:
        messages = params.messages
        if self.provider.name == EnumLLMProvider.ANTHROPIC and is_openai_format(messages[0].__dict__):
            messages = convert_openai_to_anthropic([m.__dict__ for m in messages])
        elif self.provider.name in [EnumLLMProvider.OPENAI, EnumLLMProvider.AZUREOPENAI] and is_anthropic_format(messages[0].__dict__):
            messages = convert_anthropic_to_openai([m.__dict__ for m in messages])
        return ChatCompletionParams(**{**params.__dict__, 'messages': messages})

    def chat_completion(self, params: ChatCompletionParams) -> Any:
        try:
            return self._retry_with_backoff(
                lambda: self.provider.chat_completion(params),
                params
            )
        except Exception as error:
            log_info('All chat completion attempts failed, trying fallback')
            
            if not self.fallback_provider or not self.config.fallbacks or not self.config.fallbacks.get('fallback_model'):
                raise error

            try:
                converted_params = self._convert_params_for_provider(params)
                converted_params.model = self.config.fallbacks['fallback_model']
                
                return self._retry_with_backoff(
                    lambda: self.fallback_provider.chat_completion(converted_params),
                    converted_params
                )
            except Exception as fallback_error:
                log_error('Fallback provider also failed', fallback_error)
                raise fallback_error

    def chat_completion_stream(self, params: ChatCompletionParams) -> Generator[Dict[str, Any], None, None]:
        try:
            yield from self._retry_with_backoff(
                lambda: self.provider.chat_completion_stream(params),
                params
            )
        except Exception as error:
            log_info('Stream failed, trying fallback')
            
            if not self.fallback_provider or not self.config.fallbacks or not self.config.fallbacks.get('fallback_model'):
                raise error

            converted_params = self._convert_params_for_provider(params)
            converted_params.model = self.config.fallbacks['fallback_model']
            
            yield from self._retry_with_backoff(
                lambda: self.fallback_provider.chat_completion_stream(converted_params),
                converted_params
            )
