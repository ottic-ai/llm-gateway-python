# LLM Gateway
#### Open-source library built for fast and reliable connections to different LLM providers.

Typescript library: https://www.npmjs.com/package/llm-gateway

The LLM Gateway is a lightweight, open-source library built for fast and reliable connections to LLMs. <br>

It simplifies integrations with multiple providers, offering fallbacks, caching, and minimal latency with a client-side solution.


- **Minimize Downtime**: Automatic retries and fallbacks to secondary providers like Azure or Entropic.  
- **Automatic input params convertion**: Automatically convert input params between OpenAI, Anthropic and Azure formats for fallbacks.
- **Faster Responses**: Direct client-side requests for low latency.  
- **Unified Control**: A single interface to manage requests across LLMs. 
- **Unified Output**: Consistent output format across LLMs. 
```{
    openAI/AnthropicOutput:{...}
    llmGatewayOutput: {
        type: 'text' | 'tool_calls';
        content: string; - content for text output
        tool_name: string; - name of the tool for tool_calls
        arguments: string - arguments for the tool.
    }[]
}
```
- **Easy Model Switching**: Change between OpenAI, Anthropic, and Azure models with a simple configuration change. 

> Contribute, fork, or raise issues— so we can make it better together.

> Starring this repo helps other developers discover the LLM Gateway! ⭐  


## Installation

To install the library, use npm or yarn:

```bash
pip install llmgateway
```




## Usage

Here's a basic example of how to use the LLM Gateway library:

```python
from src.llmgateway import LLMGateway
from src.types import ChatCompletionParams, EnumLLMProvider, LLMGatewayParams, Message

gateway = LLMGateway(
    LLMGatewayParams(
        provider=EnumLLMProvider.OPENAI,
        api_key='your-api-key'
    )
)
response = gateway.chat_completion(
    ChatCompletionParams(
        messages=[
            Message(role="user", content="Write a story about a cat.")
        ],
        model="gpt-4o-mini",
        temperature=0.7,
        max_completion_tokens=800
    )
)

```

## LLM Fallbacks Configuration

The LLM Gateway library supports configuring fallbacks to ensure that if one model fails, another can be used as a backup. This is useful for maintaining service availability and reliability.

### Example Configuration

```python

from src.llmgateway import LLMGateway
from src.types import ChatCompletionParams, EnumLLMProvider, LLMGatewayConfig, LLMGatewayParams, Message


gateway = LLMGateway(
    LLMGatewayParams(
                provider= EnumLLMProvider.ANTHROPIC,
                api_key= keys['ANTHROPIC_API_KEY']
            )
    , LLMGatewayConfig(
        fallbacks= {
            'fallback_provider':LLMGatewayParams(
        provider=EnumLLMProvider.OPENAI,
        api_key=keys['OPENAI_KEY_FOR_FREE_DEMO']
    ) ,
            'fallback_model': 'gpt-4o-mini'
        }
    )
)
response = gateway.chat_completion(
    ChatCompletionParams(
        messages=[
            Message(role="user", content="Write a story about a cat.")
        ],
        model="claude-3-5-sonnet-latest",
        temperature=0.7,
    )
)
print(response)
```
## Streaming Responses

The LLM Gateway supports streaming responses from all providers, with a unified interface that works consistently across OpenAI, Anthropic, and Azure.

### Basic Streaming Example

```python
from src.llmgateway import LLMGateway
from src.types import ChatCompletionParams, EnumLLMProvider, LLMGatewayParams, Message


gateway = LLMGateway(
    LLMGatewayParams(
        provider=EnumLLMProvider.OPENAI,
        api_key='your-api-key'
    )
)

stream = gateway.chat_completion_stream(
    ChatCompletionParams(
        messages=[
            Message(role="user", content="Write a story about a cat.")
        ],
        model="gpt-4o-mini",
        temperature=0.7,
        max_tokens=800
    )
)

for chunk in stream:
    print(chunk)

```

All streaming examples work consistently across different providers (OpenAI, Anthropic, Azure) and automatically handle format conversion when falling back to a different provider.

## Configuration

- **apiKey**: Your API key for the chosen LLM provider.
- **modelType**: The type of LLM provider you want to use (`OPENAI`, `ANTHROPIC`, `AZUREOPENAI`).
- **endpoint**: (Optional) The endpoint for OpenAI models.
- **deployment**: (Optional) The deployment name for Azure models.
- **apiVersion**: (Optional) The API version for Azure models.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License.
