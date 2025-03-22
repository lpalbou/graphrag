# GraphRAG Extended LLM Providers

This extension to GraphRAG adds support for additional LLM providers beyond the default OpenAI and Azure OpenAI support:

1. **Local API Provider**: Connect to locally hosted language models served through a simple API
2. **Anthropic Provider**: Connect to Anthropic's Claude models

## Installation

There are two ways to install the required dependencies:

### Option 1: Using pip with extras

Install GraphRAG with the extended providers support:

```bash
pip install graphrag[extended]
```

### Option 2: Using requirements file

Alternatively, you can install the dependencies directly using the provided requirements file:

```bash
pip install -r requirements-extended.txt
```

## Configuration

### Local API Configuration

The Local API provider allows you to connect GraphRAG to any HTTP endpoint that implements a simple API for text generation and embeddings:

```yaml
models:
  local_chat:
    type: local_api_chat
    api_base: "http://127.0.0.1:8000/generate" # or any other local endpoint
    model: "local_model_name"
    concurrent_requests: 4
    max_tokens: 4000
    temperature: 0.7
    top_p: 0.95
    request_timeout: 30

  local_embedding:
    type: local_api_embedding
    api_base: "http://127.0.0.1:8000/embeddings" # or any other local endpoint
    model: "local_embedding_model" 
    concurrent_requests: 4
    request_timeout: 30
```

### Anthropic Configuration

The Anthropic provider connects GraphRAG to Anthropic's Claude models:

```yaml
models:
  anthropic_chat:
    type: anthropic_chat
    api_key: ${GRAPHRAG_ANTHROPIC_API_KEY} # set this in the .env file
    model: "claude-3.5-haiku-20241022" # or any other Claude model
    max_tokens: 4000
    temperature: 0.7
    top_p: 0.95
    concurrent_requests: 2

  anthropic_embedding:
    type: anthropic_embedding
    api_key: ${GRAPHRAG_ANTHROPIC_API_KEY} # set this in the .env file
    model: "claude-3.5-haiku-20241022"
    concurrent_requests: 2
```

## Environment Variables

For the Anthropic provider, set your API key in the `.env` file:

```bash
GRAPHRAG_ANTHROPIC_API_KEY=<your-anthropic-api-key>
```

## API Specifications

### Local API Provider

The Local API provider expects the following interface:

#### Text Generation Endpoint

**POST** `/generate`

Request body:
```json
{
  "prompt": "Your prompt here",
  "model": "model-name",
  "max_tokens": 1000,
  "temperature": 0.7,
  "top_p": 0.95,
  "stream": false
}
```

Response body:
```json
{
  "text": "Generated text response..."
}
```

#### Embeddings Endpoint

**POST** `/embeddings`

Request body:
```json
{
  "texts": ["Text 1 to embed", "Text 2 to embed"],
  "model": "embedding-model-name"
}
```

Response body:
```json
{
  "embeddings": [[0.1, 0.2, ...], [0.3, 0.4, ...]]
}
```

## Local API Server Example

A simple example server implementation is provided in the `examples/local_api_server` directory. This shows the expected API format for serving local models.

## Known Limitations

- The Anthropic embedding implementation is a placeholder because Anthropic's API doesn't currently expose embeddings directly. It attempts to extract embeddings from the message response, but this may not work reliably.
- Local API servers must implement the expected request/response format detailed in the example server. 