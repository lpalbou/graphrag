# Local API Server Example for GraphRAG

This is a simple example of how to create a server that implements the API expected by GraphRAG's Local API provider.

## Overview

The server provides two main endpoints:

1. `/generate` - Text generation endpoint for chat models
2. `/embeddings` - Embedding generation endpoint for embedding models

Both endpoints provide mock implementations to demonstrate the expected request and response formats.

## Running the Server

### Installation

```bash
pip install -r requirements.txt
```

### Starting the Server

```bash
uvicorn server:app --reload --port 8000
```

This will start the server at `http://127.0.0.1:8000`.

## API Documentation

Once the server is running, you can view the API documentation at:

```
http://127.0.0.1:8000/docs
```

## Using with GraphRAG

To use this server with GraphRAG, configure your `settings.yaml` file with the following models:

```yaml
models:
  local_chat:
    type: local_api_chat
    api_base: "http://127.0.0.1:8000/generate"
    model: "local-model"
    concurrent_requests: 1
    max_tokens: 500
    temperature: 0.7
    top_p: 0.95
    request_timeout: 30

  local_embedding:
    type: local_api_embedding
    api_base: "http://127.0.0.1:8000/embeddings"
    model: "local-embedding-model"
    concurrent_requests: 4
    request_timeout: 30
```

Then, update your workflow configurations to use these model IDs:

```yaml
extract_graph:
  model_id: local_chat
  prompt: "prompts/extract_graph.txt"
  # ... other settings ...

embed_text:
  model_id: local_embedding
  # ... other settings ...
```

## Extending for Real Models

In a real implementation, you would replace the mock generation and embedding logic with calls to your locally hosted models. Some options include:

- [Ollama](https://github.com/ollama/ollama) for local LLMs
- [TGI (Text Generation Inference)](https://github.com/huggingface/text-generation-inference) for text generation
- [ONNX Runtime](https://onnxruntime.ai/) for running models locally
- [SentenceTransformers](https://www.sbert.net/) for embeddings 

## Compatibility Checker for Phi-4-4B Server

We've included a compatibility checker script specifically designed to test if your running Phi-4-4B server is compatible with GraphRAG's Local API provider.

### Running the Compatibility Check

After starting your Phi-4-4B server, run the compatibility checker:

```bash
python phi4_compatibility.py --host 127.0.0.1 --port 8000
```

The script will test:
1. Basic connectivity to the server
2. Standard (non-streaming) chat generation
3. Streaming chat generation
4. Embedding generation

### Interpreting Results

If all tests pass, you'll see a summary of the results and a sample configuration for your `settings.yaml` file that you can use with GraphRAG.

If any tests fail, the checker will provide specific error messages to help you identify and fix compatibility issues.

### Using Phi-4-4B with GraphRAG

Once your server passes all compatibility tests, you can use Phi-4-4B with GraphRAG by:

1. Copying the provided YAML configuration to your `settings.yaml` file
2. Updating your workflow configurations to use the new model IDs (`phi4_chat` and `phi4_embedding`)

## Phi-4-4B FastAPI Adapter Server

If your Phi-4-4B server doesn't fully implement the GraphRAG-compatible API, you can use the provided adapter server as a bridge between GraphRAG and your Phi-4-4B server.

### Features

- Acts as a proxy between GraphRAG and the Phi-4-4B server
- Adapts the request/response formats as needed
- Supports both streaming and non-streaming generation
- Handles embedding requests
- Includes system prompt support

### Running the Adapter Server

First, make sure your Phi-4-4B server is running, then start the adapter server:

```bash
python phi4_fastapi_server.py --host 127.0.0.1 --port 8001 --phi4-host 127.0.0.1 --phi4-port 8000
```

### Configuration

With the adapter server running, configure GraphRAG to connect to it instead of directly to the Phi-4-4B server:

```yaml
models:
  phi4_chat:
    type: local_api_chat
    api_base: "http://127.0.0.1:8001/generate"  # Point to the adapter server
    model: "phi-4-mini"
    concurrent_requests: 1
    max_tokens: 500
    temperature: 0.7
    top_p: 0.95
    request_timeout: 60

  phi4_embedding:
    type: local_api_embedding
    api_base: "http://127.0.0.1:8001/embeddings"  # Point to the adapter server
    model: "phi-4-mini"
    concurrent_requests: 1
    request_timeout: 60
``` 