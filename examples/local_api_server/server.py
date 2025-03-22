#!/usr/bin/env python
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""
Example Local API Server for GraphRAG.

This example demonstrates a simple API server that can be used with the Local API provider in GraphRAG.
It provides mock implementations for both the chat and embedding endpoints.

To run this server:
1. Install requirements: pip install fastapi uvicorn numpy
2. Run the server: uvicorn server:app --reload --port 8000
3. Configure GraphRAG to use the Local API provider with the appropriate endpoints
"""

import asyncio
import json
import random
from typing import Dict, List, Optional, Union

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Chat request model."""

    prompt: str
    history: Optional[List[Dict[str, str]]] = Field(default_factory=list)
    model: Optional[str] = "local-model"
    max_tokens: Optional[int] = 500
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    stream: Optional[bool] = False


class EmbeddingRequest(BaseModel):
    """Embedding request model."""

    texts: List[str]
    model: Optional[str] = "local-embedding-model"


app = FastAPI(title="GraphRAG Local API Example")


@app.get("/")
def read_root():
    """Root endpoint."""
    return {
        "message": "GraphRAG Local API Example Server",
        "endpoints": [
            {"path": "/generate", "method": "POST", "description": "Generate text from a prompt"},
            {"path": "/embeddings", "method": "POST", "description": "Generate embeddings for text"},
        ]
    }


@app.post("/generate")
async def generate(request: ChatRequest):
    """
    Generate text from a prompt.
    
    If stream=True, returns a streaming response.
    Otherwise, returns a JSON response.
    """
    if request.stream:
        return StreamingResponse(
            stream_response(request.prompt, request.max_tokens),
            media_type="text/event-stream"
        )
    
    # Simple mock response
    response_text = f"This is a response to: {request.prompt}"
    
    # If we have history, acknowledge it in the response
    if request.history and len(request.history) > 0:
        response_text += f"\nI see you have {len(request.history)} previous messages."
    
    return {"text": response_text, "model": request.model}


async def stream_response(prompt: str, max_tokens: int):
    """Stream a response one token at a time."""
    words = f"This is a streaming response to: {prompt}".split()
    
    for i, word in enumerate(words):
        # Convert to the format expected by the client
        data = json.dumps({"text": word + " "})
        yield f"data: {data}\n\n"
        await asyncio.sleep(0.1)  # Simulate delay between words
    
    yield f"data: {json.dumps({'text': '\n'})}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/embeddings")
async def embeddings(request: EmbeddingRequest):
    """Generate embeddings for text."""
    # Create random embeddings of dimension 384 (simulating a small embedding model)
    dimension = 384
    
    # Generate deterministic embeddings based on text content
    embeddings_list = []
    for text in request.texts:
        # Use a hash of the text to seed the random generator for deterministic results
        np.random.seed(hash(text) % 2**32)
        
        # Generate a random embedding
        embedding = np.random.normal(0, 1, dimension)
        
        # Normalize to unit length
        embedding = embedding / np.linalg.norm(embedding)
        
        embeddings_list.append(embedding.tolist())
    
    return {
        "model": request.model,
        "embeddings": embeddings_list
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000) 