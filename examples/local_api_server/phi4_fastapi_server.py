#!/usr/bin/env python
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""
Phi-4-4B FastAPI Server for GraphRAG.

This is a FastAPI server that adapts the Phi-4-4B server to the GraphRAG
Local API provider interface. It acts as a bridge between the two systems.
"""

import asyncio
import json
import logging
import os
from typing import Dict, List, Optional, Union, Any

import requests
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("phi4_fastapi_server")

# Create FastAPI app
app = FastAPI(title="Phi-4-4B FastAPI Server for GraphRAG")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration for the Phi-4-4B server
PHI4_SERVER_HOST = os.environ.get("PHI4_SERVER_HOST", "127.0.0.1")
PHI4_SERVER_PORT = int(os.environ.get("PHI4_SERVER_PORT", "8000"))
PHI4_SERVER_BASE_URL = f"http://{PHI4_SERVER_HOST}:{PHI4_SERVER_PORT}"

# Models for request/response
class ChatRequest(BaseModel):
    """Chat request model for GraphRAG-compatible API."""
    prompt: str
    history: Optional[List[Dict[str, str]]] = Field(default_factory=list)
    model: Optional[str] = "phi-4-mini"
    max_tokens: Optional[int] = 500
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    stream: Optional[bool] = False
    system_prompt: Optional[str] = None

class EmbeddingRequest(BaseModel):
    """Embedding request model for GraphRAG-compatible API."""
    texts: List[str]
    model: Optional[str] = "phi-4-mini"


@app.get("/")
def read_root():
    """Root endpoint."""
    return {
        "message": "Phi-4-4B FastAPI Server for GraphRAG",
        "endpoints": [
            {"path": "/generate", "method": "POST", "description": "Generate text from a prompt"},
            {"path": "/embeddings", "method": "POST", "description": "Generate embeddings for text"},
        ]
    }


@app.post("/generate")
async def generate(request: ChatRequest):
    """
    Generate text endpoint that adapts to the Phi-4-4B server.
    
    If stream=True, returns a streaming response.
    Otherwise, returns a JSON response.
    """
    try:
        # Prepare the request for the Phi-4-4B server
        phi4_request = {
            "prompt": request.prompt,
            "model": request.model,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stream": request.stream
        }
        
        # If there's a system prompt, pass it to the Phi-4-4B server
        if request.system_prompt:
            phi4_request["system_prompt"] = request.system_prompt
        
        # Log what we're doing
        logger.info(f"Forwarding request to Phi-4-4B server: {PHI4_SERVER_BASE_URL}/generate")
        
        if request.stream:
            # Handle streaming response
            async def stream_response():
                # Create a streaming request to the Phi-4-4B server
                response = requests.post(
                    f"{PHI4_SERVER_BASE_URL}/generate",
                    json=phi4_request,
                    stream=True
                )
                
                # Forward the streaming response
                for line in response.iter_lines():
                    if line:
                        yield line + b"\n"
                        await asyncio.sleep(0)  # Allow other tasks to run
            
            return StreamingResponse(
                stream_response(),
                media_type="text/event-stream"
            )
        else:
            # Handle non-streaming response
            response = requests.post(
                f"{PHI4_SERVER_BASE_URL}/generate",
                json=phi4_request
            )
            
            if response.status_code != 200:
                logger.error(f"Error from Phi-4-4B server: {response.status_code}")
                return Response(
                    content=json.dumps({"error": f"Phi-4-4B server error: {response.status_code}"}),
                    status_code=response.status_code,
                    media_type="application/json"
                )
            
            result = response.json()
            return result
    
    except Exception as e:
        logger.error(f"Error generating text: {e}")
        return {"error": str(e)}


@app.post("/embeddings")
async def embeddings(request: EmbeddingRequest):
    """Generate embeddings for text by forwarding to the Phi-4-4B server."""
    try:
        # Prepare the request for the Phi-4-4B server
        phi4_request = {
            "texts": request.texts,
            "model": request.model
        }
        
        # Log what we're doing
        logger.info(f"Forwarding embeddings request to Phi-4-4B server: {PHI4_SERVER_BASE_URL}/embeddings")
        
        # Forward the request to the Phi-4-4B server
        response = requests.post(
            f"{PHI4_SERVER_BASE_URL}/embeddings",
            json=phi4_request
        )
        
        if response.status_code != 200:
            logger.error(f"Error from Phi-4-4B server: {response.status_code}")
            return Response(
                content=json.dumps({"error": f"Phi-4-4B server error: {response.status_code}"}),
                status_code=response.status_code,
                media_type="application/json"
            )
        
        # Return the embeddings
        result = response.json()
        return result
    
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Phi-4-4B FastAPI Server for GraphRAG")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind the server to")
    parser.add_argument("--phi4-host", default="127.0.0.1", help="Phi-4-4B server host")
    parser.add_argument("--phi4-port", type=int, default=8000, help="Phi-4-4B server port")
    
    args = parser.parse_args()
    
    # Set environment variables for the Phi-4-4B server
    os.environ["PHI4_SERVER_HOST"] = args.phi4_host
    os.environ["PHI4_SERVER_PORT"] = str(args.phi4_port)
    
    print(f"Starting Phi-4-4B FastAPI Server for GraphRAG on {args.host}:{args.port}")
    print(f"Connecting to Phi-4-4B server at {args.phi4_host}:{args.phi4_port}")
    
    # Run the server
    uvicorn.run(app, host=args.host, port=args.port) 