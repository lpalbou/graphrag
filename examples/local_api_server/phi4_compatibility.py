#!/usr/bin/env python
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""
Phi-4-4B Server Compatibility Checker for GraphRAG.

This script tests if a running Phi-4-4B server is compatible with GraphRAG's
Local API provider by making sample requests to both the generate and embeddings
endpoints and verifying the response format.
"""

import argparse
import json
import sys
from typing import Dict, List, Optional, Union

import requests


def test_chat_endpoint(base_url: str, is_streaming: bool = False) -> bool:
    """Test the chat generation endpoint."""
    endpoint = f"{base_url}/generate"
    
    # Test payload that should work with the Phi-4-4B server
    payload = {
        "prompt": "What are the key features of GraphRAG?",
        "model": "phi-4-mini",
        "max_tokens": 200,
        "temperature": 0.7,
        "stream": is_streaming
    }
    
    print(f"\nTesting {'streaming' if is_streaming else 'standard'} chat endpoint at {endpoint}...")
    
    try:
        if is_streaming:
            # For streaming, we need to use a different approach
            response = requests.post(endpoint, json=payload, stream=True)
            
            if response.status_code != 200:
                print(f"❌ Error: Received status code {response.status_code}")
                return False
            
            # Check if we're getting SSE format responses
            first_chunk = True
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith('data:'):
                        # This looks like a proper SSE format
                        if first_chunk:
                            print("✅ Streaming response format looks correct")
                            first_chunk = False
                        # Just read a few chunks, then break
                        if '[DONE]' in decoded_line:
                            break
            return True
        else:
            # Standard non-streaming request
            response = requests.post(endpoint, json=payload)
            
            if response.status_code != 200:
                print(f"❌ Error: Received status code {response.status_code}")
                return False
            
            result = response.json()
            
            # Check expected fields
            if "text" in result and isinstance(result["text"], str):
                print(f"✅ Response contains text field: {result['text'][:50]}...")
                return True
            else:
                print("❌ Error: Response doesn't contain expected 'text' field")
                print(f"Received: {result}")
                return False
            
    except Exception as e:
        print(f"❌ Error connecting to server: {e}")
        return False


def test_embeddings_endpoint(base_url: str) -> bool:
    """Test the embeddings endpoint."""
    endpoint = f"{base_url}/embeddings"
    
    # Test payload for embeddings
    payload = {
        "texts": ["This is a test sentence for embeddings.", 
                  "GraphRAG is a graph-based RAG system."],
        "model": "phi-4-mini"
    }
    
    print(f"\nTesting embeddings endpoint at {endpoint}...")
    
    try:
        response = requests.post(endpoint, json=payload)
        
        if response.status_code != 200:
            print(f"❌ Error: Received status code {response.status_code}")
            return False
        
        result = response.json()
        
        # Check expected fields
        if "embeddings" in result and isinstance(result["embeddings"], list):
            if len(result["embeddings"]) == len(payload["texts"]):
                print(f"✅ Received {len(result['embeddings'])} embeddings as expected")
                
                # Check if embeddings are properly formatted
                if isinstance(result["embeddings"][0], list) and len(result["embeddings"][0]) > 0:
                    print(f"✅ Embeddings have correct format (vector of length {len(result['embeddings'][0])})")
                    return True
                else:
                    print("❌ Error: Embeddings are not in the expected format")
                    return False
            else:
                print(f"❌ Error: Expected {len(payload['texts'])} embeddings, got {len(result['embeddings'])}")
                return False
        else:
            print("❌ Error: Response doesn't contain expected 'embeddings' field")
            print(f"Received: {result}")
            return False
            
    except Exception as e:
        print(f"❌ Error connecting to server: {e}")
        return False


def test_graphrag_compatibility(host: str, port: int) -> bool:
    """Run all compatibility tests for GraphRAG."""
    base_url = f"http://{host}:{port}"
    
    print(f"Testing GraphRAG compatibility with server at {base_url}")
    print("=" * 80)
    
    # Test basic connectivity
    try:
        response = requests.get(base_url)
        if response.status_code == 200:
            print("✅ Server is running and reachable")
        else:
            print(f"❌ Server returned unexpected status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Unable to connect to server: {e}")
        return False
    
    # Run tests
    standard_chat_result = test_chat_endpoint(base_url, is_streaming=False)
    streaming_chat_result = test_chat_endpoint(base_url, is_streaming=True)
    embeddings_result = test_embeddings_endpoint(base_url)
    
    # Print summary
    print("\nSummary:")
    print("=" * 80)
    print(f"Standard Chat Generation: {'✅ PASSED' if standard_chat_result else '❌ FAILED'}")
    print(f"Streaming Chat Generation: {'✅ PASSED' if streaming_chat_result else '❌ FAILED'}")
    print(f"Embeddings Generation:    {'✅ PASSED' if embeddings_result else '❌ FAILED'}")
    
    # Print GraphRAG configuration example
    print("\nIf all tests passed, you can use the following configuration in your settings.yaml:")
    print("=" * 80)
    print("""
models:
  phi4_chat:
    type: local_api_chat
    api_base: "{base_url}/generate"
    model: "phi-4-mini"
    concurrent_requests: 1
    max_tokens: 500
    temperature: 0.7
    top_p: 0.95
    request_timeout: 60

  phi4_embedding:
    type: local_api_embedding
    api_base: "{base_url}/embeddings"
    model: "phi-4-mini"
    concurrent_requests: 1
    request_timeout: 60
""".format(base_url=base_url))
    
    # Overall result
    all_passed = standard_chat_result and streaming_chat_result and embeddings_result
    return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Phi-4-4B server compatibility with GraphRAG")
    parser.add_argument("--host", default="127.0.0.1", help="Server host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    
    args = parser.parse_args()
    
    result = test_graphrag_compatibility(args.host, args.port)
    
    if result:
        print("\n✅ All tests passed! Your Phi-4-4B server is compatible with GraphRAG.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. See above for details.")
        sys.exit(1) 