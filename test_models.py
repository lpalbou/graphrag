#!/usr/bin/env python
"""Test script for Anthropic and SentenceTransformer models.

This script tests the direct integration of Anthropic for chat and SentenceTransformer 
for embeddings without relying on the full GraphRAG pipeline.
"""

import os
import asyncio
from pathlib import Path
import json

# Import sentence_transformers directly 
from sentence_transformers import SentenceTransformer

# Set up Anthropic parameters
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY environment variable must be set")

# Set up models
model_name = "claude-3-5-haiku-20241022"
embedding_model_name = "all-MiniLM-L6-v2"

# Test sentences
test_sentences = [
    "GraphRAG combines knowledge graphs with traditional RAG for improved retrieval.",
    "The system extracts entities and relationships to create knowledge graphs.",
    "These graphs are used to improve contextual information retrieval.",
]

def test_sentence_transformer():
    """Test SentenceTransformer embedding generation."""
    print("\n--- Testing SentenceTransformer ---")
    
    try:
        # Load the model
        print(f"Loading embedding model: {embedding_model_name}")
        model = SentenceTransformer(embedding_model_name)
        
        # Generate embeddings
        print("Generating embeddings...")
        embeddings = model.encode(test_sentences)
        
        # Print results
        print(f"Generated {len(embeddings)} embeddings")
        print(f"Embedding dimension: {embeddings.shape[1]}")
        print("First embedding sample (first 10 values):", embeddings[0][:10])
        
        return True
    except Exception as e:
        print(f"Error in SentenceTransformer test: {str(e)}")
        return False

async def test_anthropic():
    """Test Anthropic API for chat."""
    print("\n--- Testing Anthropic API ---")
    
    try:
        # Dynamically import anthropic to avoid issues if not installed
        import anthropic
        
        # Create client
        print("Creating Anthropic client...")
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        
        # Send a test message
        print(f"Sending test message to model: {model_name}")
        prompt = "Explain what GraphRAG is in 2-3 sentences."
        
        message = client.messages.create(
            model=model_name,
            max_tokens=150,
            temperature=0.7,
            system="You are a helpful AI assistant.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        # Print response
        print("\nResponse:")
        print(f"{message.content[0].text}")
        
        return True
    except Exception as e:
        print(f"Error in Anthropic test: {str(e)}")
        return False

async def main():
    """Run all tests."""
    print("=== Testing Anthropic and SentenceTransformer Models ===")
    
    # Test SentenceTransformer
    st_success = test_sentence_transformer()
    
    # Test Anthropic
    anthropic_success = await test_anthropic()
    
    # Print summary
    print("\n=== Test Summary ===")
    print(f"SentenceTransformer: {'✅ Passed' if st_success else '❌ Failed'}")
    print(f"Anthropic API: {'✅ Passed' if anthropic_success else '❌ Failed'}")

if __name__ == "__main__":
    asyncio.run(main()) 