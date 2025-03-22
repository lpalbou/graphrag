import os
import sys
from typing import Optional, List, Any
import anthropic
from pydantic import BaseModel

# Define a simple ModelOutput class similar to BaseModelOutput
class SimpleModelOutput(BaseModel):
    content: str

# Define a simple ModelResponse class similar to BaseModelResponse
class SimpleModelResponse(BaseModel):
    output: SimpleModelOutput
    parsed_response: Optional[Any] = None
    history: List[Any] = []
    tool_calls: List[Any] = []
    metrics: Optional[Any] = None
    cache_hit: Optional[bool] = None
    raw_response: Optional[Any] = None

def convert_history_to_anthropic_format(prompt: str, history: Optional[List] = None):
    """Convert chat history to Anthropic format."""
    messages = []
    
    if history:
        for entry in history:
            if entry.get("role") == "user":
                messages.append({"role": "user", "content": entry.get("content", "")})
            elif entry.get("role") == "assistant":
                messages.append({"role": "assistant", "content": entry.get("content", "")})
    
    # Add the current prompt
    messages.append({"role": "user", "content": prompt})
    
    return messages

async def achat(client, model, prompt, history=None, max_tokens=1000, temperature=0.7, top_p=0.95):
    """Async Chat method for Anthropic."""
    try:
        # Convert history to Anthropic format if provided
        messages = convert_history_to_anthropic_format(prompt, history)
        
        # Make the API call
        response = await client.messages.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        
        # Extract the text from the response
        content = response.content
        text = ""
        if content:
            for block in content:
                if block.type == "text":
                    text += block.text
        
        # Create a properly formatted ModelResponse
        model_output = SimpleModelOutput(content=text)
        return SimpleModelResponse(
            output=model_output,
            raw_response=response
        )
    except Exception as e:
        print(f"Error during Anthropic API call: {str(e)}")
        raise

def run_coroutine_sync(coroutine):
    """Run a coroutine synchronously."""
    import asyncio
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coroutine)

def test_anthropic():
    """Test function to try Anthropic API."""
    print(f"Using Python from: {sys.executable}")

    # Get API key from environment
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not found.")
        return

    print("API key found.")

    try:
        # Create Anthropic client
        print("Creating Anthropic client...")
        client = anthropic.AsyncAnthropic(api_key=api_key)
        
        # Set model name with correct format
        model = "claude-3-5-haiku-20241022"
        
        # Send a message to Claude
        prompt = "Hello, how are you?"
        response = run_coroutine_sync(achat(client, model, prompt))
        
        # Print out the response
        print(f"Response: {response.output.content}")
        print("Response object structure:")
        print(f"- output: {response.output}")
        print(f"- parsed_response: {response.parsed_response}")
        print(f"- history: {response.history}")
        print(f"- tool_calls: {response.tool_calls}")
        print(f"- metrics: {response.metrics}")
        print(f"- cache_hit: {response.cache_hit}")
        print(f"- raw_response: {type(response.raw_response)}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_anthropic() 