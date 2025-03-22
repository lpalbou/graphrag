import os
import anthropic
import sys

print(f"Using Python from: {sys.executable}")

# Get API key from environment variable
api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    print("ANTHROPIC_API_KEY environment variable not set")
    exit(1)
else:
    print(f"API key found: {api_key[:5]}...")

# Create Anthropic client
print("Creating Anthropic client...")
client = anthropic.Anthropic(api_key=api_key)

# Send a simple message
try:
    print("Sending message to Claude...")
    message = client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=1000,
        messages=[{"role": "user", "content": "Hello, Claude! Please respond with a short greeting."}]
    )
    print("API call successful!")
    print(f"Response: {message.content[0].text}")
except Exception as e:
    print(f"Error: {e}") 