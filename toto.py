import anthropic
import os

# Use your existing API key
api_key = os.environ.get("ANTHROPIC_API_KEY")

client = anthropic.Anthropic(api_key=api_key)

try:
    # Try to create an embedding
    response = client.embeddings.create(
        model="claude-3-haiku-20240307",
        input="This is a test to check if I have access to Anthropic embeddings."
    )
    print("Success! You have access to embeddings.")
    print("Embedding size: {}".format(len(response.embeddings[0].embedding)))
    print("First few values: {}".format(response.embeddings[0].embedding[:5]))
except Exception as e:
    print("Error: {}".format(e))
    print("You might not have access to embeddings or there might be another issue.")
