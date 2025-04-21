from google import genai
from google.genai import types

GEMINI_API_KEY="AIzaSyBwQdR9m-xEm6wgaxPb5vCDwc3k3a_rZf8"

client = genai.Client(api_key=GEMINI_API_KEY)

# Generate embeddings for a single string
result = client.models.embed_content(
    model="gemini-embedding-exp-03-07",
    contents="How do neural networks learn?",
    # Optional: optimize for a task
    config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
)

print(result.embeddings)  # List[float] of length 3072
