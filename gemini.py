# Run this to test the Gemini API
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
import time # Import the time module

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
# Alternate Model : gemini-embedding-exp-03-07


# Generate embeddings for a single string

start_time = time.time() # Record start time for performance measurement


result = client.models.embed_content(
    model="text-embedding-004",
    contents="brown",
    # Optional: optimize for a task
    # config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
    config=types.EmbedContentConfig(output_dimensionality=10, 
                                     task_type="SEMANTIC_SIMILARITY")
    # config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
)

end_time = time.time() # Record end time


print(type(result.embeddings))  # List[float] of length 3072
print(f"API call took: {end_time - start_time:.4f} seconds") # Print the duration


print((result.embeddings[0].values))  

