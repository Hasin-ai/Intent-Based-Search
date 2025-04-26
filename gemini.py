# Run this to test the Gemini API
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
import time # Import the time module

load_dotenv()

# client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
# # Alternate Model : gemini-embedding-exp-03-07


# # Generate embeddings for a single string

# start_time = time.time() # Record start time for performance measurement


# result = client.models.embed_content(
#     model="text-embedding-004",
#     contents="brown",
#     # Optional: optimize for a task
#     # config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
#     config=types.EmbedContentConfig(output_dimensionality=10, 
#                                      task_type="SEMANTIC_SIMILARITY")
#     # config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
# )

# end_time = time.time() # Record end time


# print(type(result.embeddings))  # List[float] of length 3072
# print(f"API call took: {end_time - start_time:.4f} seconds") # Print the duration


# print((result.embeddings[0].values))  



# Load model directly
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en")
model = AutoModel.from_pretrained("BAAI/bge-small-en")
import torch
import numpy as np
# Tokenize the input text
inputs = tokenizer("brown", return_tensors="pt")
# Get the embeddings
with torch.no_grad():
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
# Print the shape of the embeddings
print(embeddings.shape)  # Should be (768,)
# Print the embeddings
print(embeddings)
# Print the embeddings as a list
print(embeddings.tolist())
# Print the embeddings as a numpy array 
# 