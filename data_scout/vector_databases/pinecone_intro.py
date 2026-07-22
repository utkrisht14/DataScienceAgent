from pinecone import Pinecone
import os
from dotenv import load_dotenv

load_dotenv()

MODEL = "llama-text-embed-v2"

# Initialize the Pinecone client with the API key
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Create a new index
index_name = "pinecone-programs-index"

if not pc.has_index(index_name):
    pc.create_index_for_model(
        name = index_name,
        cloud = "aws",
        region = "us-east-1",
        embed = {
            "model": MODEL,
            "field_map" : {
                "text" : "chunk_text"
            }
        }
    )


