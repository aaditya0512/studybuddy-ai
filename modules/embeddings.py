import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from config import Config

def get_embeddings():
    """Returns the Google Gemini Embeddings model."""
    api_key = Config.GOOGLE_API_KEY
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment.")
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2", google_api_key=api_key)
    return embeddings
