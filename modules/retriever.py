import os
from langchain_pinecone import PineconeVectorStore
from modules.embeddings import get_embeddings
from config import Config

def store_chunks(chunks, collection_name="studybuddy_collection"):
    """Stores chunks into Pinecone using namespaces."""
    embeddings = get_embeddings()
    
    if not Config.PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY is not set.")
        
    vectorstore = PineconeVectorStore.from_texts(
        texts=chunks, 
        embedding=embeddings, 
        index_name=Config.PINECONE_INDEX_NAME,
        namespace=collection_name
    )
    return vectorstore

def get_vectorstore(collection_name="studybuddy_collection"):
    """Gets the existing Pinecone vector store."""
    embeddings = get_embeddings()
    
    if not Config.PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY is not set.")
        
    vectorstore = PineconeVectorStore(
        index_name=Config.PINECONE_INDEX_NAME, 
        embedding=embeddings,
        namespace=collection_name
    )
    return vectorstore

def retrieve_relevant_chunks(query, k=4, collection_name="studybuddy_collection"):
    """Retrieves top k relevant chunks for a given query."""
    vectorstore = get_vectorstore(collection_name)
    docs = vectorstore.similarity_search(query, k=k)
    return [doc.page_content for doc in docs]
