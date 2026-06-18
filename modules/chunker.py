import re
from langchain_text_splitters import RecursiveCharacterTextSplitter

def clean_text(text):
    """Removes extra spaces and empty lines."""
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r' +', ' ', text)
    return text.strip()

def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    """Splits text into smaller chunks."""
    cleaned_text = clean_text(text)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_text(cleaned_text)
    return chunks
