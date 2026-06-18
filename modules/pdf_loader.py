import os
import pdfplumber

def extract_text_from_pdf(filepath):
    """Extracts all text from a given PDF file using pdfplumber."""
    text = ""
    try:
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    except Exception as e:
        print(f"Error reading PDF {filepath}: {e}")
        return None
