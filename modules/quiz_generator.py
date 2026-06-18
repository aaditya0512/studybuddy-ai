from modules.chatbot import get_llm
from langchain_core.prompts import PromptTemplate
import json

def generate_quiz(text, num_questions=5):
    """Generates MCQs from text."""
    llm = get_llm()
    
    import random
    salt = random.randint(1000, 9999)
    
    prompt_template = """
    [System Seed: {salt}]
    Generate {num_questions} unique, highly diverse, and challenging Multiple Choice Questions based on the following text.
    Ensure these questions are different from typical, obvious ones.
    Return the result ONLY as a JSON array of objects. Do not include markdown code blocks or any other text.
    Each object should have:
    - "question": The question text
    - "options": An array of 4 strings
    - "answer": The correct option string
    - "explanation": Brief explanation of the answer
    
    Text:
    {text}
    """
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["num_questions", "text", "salt"])
    formatted_prompt = prompt.format(num_questions=num_questions, text=text, salt=salt)
    
    response = llm.invoke(formatted_prompt)
    try:
        content = response.content.strip()
        if content.startswith('```json'):
            content = content[7:-3].strip()
        elif content.startswith('```'):
            content = content[3:-3].strip()
        return json.loads(content)
    except Exception as e:
        print(f"Error parsing quiz JSON: {e}")
        print("Raw response:", response.content)
        return []

def generate_flashcards(text, num_cards=5):
    """Generates Flashcards from text."""
    llm = get_llm()
    
    prompt_template = """
    Generate {num_cards} flashcards based on the following text.
    Return the result ONLY as a JSON array of objects. Do not include markdown code blocks or any other text.
    Each object should have:
    - "question": The concept or question
    - "answer": The definition or answer
    
    Text:
    {text}
    """
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["num_cards", "text"])
    formatted_prompt = prompt.format(num_cards=num_cards, text=text)
    
    response = llm.invoke(formatted_prompt)
    try:
        content = response.content.strip()
        if content.startswith('```json'):
            content = content[7:-3].strip()
        elif content.startswith('```'):
            content = content[3:-3].strip()
        return json.loads(content)
    except Exception as e:
        return []
