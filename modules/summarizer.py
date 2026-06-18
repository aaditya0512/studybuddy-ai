from modules.chatbot import get_llm
from langchain_core.prompts import PromptTemplate

def generate_summary(text, length="short"):
    """Generates a summary of the provided text."""
    llm = get_llm()
    
    if length == "short":
        instructions = "Provide a brief 3-sentence summary of the following text."
    elif length == "medium":
        instructions = "Provide a structured summary of the following text with key points."
    else:
        instructions = "Provide a highly detailed summary of the following text, capturing all important concepts and details."
        
    prompt_template = """
    {instructions}
    
    Text:
    {text}
    
    Summary:
    """
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["instructions", "text"])
    formatted_prompt = prompt.format(instructions=instructions, text=text)
    
    response = llm.invoke(formatted_prompt)
    return response.content
