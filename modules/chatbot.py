from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from config import Config
from modules.retriever import retrieve_relevant_chunks

def get_llm():
    api_key = Config.GOOGLE_API_KEY
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment.")
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key, temperature=0.3)

def generate_answer(query, collection_name="studybuddy_collection"):
    """Generates an answer using retrieved chunks and Gemini."""
    chunks = retrieve_relevant_chunks(query, k=4, collection_name=collection_name)
    context = "\n\n".join(chunks)
    
    prompt_template = """
    Answer the question using only the provided context. If the answer is not contained in the context, say "I don't have enough information to answer that based on the uploaded documents."
    
    Context:
    {context}
    
    Question:
    {question}
    
    Answer:
    """
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    llm = get_llm()
    
    formatted_prompt = prompt.format(context=context, question=query)
    response = llm.invoke(formatted_prompt)
    
    return response.content, context
