import os
from typing import TypedDict, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from config import Config
from modules.retriever import retrieve_relevant_chunks
from langgraph.graph import StateGraph, START, END

# Define AgentState
class AgentState(TypedDict):
    question: str
    collection_name: str
    documents: List[str]
    generation: str
    route: str  # 'document_question' or 'greeting'
    relevant: bool  # True if documents are relevant

def get_llm():
    api_key = Config.GOOGLE_API_KEY
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment.")
    return ChatGoogleGenerativeAI(model="gemini-3-flash", google_api_key=api_key, temperature=0.3)

# ----------------- Nodes -----------------

def route_query(state: AgentState):
    """
    Route the question to determine if it is a general greeting/casual chat
    or a question requiring document retrieval.
    """
    question = state["question"]
    llm = get_llm()
    
    router_prompt = f"""You are a query router. Classify the user query into one of two categories:
    1. 'greeting': If the query is a simple greeting (e.g., hello, hi, hey, who are you, what can you do, etc.) or casual chit-chat.
    2. 'document_question': If the query is a question requesting information, explanation, or summary about specific concepts, subjects, or documents.
    
    Query: {question}
    
    Respond only with either 'greeting' or 'document_question'. Do not include any other text.
    Classification:"""
    
    response = llm.invoke(router_prompt)
    classification = response.content.strip().lower()
    
    # Clean output just in case
    if "greeting" in classification:
        route = "greeting"
    else:
        route = "document_question"
        
    return {"route": route}

def retrieve_context(state: AgentState):
    """Retrieves relevant chunks from the Pinecone vector database."""
    question = state["question"]
    collection_name = state["collection_name"]
    
    chunks = retrieve_relevant_chunks(question, k=4, collection_name=collection_name)
    return {"documents": chunks}

def grade_documents(state: AgentState):
    """
    Grades the retrieved documents for relevance to the question.
    """
    question = state["question"]
    documents = state.get("documents", [])
    
    if not documents:
        return {"relevant": False}
        
    context = "\n\n".join(documents)
    llm = get_llm()
    
    grader_prompt = f"""You are a document relevance grader. Grade if the provided document context contains information relevant to answering the user question.
    
    Document Context:
    {context}
    
    User Question:
    {question}
    
    Respond ONLY with 'yes' if the context is relevant/contains info to answer, or 'no' if the context is completely unrelated/cannot answer the question. Do not explain.
    Relevance (yes/no):"""
    
    response = llm.invoke(grader_prompt)
    decision = response.content.strip().lower()
    
    relevant = "yes" in decision
    return {"relevant": relevant}

def generate_answer_node(state: AgentState):
    """Generates an answer using the retrieved context."""
    question = state["question"]
    documents = state.get("documents", [])
    context = "\n\n".join(documents)
    
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
    
    formatted_prompt = prompt.format(context=context, question=question)
    response = llm.invoke(formatted_prompt)
    
    return {"generation": response.content}

def generate_general_chat(state: AgentState):
    """Handles general chit-chat or greetings directly."""
    question = state["question"]
    llm = get_llm()
    
    prompt = f"""You are StudyBuddy AI, an assistant helping students study their uploaded PDF documents, take quizzes, and study flashcards.
    Respond friendly and concisely to the following user query:
    {question}"""
    
    response = llm.invoke(prompt)
    return {"generation": response.content, "documents": []}

def generate_fallback(state: AgentState):
    """Handles cases where context is not relevant."""
    return {
        "generation": "I don't have enough information to answer that based on the uploaded documents.",
        "documents": state.get("documents", [])
    }

# ----------------- Router Edges -----------------

def decide_route(state: AgentState):
    """Conditional edge based on classification."""
    return state["route"]

def decide_generation(state: AgentState):
    """Conditional edge based on grading."""
    if state["relevant"]:
        return "generate"
    else:
        return "fallback"

# ----------------- Build Graph -----------------

workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("router", route_query)
workflow.add_node("retrieve", retrieve_context)
workflow.add_node("grade", grade_documents)
workflow.add_node("generate", generate_answer_node)
workflow.add_node("general_chat", generate_general_chat)
workflow.add_node("fallback", generate_fallback)

# Add Edges & Conditional Routing
workflow.add_edge(START, "router")

workflow.add_conditional_edges(
    "router",
    decide_route,
    {
        "greeting": "general_chat",
        "document_question": "retrieve"
    }
)

workflow.add_edge("retrieve", "grade")

workflow.add_conditional_edges(
    "grade",
    decide_generation,
    {
        "generate": "generate",
        "fallback": "fallback"
    }
)

workflow.add_edge("general_chat", END)
workflow.add_edge("generate", END)
workflow.add_edge("fallback", END)

# Compile Graph
compiled_app = workflow.compile()

# Public entry point that matches original signature
def generate_answer(query, collection_name="studybuddy_collection"):
    """
    Invokes the LangGraph Agent workflow.
    Returns: (generation, context_string)
    """
    inputs = {
        "question": query,
        "collection_name": collection_name,
        "documents": [],
        "generation": "",
        "route": "",
        "relevant": False
    }
    
    output = compiled_app.invoke(inputs)
    
    documents = output.get("documents", [])
    context = "\n\n".join(documents) if documents else ""
    return output.get("generation", ""), context
