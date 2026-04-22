import os
import json
from typing import Annotated, TypedDict, Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()

# --- 1. State & Schema Definition ---
class UserInfo(BaseModel):
    name: str = Field(default=None)
    email: str = Field(default=None)
    platform: str = Field(default=None)

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    intent: str
    user_info: UserInfo # Tracks lead details across turns

# Initialize LLM
model_name = os.getenv("LLM_MODEL", "gemini-3-flash-preview")
llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)
print(f"--- Agent initialized with model: {model_name} ---")

# --- 2. Node Functions ---

def greeting_node(state: AgentState):
    """Responds to casual greetings."""
    return {"messages": [AIMessage(content="Hello! I'm the AutoStream assistant. How can I help you with your video editing needs today?")]}

def intent_classifier(state: AgentState):
    """Classifies user intent into Greeting, Inquiry, or High-Intent."""
    last_msg = state["messages"][-1].content
    prompt = f"Classify intent: '{last_msg}'. Categories: 'greeting', 'inquiry', 'lead'. Return only the word."
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"intent": response.content.strip().lower()}

def router(state: AgentState):
    intent = state.get("intent")
    if intent == "inquiry": return "rag"
    if intent == "lead": return "lead"
    if intent == "greeting": return "greeting" # Point to the new node
    return END

def rag_node(state: AgentState):
    """Retrieves AutoStream info from the local knowledge base."""
    # Ensure the path is correct for your VS Code structure
    kb_path = os.path.join("data", "knowledge_base.md")
    
    with open(kb_path, "r", encoding="utf-8") as f:
        kb_content = f.read()
    
    user_msg = state["messages"][-1].content
    
    # Combined prompt in a single HumanMessage for better 2026 model stability
    full_prompt = f"""
    You are the AutoStream expert. Use the Knowledge Base below to answer the user's question accurately.
    
    KNOWLEDGE BASE:
    {kb_content}
    
    USER QUESTION:
    {user_msg}
    
    ANSWER:
    """
    
    # Use HumanMessage instead of SystemMessage if the crash persists
    response = llm.invoke([HumanMessage(content=full_prompt)])
    return {"messages": [AIMessage(content=response.content)]}

def lead_capture_node(state: AgentState):
    """Collects missing info (Name, Email, Platform) before triggering tool."""
    info = state.get("user_info", UserInfo())
    last_msg = state["messages"][-1].content
    
    # Simple extraction logic (can be improved with structured output)
    prompt = f"Extract name, email, platform from: '{last_msg}'. Current: {info.dict()}. Return JSON."
    # For the assignment, if info is complete, we call the mock tool
    if info.name and info.email and info.platform:
        print(f"--- MOCK TOOL TRIGGERED: Lead captured for {info.name} ---") # cite: 53-54
        return {"messages": [AIMessage(content="Details saved! You're all set.")]}
    
    # Otherwise, ask for what's missing
    return {"messages": [AIMessage(content="I'd love to help! Can you provide your name, email, and platform?")]}

# --- 3. Graph Construction ---
workflow = StateGraph(AgentState)
workflow.add_node("classifier", intent_classifier)
workflow.add_node("rag", rag_node)
workflow.add_node("lead", lead_capture_node)
workflow.add_node("greeting", greeting_node) # ADD THIS LINE

workflow.set_entry_point("classifier")
workflow.add_conditional_edges("classifier", router)
workflow.add_edge("greeting", END) # ADD THIS LINE
workflow.add_edge("rag", END)
workflow.add_edge("lead", END)

app = workflow.compile()