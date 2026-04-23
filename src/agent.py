import os
import json
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

# --- 1. Mandatory Mock Function ---
def mock_lead_capture(name, email, platform):
    """
    Required Function per Assignment Specifications[cite: 52, 53].
    This function is only called once all user details are collected[cite: 51, 74].
    """
    print(f"\n[SYSTEM] Lead captured successfully: {name}, {email}, {platform}") # [cite: 54]

# --- 2. State Definition ---
class AgentState(TypedDict):
    # Retains memory across the mandatory 5-6 conversation turns [cite: 90, 91]
    messages: Annotated[list, add_messages]
    intent: str
    # Tracks user details (Name, Email, Creator Platform) [cite: 48, 49, 50]
    user_info: dict 

# Fetch model from .env (Updated to Gemini 3.1 Flash Lite)
model_name = os.getenv("LLM_MODEL", "gemini-3.1-flash-lite")
llm = ChatGoogleGenerativeAI(model=model_name, temperature=0)

# --- 3. Node Functions ---

def intent_classifier(state: AgentState):
    """Identifies if the user is greeting, inquiring, or showing high-intent [cite: 20-23]."""
    last_msg = state["messages"][-1].content
    prompt = f"Classify intent: '{last_msg}'. Options: 'greeting', 'inquiry', 'lead'. Return ONLY the word."
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"intent": response.content.strip().lower()}

def greeting_node(state: AgentState):
    """Handles casual greetings[cite: 21, 58]."""
    return {"messages": [AIMessage(content="Hello! I'm the AutoStream assistant. How can I help you with your video editing needs today?")]}

def rag_node(state: AgentState):
    """Answers pricing and feature questions using local knowledge retrieval[cite: 24, 25, 43]."""
    kb_path = os.path.join("data", "knowledge_base.md")
    try:
        with open(kb_path, "r", encoding="utf-8") as f:
            kb_content = f.read()
    except FileNotFoundError:
        return {"messages": [AIMessage(content="I'm sorry, I cannot access my pricing data right now.")]}
    
    user_msg = state["messages"][-1].content
    # Context-aware retrieval using the AutoStream KB [cite: 26, 27, 40]
    prompt = f"Use this AutoStream Knowledge Base:\n{kb_content}\n\nUser Question: {user_msg}\n\nAnswer accurately:"
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"messages": [AIMessage(content=response.content)]}

def lead_capture_node(state: AgentState):
    """Collects Name, Email, and Platform before triggering the mock tool [cite: 45-51, 112]."""
    info = state.get("user_info", {"name": None, "email": None, "platform": None})
    last_msg = state["messages"][-1].content
    
    # Logic to extract details from the current message
    extraction_prompt = f"""
    Current info: {info}
    Extract Name, Email, and Creator Platform from the user message: "{last_msg}"
    Return as a JSON object. If a value is unknown, keep it as null.
    """
    response = llm.invoke([SystemMessage(content=extraction_prompt)])
    try:
        extracted = json.loads(response.content)
        # Update state with newly extracted info
        for key in info:
            if extracted.get(key):
                info[key] = extracted[key]
    except:
        pass # Fallback if model doesn't return clean JSON
    
    # Ensure tool is not triggered prematurely [cite: 55, 73]
    if not info["name"]:
        return {"messages": [AIMessage(content="I'd love to help you with that! What's your name?")], "user_info": info}
    if not info["email"]:
        return {"messages": [AIMessage(content=f"Nice to meet you, {info['name']}! What's your email address?")], "user_info": info}
    if not info["platform"]:
        return {"messages": [AIMessage(content="And which creator platform do you primarily use (YouTube, Instagram, etc.)?")], "user_info": info}
    
    # Final step: Tool Execution [cite: 51, 71, 113, 119]
    mock_lead_capture(info['name'], info['email'], info['platform'])
    return {"messages": [AIMessage(content="Perfect! Your details have been captured. An AutoStream expert will reach out soon.")], "user_info": info}

# --- 4. Build the Graph ---
workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("classifier", intent_classifier)
workflow.add_node("greeting", greeting_node)
workflow.add_node("rag", rag_node)
workflow.add_node("lead", lead_capture_node)

# Routing Logic [cite: 79, 97]
def router(state: AgentState):
    it = state.get("intent")
    if it == "inquiry": return "rag"
    if it == "lead": return "lead"
    if it == "greeting": return "greeting"
    return END

workflow.set_entry_point("classifier")
workflow.add_conditional_edges("classifier", router)
workflow.add_edge("greeting", END)
workflow.add_edge("rag", END)
workflow.add_edge("lead", END)

# Compile with memory for persistent state management [cite: 91, 118]
app = workflow.compile(checkpointer=MemorySaver())