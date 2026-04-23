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
    """Accurately identifies user intent with support for list-based content blocks."""
    last_msg = state["messages"][-1].content
    prompt = f"Classify intent: '{last_msg}'. Options: 'greeting', 'inquiry', 'lead'. Return ONLY the word."
    response = llm.invoke([HumanMessage(content=prompt)])
    
    # 2026 SDK Fix: Handle content whether it's a string or a list of blocks
    content = response.content
    if isinstance(content, list):
        # Join all text blocks together
        content = "".join([c if isinstance(c, str) else c.get("text", "") for c in content])
    
    return {"intent": str(content).strip().lower()}

def get_clean_text(response):
    """Helper to strip metadata and signatures from 2026 model outputs."""
    if isinstance(response.content, list):
        return "".join([p.get("text", "") for p in response.content if isinstance(p, dict) and "text" in p])
    return response.content

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
    """Collects Name, Email, and Platform."""
    info = state.get("user_info", {"name": None, "email": None, "platform": None})
    
    # --- 2026 FIX: Robust Content Extraction ---
    last_msg = state["messages"][-1].content
    if isinstance(last_msg, list):
        last_msg = "".join([p.get("text", "") for p in last_msg if isinstance(p, dict) and "text" in p])
    
    # Combine System Instructions + Human Message to avoid 'contents are required' error
    extraction_prompt = f"""
    SYSTEM: You are a data extraction bot.
    CURRENT DATA: {info}
    USER MESSAGE: "{last_msg}"
    
    TASK: Extract Name, Email, and Creator Platform. 
    Return ONLY a JSON object. If a value is missing, use null.
    """
    
    # We use HumanMessage here to satisfy the API's requirement for 'user' content
    response = llm.invoke([HumanMessage(content=extraction_prompt)])
    
    # Clean the response (handling the 2026 metadata blocks)
    res_content = response.content
    if isinstance(res_content, list):
        res_content = "".join([p.get("text", "") for p in res_content if isinstance(p, dict) and "text" in p])
    
    try:
        # Simple extraction logic
        import json
        # Strip potential markdown code blocks if the model adds them
        json_str = res_content.replace("```json", "").replace("```", "").strip()
        extracted = json.loads(json_str)
        for key in info:
            if extracted.get(key):
                info[key] = extracted[key]
    except:
        pass # Fallback if extraction fails
    
    # --- 4. Logic to ask for missing info ---
    if not info.get("name"):
        return {"messages": [AIMessage(content="I'd love to help you with that! What's your name?")], "user_info": info}
    if not info.get("email"):
        return {"messages": [AIMessage(content=f"Thanks, {info['name']}! And your email address?")], "user_info": info}
    if not info.get("platform"):
        return {"messages": [AIMessage(content="Almost there! Which platform do you use (YouTube or Instagram)?")], "user_info": info}
    
    # --- 5. Final Tool Trigger ---
    mock_lead_capture(info['name'], info['email'], info['platform'])
    return {"messages": [AIMessage(content="Perfect! Your details are captured. We'll be in touch!")], "user_info": info}
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