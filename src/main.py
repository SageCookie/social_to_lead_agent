from agent import app
from langchain_core.messages import HumanMessage

def run_chat():
    # 1. Define the config (Crucial for the 5-6 turns memory requirement)
    config = {"configurable": {"thread_id": "session_1"}}
    
    print("AutoStream Agent Active. Type 'exit' to stop.")
    
    while True:
        user_input = input("User: ")
        if user_input.lower() == "exit": 
            break
        
        try:
            # We pass the message and the config for session persistence
            for event in app.stream({"messages": [HumanMessage(content=user_input)]}, config=config):
                for value in event.values():
                    if "messages" in value:
                        # --- 2026 CLEANUP LOGIC ---
                        # Extracts only text and ignores metadata/signatures
                        raw_content = value['messages'][-1].content
                        
                        if isinstance(raw_content, list):
                            clean_text = "".join([
                                part.get("text", "") 
                                for part in raw_content 
                                if isinstance(part, dict) and "text" in part
                            ])
                        else:
                            clean_text = raw_content
                        
                        if clean_text.strip():
                            print(f"Agent: {clean_text}")
                            
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    run_chat()