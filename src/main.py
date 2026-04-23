from agent import app
from langchain_core.messages import HumanMessage

def run_chat():
    # thread_id is key for the 'memory' requirement [cite: 90, 91]
    config = {"configurable": {"thread_id": "session_1"}}
    
    print("AutoStream Agent Active. Type 'exit' to stop.")
    
    while True:
        user_input = input("User: ")
        if user_input.lower() == "exit": break
        
        # We only pass the new message; LangGraph handles the rest
        try:
            for event in app.stream({"messages": [HumanMessage(content=user_input)]}, config=config):
                for value in event.values():
                    if "messages" in value:
                        print(f"Agent: {value['messages'][-1].content}")
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    run_chat()