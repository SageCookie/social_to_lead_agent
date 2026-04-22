from agent import app
from langchain_core.messages import HumanMessage

def run_chat():
    state = {"messages": [], "user_info": {"name": None, "email": None, "platform": None}}
    
    print("AutoStream Agent Active (Type 'exit' to quit)")
    while True:
        user_input = input("User: ")
        if user_input.lower() == "exit": break
        
        # Stream the results from the graph
        events = app.stream({"messages": [HumanMessage(content=user_input)]}, config={"configurable": {"thread_id": "1"}})
        for event in events:
            for value in event.values():
                if "messages" in value:
                    print(f"Agent: {value['messages'][-1].content}")

if __name__ == "__main__":
    run_chat()