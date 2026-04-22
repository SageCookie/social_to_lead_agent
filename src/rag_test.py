import os

def load_knowledge_base():
    file_path = os.path.join("data", "knowledge_base.md")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            print("Successfully loaded knowledge base!")
            return content
    except FileNotFoundError:
        print("Error: knowledge_base.md not found in the data/ directory.")
        return None

if __name__ == "__main__":
    data = load_knowledge_base()
    if data:
        print("\n--- Preview ---")
        print(data[:150] + "...") # Show just the beginning