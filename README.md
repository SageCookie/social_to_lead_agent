# Project: Social-to-Lead Agentic Workflow

## Project Overview
This project involves the development of a sophisticated Conversational AI Agent designed for AutoStream, a fictional SaaS company that provides automated video editing tools for content creators.

The system is built for ServiceHive as part of their Inflx platform, which aims to convert social media conversations into qualified business leads.

Unlike standard chatbots, this agent is an agentic workflow capable of:
- Intent Identification: Accurately classifying user messages into greetings, product/pricing inquiries, or high-intent leads.
- RAG-Powered Knowledge Retrieval: Answering technical product questions and policy inquiries using a local Markdown-based knowledge base.
- Lead Qualification: Identifying users ready to sign up and interactively collecting necessary details.
- Automated Tool Execution: Triggering a backend lead capture function only after validating that all required user information (Name, Email, and Platform) has been provided.

The core of this project is built using LangGraph to manage complex, cyclical conversation states and maintain memory across multiple turns, ensuring a seamless transition from general inquiry to lead capture.

## Architecture & State Management
For this project, I chose LangGraph over standard linear chains because the "Social-to-Lead" workflow is inherently cyclical.

Users often shift between asking about pricing and expressing interest in signing up.

LangGraph allows the agent to maintain a persistent state across the mandatory 5-6 conversation turns by utilizing a TypedDict state and a MemorySaver checkpointer.

The architecture consists of four primary nodes:
1. Intent Classifier: Categorizes input into greetings, inquiries, or leads.
2. RAG Retriever: Pulls pricing and policy data from a local Markdown knowledge base.
3. Lead Collector: A stateful node that interactively requests missing user details (Name, Email, Platform).
4. Tool Executor: Triggers the mock_lead_capture function only when the state is complete.

This modular design ensures that the agent logic remains "clean" and that tools are never triggered prematurely.

## How to Run Locally
Follow these steps to set up the development environment and run the AutoStream agent on your local machine.

### 1. Prerequisites
- Python 3.10+
- A Google Gemini API Key (Obtainable from Google AI Studio)

### 2. Installation & Setup
Clone the repository and navigate into the project directory:

```bash
git clone <your-repository-link>
cd social-to-lead-agent
```

Create a virtual environment and activate it:

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### 3. Environment Configuration
Create a .env file in the root directory and add your credentials:

```env
# .env
GOOGLE_API_KEY=your_actual_api_key_here
LLM_MODEL=gemini-3.1-flash-lite-preview
```

### 4. Running the Agent
Start the interactive terminal session by running the main script:

```bash
python src/main.py
```

## WhatsApp Deployment via Webhooks
To integrate this agent with WhatsApp, I would utilize the Meta Business API or Twilio WhatsApp API.

- Webhook Configuration: I would deploy the Python backend using a framework like FastAPI and expose a POST endpoint (e.g., /whatsapp/webhook).
- Request Handling: When a user sends a message, the WhatsApp provider sends a JSON payload to the webhook. The backend extracts the text and the user's phone number.
- Session Persistence: The phone number would serve as the thread_id in LangGraph, ensuring the agent remembers the specific user's details across multiple days or turns.
- Response: The agent's output is then sent back to the user via a triggered API call to the WhatsApp provider.