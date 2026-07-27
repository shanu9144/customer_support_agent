# Customer Support Agent with LangGraph

An intelligent customer support agent built with **LangGraph** and **LangChain** that categorizes queries, analyzes sentiment, and routes them to appropriate handlers — all powered by OpenAI's GPT.

## How It Works

```
Customer Query
      │
      ▼
┌─────────────┐
│  Categorize │  → Technical / Billing / General
└─────┬───────┘
      ▼
┌─────────────────┐
│Analyze Sentiment│  → Positive / Neutral / Negative
└─────┬───────────┘
      ▼
┌─────────────────────────────────────────────┐
│ Route:                                      │
│  • Negative sentiment  → Escalate to human  │
│  • Technical           → Tech support reply │
│  • Billing             → Billing reply      │
│  • General             → General reply      │
└─────────────────────────────────────────────┘
```

## Features

- **Query Categorization** — Classifies into Technical, Billing, or General
- **Sentiment Analysis** — Detects Positive, Neutral, or Negative tone
- **Conditional Routing** — Negative queries get escalated; others get AI-generated responses
- **Gradio UI** — Simple web chat interface for interaction
- **Deployable** — Ready for Hugging Face Spaces, Render, Railway, or any cloud

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/shanu9144/customer_support_agent.git
cd customer_support_agent
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your OpenAI API key

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your-key-here
```

### 4. Run the app

```bash
python app.py
```

Open `http://127.0.0.1:7860` in your browser.

## Deployment

### Hugging Face Spaces (Free)

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space) with SDK = **Gradio**
2. Push this repo to the Space
3. Add `OPENAI_API_KEY` in **Settings → Secrets**

### Render / Railway

1. Connect your GitHub repo
2. Set build command: `pip install -r requirements.txt`
3. Set start command: `python app.py`
4. Add `OPENAI_API_KEY` as an environment variable

## Tech Stack

- [LangGraph](https://github.com/langchain-ai/langgraph) — Graph-based agent orchestration
- [LangChain](https://github.com/langchain-ai/langchain) — LLM framework
- [OpenAI GPT](https://platform.openai.com/) — Language model
- [Gradio](https://gradio.app/) — Web UI

## License

MIT
