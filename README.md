# Local AI Agent (Ollama + Mem0)

A fully local, fully free AI agent that runs entirely on your own machine —
no API keys, no cloud calls, no billing. It uses a local LLM to hold a
conversation and a local memory system so it remembers facts about you
across sessions.

## How it works

- **Ollama** runs the LLM locally on your machine.
- **Llama 3.2 (3B)** is the model doing the actual reasoning/chatting.
- **nomic-embed-text** is a separate, small model used only to convert text
  into embeddings (vectors) — this is what powers memory search.
- **Mem0** stores and retrieves memories using those embeddings, so the
  agent can recall things you've told it in past conversations.

Every turn follows this loop:

```
user message → search memory → build prompt with relevant memories →
ask local LLM → store new memory → reply
```

No internet access is used at any point — everything runs offline once
the models are downloaded.

## Prerequisites

- Python 3.9+
- [Ollama](https://ollama.com) installed
- Pull the two required models:
  ```bash
  ollama pull llama3.2:3b
  ollama pull nomic-embed-text
  ```
- Install Python dependencies:
  ```bash
  pip install mem0ai ollama --break-system-packages
  ```

## Running it

```bash
python local_agent.py
```

Type a message and press Enter. Type `quit` to exit.

Memory persists on disk between runs — close the program and reopen it
later, and it will still remember what you told it.

## Notes

- Everything runs on CPU unless you have a GPU Ollama can use, so replies
  can be slow, especially on laptops without a dedicated GPU.
- `llama3.2:3b` was chosen over the full `llama3` (8B) specifically for
  speed on modest hardware — swap the model name in `local_agent.py` if
  you have better hardware and want stronger reasoning.
- This is a memory-only agent — it has no web search or tool use. It can
  only answer from its own general knowledge and what it remembers about
  you.