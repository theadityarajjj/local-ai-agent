"""
Fully local, free agent: Ollama (Llama3.2:3b) + Mem0 (memory)

Prereqs:
    ollama pull llama3.2:3b
    ollama pull nomic-embed-text
    pip install mem0ai --break-system-packages
    pip install ollama --break-system-packages
"""

import warnings
warnings.filterwarnings("ignore", message="Payload indexes have no effect in the local Qdrant.*")

import ollama
from mem0 import Memory

USER_ID = "aditya"

# Point Mem0's own internal LLM (used for fact-extraction) at Ollama/Llama3
config = {
    "llm": {
        "provider": "ollama",
        "config": {
            "model": "llama3.2:3b",
        },
    },
    "embedder": {
        "provider": "ollama",
        "config": {
            "model": "nomic-embed-text",
            "embedding_dims": 768,
        },
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "local_agent_memory",
            "embedding_model_dims": 768,
        },
    },
}

memory = Memory.from_config(config)


def run_agent(user_message: str) -> str:
    # 1. Pull relevant memory
    past = memory.search(query=user_message, filters={"user_id": USER_ID}, limit=5)
    memory_context = "\n".join(m["memory"] for m in past.get("results", []))

    system_prompt = f"""You are a helpful assistant with no internet access.
Answer only from what The User has already mentioned

Relevant memory about the user:
{memory_context}
"""

    # 2. Call local Llama3.2:3b via Ollama
    response = ollama.chat(
        model="llama3.2:3b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )

    final_text = response["message"]["content"]

    # 3. Store this interaction back into memory
    memory.add(
        [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": final_text},
        ],
        user_id=USER_ID,
    )

    return final_text


if __name__ == "__main__":
    print("Local agent ready. Type 'quit' to exit.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break
        reply = run_agent(user_input)
        print(f"\nAgent: {reply}\n")