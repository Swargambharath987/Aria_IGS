from rag.query_engine import build_chat_engine, is_slurm_related

print("Loading RAG engine (first load takes ~10 seconds)...\n")
engine = build_chat_engine()
print("Engine ready. Type your question or 'quit' to exit.\n")
print("-" * 60)

while True:
    query = input("\nYou: ").strip()
    if not query:
        continue
    if query.lower() in ("quit", "exit"):
        break
    if not is_slurm_related(query):
        print("Bot: I can only help with IGS grid and Slurm-related questions.")
        continue
    response = engine.chat(query)
    print(f"\nBot: {response}")
