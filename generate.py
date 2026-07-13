import ollama


def generate_answer(query, context_chunks):
    context = "\n\n".join(context_chunks)
    prompt = f"""Answer the question using only the context below.

Context:
{context}

Question: {query}
Answer:"""

    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]


def rewrite_query(current_query, chat_history):
    if not chat_history:
        return current_query

    history_text = ""
    for msg in chat_history[-4:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"

    prompt = f"""Given this conversation history and a new user message, rewrite the new message into a complete, standalone question. If it's already standalone, return it unchanged. Only output the rewritten question, nothing else.

Conversation history:
{history_text}

New message: {current_query}

Standalone question:"""

    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"].strip()


if __name__ == "__main__":
    query = "How can reading books improve a person's skills?"
    context_chunks = [
        "Reading books remains one of the most effective ways to develop critical thinking and creativity.",
        "Even spending just twenty minutes a day reading can improve vocabulary, concentration, and overall understanding of complex topics."
    ]

    answer = generate_answer(query, context_chunks)
    print("=== ANSWER ===")
    print(answer)