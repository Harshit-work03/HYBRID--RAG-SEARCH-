def chunk_text(text, chunk_size=100, overlap=20):
    
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        if i + chunk_size >= len(words):
            break
        i += chunk_size - overlap
    return chunks


if __name__ == "__main__":
    with open("sample.txt", "r", encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_text(text, chunk_size=100, overlap=20)

    print(f"Total chunks: {len(chunks)}\n")
    for idx, c in enumerate(chunks):
        print(f"--- Chunk {idx} ---")
        print(c)
        print()