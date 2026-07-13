from rank_bm25 import BM25Okapi
from chunking import chunk_text


def build_bm25_index(filepath="sample.txt"):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_text(text, chunk_size=100, overlap=20)

    # BM25 needs tokenized (word-split) text, not raw sentences
    tokenized_chunks = [chunk.lower().split() for chunk in chunks]

    bm25 = BM25Okapi(tokenized_chunks)

    return bm25, chunks


def bm25_search(bm25, chunks, query, top_k=3):
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    # Pair each chunk with its score, then sort highest score first
    ranked = sorted(zip(chunks, scores), key=lambda x: -x[1])

    return ranked[:top_k]


if __name__ == "__main__":
    bm25, chunks = build_bm25_index()

    query = input("Enter your question: ")
    results = bm25_search(bm25, chunks, query)

    for i, (doc, score) in enumerate(results):
        print(f"\n--- Result {i+1} (BM25 score: {score:.4f}) ---")
        print(doc)