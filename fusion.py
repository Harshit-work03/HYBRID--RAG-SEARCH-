from search import dense_search
from bm25_search import build_bm25_index, bm25_search


def reciprocal_rank_fusion(dense_results, sparse_results, k=60):
    """
    Combines two ranked lists into one, using rank position (not raw scores).
    k is a constant that softens the impact of rank differences (60 is standard).
    """
    scores = {}

    # dense_results: list of document texts, already ranked best -> worst
    for rank, doc in enumerate(dense_results):
        scores[doc] = scores.get(doc, 0) + 1 / (k + rank + 1)

    # sparse_results: list of (doc, bm25_score) tuples, ranked best -> worst
    for rank, (doc, _) in enumerate(sparse_results):
        scores[doc] = scores.get(doc, 0) + 1 / (k + rank + 1)

    # Sort combined scores highest first
    fused = sorted(scores.items(), key=lambda x: -x[1])
    return fused


if __name__ == "__main__":
    query = input("Enter your question: ")

    # Get dense search results (just the text, not the distances)
    dense_raw = dense_search(query, top_k=2)
    dense_docs = dense_raw["documents"][0]

    # Get BM25 results
    bm25, chunks = build_bm25_index()
    sparse_results = bm25_search(bm25, chunks, query, top_k=2)

    # Fuse them
    fused_results = reciprocal_rank_fusion(dense_docs, sparse_results)

    print("\n=== FUSED RESULTS (best of both) ===")
    for i, (doc, score) in enumerate(fused_results):
        print(f"\n--- Result {i+1} (fused score: {score:.4f}) ---")
        print(doc)