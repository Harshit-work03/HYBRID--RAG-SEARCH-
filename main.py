from search import dense_search
from bm25_search import build_bm25_index, bm25_search
from fusion import reciprocal_rank_fusion
from generate import generate_answer


def run_pipeline(query, top_k=3):
    """
    Runs the full hybrid RAG pipeline:
    1. Dense (semantic) search
    2. Sparse (keyword/BM25) search
    3. Fuse both result sets using Reciprocal Rank Fusion
    4. Take the top chunks after fusion
    5. Generate a final answer using an LLM, grounded in those chunks
    """

    # Step 1: Dense search — find chunks similar in MEANING to the query
    dense_raw = dense_search(query, top_k=top_k)
    dense_docs = dense_raw["documents"][0]

    # Step 2: Sparse search — find chunks with matching KEYWORDS
    bm25, chunks = build_bm25_index()
    sparse_results = bm25_search(bm25, chunks, query, top_k=top_k)

    # Step 3: Combine both ranked lists into one fused ranking
    fused_results = reciprocal_rank_fusion(dense_docs, sparse_results)

    # Step 4: Keep only the top N chunks after fusion
    top_chunks = [doc for doc, score in fused_results[:top_k]]

    # Step 5: Pass those chunks + the query to the LLM to generate a real answer
    answer = generate_answer(query, top_chunks)

    return answer, top_chunks


if __name__ == "__main__":
    query = input("Enter your question: ")
    answer, used_chunks = run_pipeline(query)

    print("\n=== RETRIEVED CONTEXT ===")
    for i, chunk in enumerate(used_chunks):
        print(f"\n--- Chunk {i+1} ---")
        print(chunk)

    print("\n=== FINAL ANSWER ===")
    print(answer)