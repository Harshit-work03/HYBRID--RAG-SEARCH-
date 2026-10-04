from search import dense_search
from bm25_search import build_bm25_index, bm25_search
from fusion import reciprocal_rank_fusion

TOP_K = 3

# Each test has a question and a short phrase that appears in the chunk
# that answers it (matched ignoring capitals).
TESTS = [
    # Questions that reuse the document's wording
    {"question": "What is machine learning a subset of?", "phrase": "subset of AI"},
    {"question": "Which libraries does Python have for data analysis?", "phrase": "Pandas for data analysis"},
    {"question": "What does SQL stand for?", "phrase": "Structured Query Language"},
    {"question": "Which database systems are popular?", "phrase": "PostgreSQL, MySQL, SQLite"},
    {"question": "What do routers do with data packets?", "phrase": "direct data packets"},
    {"question": "How do solar panels produce electricity?", "phrase": "convert sunlight into electricity"},
    {"question": "What activity causes greenhouse gas emissions?", "phrase": "burning fossil fuels"},
    # Questions that paraphrase the document
    {"question": "Which tool can I use to build interactive web apps?", "phrase": "Streamlit for building interactive web applications"},
    {"question": "How much rest should grown-ups get each night?", "phrase": "seven to eight hours of sleep"},
    {"question": "How hot can it get during the warm season?", "phrase": "rise above 35"},
    {"question": "Have people travelled to the Moon?", "phrase": "reached the Moon"},
    {"question": "What happens to the oceans as the planet warms?", "phrase": "rising sea levels"},
    {"question": "Why mix keyword matching with vector search?", "phrase": "improves retrieval accuracy"},
    {"question": "Why do weather predictions matter to people?", "phrase": "prepare for changing conditions"},
    {"question": "Which fields use artificial intelligence?", "phrase": "healthcare, finance, transportation, and education"},
]

bm25, chunks = build_bm25_index()


def found(retrieved_chunks, phrase):
    """Return the 1-based rank of the first chunk containing the phrase, or None."""
    for rank, chunk in enumerate(retrieved_chunks, start=1):
        if phrase.lower() in chunk.lower():
            return rank
    return None


def get_results(question):
    dense_docs = dense_search(question, top_k=TOP_K)["documents"][0]
    sparse = bm25_search(bm25, chunks, question, top_k=TOP_K)
    sparse_docs = [doc for doc, _ in sparse]
    fused = reciprocal_rank_fusion(dense_docs, sparse)
    fused_docs = [doc for doc, _ in fused[:TOP_K]]
    return {"Dense only": dense_docs, "BM25 only": sparse_docs, "Hybrid (RRF)": fused_docs}


stats = {m: {"hit1": 0, "hit3": 0} for m in ["Dense only", "BM25 only", "Hybrid (RRF)"]}

for t in TESTS:
    results = get_results(t["question"])
    print(f"\nQ: {t['question']}")
    for method, docs in results.items():
        rank = found(docs, t["phrase"])
        if rank == 1:
            stats[method]["hit1"] += 1
        if rank is not None:
            stats[method]["hit3"] += 1
        print(f"  {method:14} -> {'rank ' + str(rank) if rank else 'NOT FOUND'}")

n = len(TESTS)
print(f"\n=== RESULTS ({n} questions) ===")
print(f"{'Method':14} | Top-1 | Top-{TOP_K}")
for method, s in stats.items():
    print(f"{method:14} | {s['hit1']}/{n}   | {s['hit3']}/{n}")