# Hybrid RAG Search

A retrieval-augmented generation (RAG) system that combines **dense semantic search** with **BM25 keyword search**, merges the results with **Reciprocal Rank Fusion**, and generates answers with a **locally hosted Llama 3.2** model (via Ollama). Includes a Streamlit chat interface with conversation memory.


## Why hybrid search?

Dense embeddings find text that is similar in *meaning*, but can miss exact terms. BM25 finds exact *keyword* matches, but misses paraphrases. Combining both keeps results accurate even when a query does not match the document wording.

## How it works

```
Question
   │
   ├─► Query rewriting (uses last 4 chat messages to make follow-ups standalone)
   │
   ├─► Dense search   (all-MiniLM-L6-v2 embeddings in ChromaDB)  ─┐
   │                                                              ├─► Reciprocal Rank Fusion ─► Top 3 chunks
   └─► BM25 search    (rank-bm25 over the same chunks)           ─┘
                                                                          │
                                                                          ▼
                                              Llama 3.2 (Ollama) answers using only those chunks
```

1. **Chunking** (`chunking.py`): the document is split into chunks of 100 words with a 20-word overlap.
2. **Indexing** (`indexing.py`): chunks are embedded with `all-MiniLM-L6-v2` and stored in a persistent ChromaDB collection.
3. **Dense search** (`search.py`) and **BM25 search** (`bm25_search.py`) each return a ranked list of chunks.
4. **Fusion** (`fusion.py`): Reciprocal Rank Fusion (k = 60) merges both ranked lists using rank position rather than raw scores, so the two scoring scales do not need to be compared.
5. **Generation** (`generate.py`): Llama 3.2 is prompted to answer using only the retrieved context, which reduces made-up answers.
6. **Query rewriting** (`generate.py`): follow-up questions are rewritten into standalone questions using recent chat history.

## Project structure

| File | Purpose |
|---|---|
| `app.py` | Streamlit chat interface |
| `main.py` | Full pipeline (`run_pipeline`) and command-line version |
| `chunking.py` | Splits text into overlapping word chunks |
| `indexing.py` | Builds the ChromaDB vector index from `sample.txt` |
| `search.py` | Dense (semantic) search |
| `bm25_search.py` | BM25 keyword search |
| `fusion.py` | Reciprocal Rank Fusion |
| `generate.py` | Answer generation and query rewriting |
| `sample.txt` | Example document to search |

## Setup

Developed with Python 3.14.

**1. Clone the repository**
```
git clone https://github.com/Harshit-work03/HYBRID--RAG-SEARCH-.git
cd HYBRID--RAG-SEARCH-
```

**2. Create a virtual environment and install dependencies**
```
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS / Linux
pip install -r requirements.txt
```

**3. Install Ollama and download the model**

Install Ollama from https://ollama.com, then run:
```
ollama pull llama3.2
```

**4. Build the index (run once)**
```
python indexing.py
```
This reads `sample.txt` and creates the `chroma_db` folder. To rebuild from scratch, delete `chroma_db` and run it again.

## Usage

**Chat interface**
```
streamlit run app.py
```

**Command line**
```
python main.py
```

To search your own document, replace `sample.txt` with your text file, delete the `chroma_db` folder, and run `python indexing.py` again.

## Tech stack

Python, Sentence-Transformers, ChromaDB, rank-bm25, Ollama (Llama 3.2), Streamlit

## Evaluation

I compared dense-only, BM25-only and hybrid (RRF) retrieval on 15 hand-written questions over `sample.txt` (7 chunks). A question counts as a hit if the chunk containing the answer is retrieved. Run it with `python eval.py`.

| Method | Top-1 | Top-3 |
|---|---|---|
| Dense only | 11/15 | 14/15 |
| BM25 only | 13/15 | 13/15 |
| Hybrid (RRF) | 12/15 | 14/15 |

**Takeaways:** BM25 was strongest at Top-1 on this corpus, since many questions reused the document's exact wording. Hybrid matched the best Top-3 score and was never the weakest method, so it is the more robust choice when query style is unknown. These results come from a very small corpus and test set, so they are indicative rather than conclusive.

## Limitations and future work

- Indexes one text file at a time.
- No quantitative evaluation yet. Planned: compare dense-only, BM25-only and hybrid retrieval on a small set of test questions.
- Re-running `indexing.py` without deleting `chroma_db` can conflict with existing chunk IDs. Switching to `collection.upsert` would fix this.