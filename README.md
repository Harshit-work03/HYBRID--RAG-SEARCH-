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

## Limitations and future work

- Indexes one text file at a time.
- No quantitative evaluation yet. Planned: compare dense-only, BM25-only and hybrid retrieval on a small set of test questions.
- Re-running `indexing.py` without deleting `chroma_db` can conflict with existing chunk IDs. Switching to `collection.upsert` would fix this.