import chromadb
from sentence_transformers import SentenceTransformer

# Load the same embedding model used for indexing
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to the same database we created in indexing.py
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("docs")


def dense_search(query, top_k=3):
    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )

    return results


if __name__ == "__main__":
    query = input("Enter your question: ")
    results = dense_search(query)

    documents = results["documents"][0]
    distances = results["distances"][0]

    for i, (doc, dist) in enumerate(zip(documents, distances)):
        print(f"\n--- Result {i+1} (distance: {dist:.4f}) ---")
        print(doc)