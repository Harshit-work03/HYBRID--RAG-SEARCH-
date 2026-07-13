import chromadb
from sentence_transformers import SentenceTransformer
from chunking import chunk_text

# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create a persistent Chroma client — saves data to disk in ./chroma_db folder
client = chromadb.PersistentClient(path="./chroma_db")

# Create (or get, if it already exists) a collection to store our chunks
collection = client.get_or_create_collection("docs")


def build_index(filepath="sample.txt"):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_text(text, chunk_size=100, overlap=20)
    embeddings = model.encode(chunks).tolist()
    ids = [str(i) for i in range(len(chunks))]

    # Store chunks + their embeddings in the database
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids
    )

    print(f"Indexed {len(chunks)} chunks into ChromaDB.")


if __name__ == "__main__":
    build_index()