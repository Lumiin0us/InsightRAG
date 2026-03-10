import json
import os
from sentence_transformers import SentenceTransformer
import chromadb

def load_all_docs(dataset_dir: str) -> list[dict]:
    all_docs = []
    for filename in os.listdir(dataset_dir):
        if filename.endswith(".json"):
            filepath = os.path.join(dataset_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                docs = json.load(f)
                for doc in docs:
                    doc["source"] = filename.replace(".json", "")
                all_docs.extend(docs)
    print(f"Loaded {len(all_docs)} docs from {dataset_dir}")
    return all_docs


def chunk_huggingface(doc: dict) -> list[dict]:
    chunks = []
    for section in doc.get("sections", []):
        text = f"{section['heading']}\n{section['content']}"
        if text.strip():
            chunks.append({
                "text": text,
                "url": doc["url"],
                "title": doc["title"],
                "source": "huggingface"
            })
    return chunks


def chunk_anthropic(doc: dict, chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    words = doc["text"].split()
    chunks = []
    i = 0
    while i < len(words):
        chunk_text = " ".join(words[i:i + chunk_size])
        chunks.append({
            "text": chunk_text,
            "url": doc["url"],
            "title": doc.get("heading", ""),
            "source": "anthropic"
        })
        i += chunk_size - overlap
    return chunks


def chunk_doc(doc: dict) -> list[dict]:
    source = doc.get("source", "")
    if source == "huggingface_docs":
        return chunk_huggingface(doc)
    elif source == "anthropic_docs":
        return chunk_anthropic(doc)
    else:
        print(f"Unknown source: {source}, skipping")
        return []


model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_chunks(chunks: list[str]):
    return model.encode(chunks, show_progress_bar=True)


client = chromadb.PersistentClient(path="./vectordb")
collection = client.get_or_create_collection("insightrag")


def store_docs(docs: list[dict]):
    for doc in docs:
        chunks = chunk_doc(doc)  # routes to correct chunker
        if not chunks:
            continue

        texts = [c["text"] for c in chunks]
        embeddings = embed_chunks(texts)

        collection.add(
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=[{
                "url": c["url"],
                "title": c["title"],
                "source": c["source"]
            } for c in chunks],
            ids=[f"{doc['url']}_{i}" for i in range(len(chunks))]
        )
        print(f"Stored {len(chunks)} chunks from {doc['url']}")


docs = load_all_docs("dataset/")
store_docs(docs)