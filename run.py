import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./vectordb")
collection = client.get_collection("insightrag")


def retrieve(query: str, top_k: int = 2) -> list[dict]:
    query_embedding = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )

    chunks = []
    for i, doc in enumerate(results["documents"][0]):
        chunks.append({
            "text": doc,
            "url": results["metadatas"][0][i]["url"],
            "title": results["metadatas"][0][i]["title"],
            "source": results["metadatas"][0][i]["source"]
        })
    return chunks


if __name__ == "__main__":
    print("🔍 InsightRAG ready! Type 'quit' to exit.\n")
    while True:
        query = input("You: ").strip()
        if query.lower() == "quit":
            break
        if not query:
            continue

        chunks = retrieve(query)
        print(f"\n Top {len(chunks)} results:\n")
        for i, chunk in enumerate(chunks):
            print(f"[{i+1}] {chunk['title']} ({chunk['source']})")
            print(f"{chunk['url']}")
            print(f"{chunk['text'][:200]}...")
            print()