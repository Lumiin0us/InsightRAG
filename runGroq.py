import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./vectordb")
collection = client.get_collection("insightrag")
groqClient = Groq(api_key="#")


def retrieve(query: str, top_k: int = 5) -> list[dict]:
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


def answer(query: str) -> str:
    chunks = retrieve(query)

    context = "\n\n".join([
        f"[{c['source']}] {c['title']}\n{c['text']}"
        for c in chunks
    ])

    print(f"\nSources used:")
    for c in chunks:
        print(f"  - {c['title']} ({c['source']}) {c['url']}")

    response = groqClient.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are InsightRAG, an AI assistant for ML and AI documentation. Answer questions using only the provided context. If the answer is not in the context, say you don't know."
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}"
            }
        ]
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    print("InsightRAG ready! Type 'quit' to exit.\n")
    while True:
        query = input("You: ").strip()
        if query.lower() == "quit":
            break
        if not query:
            continue

        response = answer(query)
        print(f"\nAssistant: {response}\n")