import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq

# page config
st.set_page_config(
    page_title="InsightRAG",
    page_icon="~",
    layout="centered"
)

# cache so models so they don't have to reload
@st.cache_resource
def load_models():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path="./vectordb")
    collection = client.get_collection("insightrag")
    groq_client = Groq(api_key="#")
    return model, collection, groq_client

model, collection, groq_client = load_models()


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


def answer(query: str) -> tuple[str, list[dict]]:
    chunks = retrieve(query)
    context = "\n\n".join([
        f"[{c['source']}] {c['title']}\n{c['text']}"
        for c in chunks
    ])
    response = groq_client.chat.completions.create(
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
    return response.choices[0].message.content, chunks

st.title("InsightRAG")
st.caption("AI assistant powered by HuggingFace & Anthropic docs")

# chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# displaying chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.markdown(f"**{source['title']}** `{source['source']}`")
                    st.markdown(f"{source['url']}")
                    st.divider()

if query := st.chat_input("Ask anything about HuggingFace or Anthropic..."):

    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response, chunks = answer(query)
        st.markdown(response)
        with st.expander("Sources"):
            for chunk in chunks:
                st.markdown(f"**{chunk['title']}** `{chunk['source']}`")
                st.markdown(f"{chunk['url']}")
                st.divider()

    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "sources": chunks
    })