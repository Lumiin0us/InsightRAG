# InsightRAG

This project implements a Retrieval-Augmented Generation (RAG) system built on official AI/ML documentation from multiple sources, including HuggingFace and Anthropic. It involves web scraping pipelines for each source, a sentence tokenizer, vector database (ChromaDB), and Groq's cloud API to run a Large Language Model for answer generation.
After scraping, several preprocessing techniques were applied to clean the data, such as removing UI noise (navigation links, button labels, page feedback prompts), stripping SVG and script tags, and normalizing whitespace. Data is then sent through respective chunking strategies suited to its format, followed by embedding and vector storage.
Followed by a Streamlit interface for interactive querying.

---

## Data Sources

1. HuggingFace Docs
2. Anthropic Docs

## Scraping & Retrieval Process

Each source has its own scraping script due to differences in page structure:

HuggingFace — JavaScript-based via Playwright. The main docs page is scraped for category links, then each category page's sidebar nav is followed to scrape all subpages.
Anthropic — Static HTML via requests. The sidebar nav is extracted and each linked page is scraped and cleaned.

When a user asks a question:
1. The query is converted into an embedding using all-MiniLM-L6-v2
2. ChromaDB retrieves the most relevant chunks via cosine similarity
3. Retrieved chunks are assembled into a prompt with source metadata
4. The prompt is sent to the LLM


---

## LLM Generation

LLM Generation
A cloud LLM running through Groq's free API generates the final answer using llama-3.3-70b-versatile.
The model is instructed to:
  - Answer only using retrieved context
  - Avoid hallucinating information
  - Cite document sources where possible

---

## Project Structure

InsightRAG/
├── dataset/
│   ├── huggingface_docs.json
│   └── anthropic_docs.json
├── scrapingScripts/
│   ├── huggingFace.py
│   └── anthropic.py
├── vectordb/
├── main.py
├── run.py
├── runGroq.py
├── runGroqStreamlit.py
└── README.md
  
---

## Workflow

User Query  
   ↓  
Embedding Model  
   ↓  
ChromaDB Vector Search  
   ↓  
Top-K Context  
   ↓  
Groq LLM (llama-3.3-70b-versatile)
   ↓  
Answer  

---

## Future Improvements

- RAG evaluation metrics (faithfulness / relevance)
- Including PDFs
- Cloud deployment
