# Milvus RAG: Chat with Your Documents using Llama 3.1 & Groq

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline. It allows you to "chat" with your local documents (PDFs and TXTs) by indexing them into a **Milvus** vector database and using **Llama 3.1** via the Groq API to generate grounded, context-aware answers.

Now with **Streamlit UI** for easy web-based interaction and **hybrid search** combining semantic embeddings with keyword search for more accurate results.

---

## 🚀 Features
- **Document Ingestion**: Automatically processes `.pdf` and `.txt` files from a local `data/` directory.
- **Text Cleaning**: Custom regex-based cleaning to fix PDF artifacts (like merged words).
- **Vector & Hybrid Search**:
  - **Semantic Search**: Embedding-based retrieval for contextual relevance.
  - **Keyword Search**: Exact term matching using sparse indices.
  - **Hybrid Search**: Combines both to improve accuracy and recall.
- **Open-Source Embeddings**: Powered by `sentence-transformers/all-MiniLM-L6-v2`.
- **Fast Inference**: Uses Groq's Llama 3.1 8B model for sub-second responses.
- **Smart Chunking**: Splits large documents into overlapping chunks for accurate retrieval.
- **Metadata Tracking**: Each chunk retains its source file for explainable answers.
- **Streamlit Web UI**: Interactively chat with your documents in a browser.

---

## 🛠️ Tech Stack
- **Vector Database**: Milvus (Lite)
- **LLM**: Llama-3.1-8b-instant (via Groq API)
- **Embedding Model**: Sentence-Transformers (384-dimensional)
- **Document Parsing**: PyPDF
- **Web Interface**: Streamlit
- **Environment**: Python 3.10+

---

## 📋 Prerequisites
Before running the scripts, ensure you have:

1. A **Groq API Key** ([console.groq.com](https://console.groq.com/))  
2. A folder named `data/` in the root directory containing your `.pdf` or `.txt` files  

---

## 🧠 How It Works

1. Documents are loaded from the `data/` folder  
2. Text is cleaned and split into chunks  
3. Each chunk is converted into embeddings  
4. Embeddings are stored in Milvus, with optional keyword indices for hybrid search  
5. On query:
   - Query is embedded
   - Top-k results retrieved via **hybrid search**
   - Context is sent to Llama 3.1 (via Groq)
   - Answer is generated

**Hybrid Search Details:**  
Combines dense embeddings (semantic meaning) with sparse keyword matches to return the most relevant results. This ensures both conceptual understanding and exact term matching, improving retrieval for complex or precise queries.

---

## ⚙️ Setup & Installation

### 1. Clone the repository:
```bash
git clone https://github.com/Harshit-077/MilvusRAG.git
cd MilvusRAG
```
### 2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```
### 3. Install dependencies:
```bash
pip install -r requirements.txt
```
### 4. Configure Environment Variables:
Create a .env file in the root directory and add your key:
```bash
GROQ_API_KEY=your_groq_api_key_here
HF_TOKEN=your_huggingface_api_token
```
---
### Terminal Chat
Place your documents in the data/ folder.
#### Run:
```bash
python main.py
```
Ask questions in the terminal. 

Type exit or quit to end the session.


### Streamlit Web UI
Start the web interface:
```bash
streamlit run app.py
```
Open your browser (usually http://localhost:8501)

Paste your question, click Submit, and view answers along with retrieved context.

---

## 🚀 Future Improvements
Support for more file types (DOCX, HTML)

Persistent collections (no re-indexing on every run)

Advanced ranking strategies for hybrid search

React or other web UI frameworks

---


Made with ❤️ by **Harshit**  

This project is a small labor of love for exploring AI, RAG, and document intelligence.  
I hope it helps you chat with your documents as easily as I enjoy building it!  

Feel free to ⭐ the repo if it inspires you.