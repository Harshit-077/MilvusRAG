# Milvus RAG: Chat with Your Documents using Llama 3.1 & Groq

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline. It allows you to "chat" with your local documents (PDFs and TXTs) by indexing them into a **Milvus** vector database and using **Llama 3.1** via the Groq API to generate grounded, context-aware answers.

## 🚀 Features
- **Document Ingestion**: Automatically processes `.pdf` and `.txt` files from a local `data/` directory.
- **Text Cleaning**: Custom regex-based cleaning to fix PDF artifacts (like merged words).
- **Vector Search**: Utilizes `MilvusClient` (Lite) for high-performance similarity search.
- **Open-Source Embeddings**: Powered by `sentence-transformers/all-MiniLM-L6-v2`.
- **Fast Inference**: Uses Groq's Llama 3.1 8B model for sub-second responses.
- **Smart Chunking**: Splits large documents into overlapping chunks for accurate retrieval.
- **Metadata Tracking**: Each chunk retains its source file for explainable answers.
---

## 🛠️ Tech Stack
- **Vector Database**: Milvus (Lite)
- **LLM**: Llama-3.1-8b-instant (via Groq API)
- **Embedding Model**: Sentence-Transformers (384-dimensional)
- **Document Parsing**: PyPDF
- **Environment**: Python 3.10+

---

## 📋 Prerequisites
Before running the script, ensure you have:
1. A **Groq API Key** (Get one at [console.groq.com](https://console.groq.com/)).
2. A folder named `data/` in the root directory containing your `.pdf` or `.txt` files.

---

## 🧠 How It Works

1. Documents are loaded from the `data/` folder
2. Text is cleaned and split into chunks
3. Each chunk is converted into embeddings
4. Embeddings are stored in Milvus
5. On query:
   - Query is embedded
   - Top-k similar chunks are retrieved
   - Context is sent to Llama 3.1 (via Groq)
   - Answer is generated


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
```commandline
GROQ_API_KEY=your_groq_api_key_here
```
## 🏃 Usage
#### 1. Place your documents in the data/ folder.

#### 2. Run the main script:
```bash
python main.py
```
#### 3. Interactive Chat:

Ask questions about your documents in the terminal.

Type exit or quit to end the session.

## 🚀 Future Improvements
- Web UI (Streamlit / React)
- Support for more file types (DOCX, HTML)
- Hybrid search (keyword + vector)
- Persistent collections (no re-indexing on every run)
   