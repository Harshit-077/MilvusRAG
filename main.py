import os
import re
from groq import Groq
from pypdf import PdfReader
from dotenv import load_dotenv
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

load_dotenv()

client = MilvusClient(uri="milvus_demo.db")
if client.has_collection(collection_name="data"):
    client.drop_collection(collection_name="data")
client.create_collection(
    collection_name="data",
    dimension=384,
)

def clean_text(text):
    text = text.replace("SFJBS", "")
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text, chunk_size=200, chunk_overlap = 50):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size-chunk_overlap):
        chunks.append(" ".join(words[i:i+chunk_size]))

    return chunks

def load_documents(folder_path="data"):
    documents = []

    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)

        # TXT
        if file.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                documents.append({
                    "text": f.read(),
                    "source": file
                })

        # PDF
        elif file.endswith(".pdf"):
            reader = PdfReader(file_path)
            text = ""

            for page in reader.pages:
                text += page.extract_text() or ""

            documents.append({
                "text": text,
                "source": file
            })

    return documents

raw_docs = load_documents("data")

docs = []
for doc in raw_docs:
    cleaned = clean_text(doc["text"])
    chunks = chunk_text(cleaned)

    for chunk in chunks:
        docs.append({
            "text": chunk,
            "source": doc["source"]
        })

model = SentenceTransformer("all-MiniLM-L6-v2")
texts = [d["text"] for d in docs]
vectors = model.encode(texts)

data = [
    {"id": i, "vector": vectors[i].tolist(), "text": docs[i]["text"], "source": docs[i]["source"]}
    for i in range(len(vectors))
]

# print("Data has", len(data), "entities, each with fields: ", data[0].keys())
# print("Vector dim:", len(data[0]["vector"]))

res = client.insert(collection_name="test", data = data)
# print(res)
client.load_collection("test")

def retrieve(query_text, top_k=2, alpha=0.7):
    # Semantic Search
    query_vector = model.encode([query_text]).tolist()
    results = client.search(
        collection_name="test",
        data=query_vector,
        limit=top_k*5,
        output_fields=["text", "source"]
    )
    # Keyword Search + Semantic Search
    scores = []
    query_words = set(query_text.lower().split())

    for hit in results[0]:
        text = hit["entity"]["text"]
        source = hit["entity"]["source"]
        sem_sc = hit["distance"]

        doc_words = set(text.lower().split())
        kw_score = len(query_words & doc_words) / max(1,len(query_words))

        # Combine both
        final_score = alpha * (1-sem_sc) + (1-alpha) * kw_score
        scores.append({
            "text": text,
            "source": source,
            "score": final_score
        })

    # Sorting
    scores.sort(key=lambda x: x["score"], reverse=True)
    return scores[:top_k]

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_answer(query, context):
    prompt = f"""
You are a helpful assistant.
Answer ONLY using the context below.
If the answer is not in the context, say "I don't know".
Always mention the source file name used for your answer.

Context:
{context}

Question:
{query}
"""

    response = groq_client.chat.completions.create(
        model = "llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def chat(query):
    context_docs = retrieve(query)
    print("Retrieved: ")
    for doc in context_docs:
        print(f"[{doc['source']}] {doc['text'][:100]}...")
    if not context_docs:
        return "I don't know."

    context = "\n".join([f"[{doc['source']}] {doc['text']}" for doc in context_docs])
    answer = generate_answer(query, context)
    return answer

if __name__ == "__main__":
    while True:
        q = input("Ask: ")
        if q.lower() in ["exit", "quit"]:
            break
        print(chat(q))



