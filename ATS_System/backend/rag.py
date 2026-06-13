import re
import chromadb
from sentence_transformers import SentenceTransformer
from ats_system import call_mistral

# Setup
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="data/chroma_db")
collection = chroma_client.get_or_create_collection("resumes")

SECTION_HEADERS = ["skills", "experience", "education", "projects", "certifications", "summary"]

# Chunking 
def chunk_resume(text: str, candidate_id: str) -> list[dict]:
    chunks = []
    current_section = "general"
    current_chunk = []

    for line in text.split("\n"):
        lower = line.lower().strip()
        matched = next((h for h in SECTION_HEADERS if h in lower and len(lower) < 30), None)

        if matched:
            if current_chunk:
                chunks.append({"text": "\n".join(current_chunk),
                                "metadata": {"candidate_id": candidate_id, "section": current_section}})
            current_section = matched
            current_chunk = [line]
        else:
            current_chunk.append(line)

    if current_chunk:
        chunks.append({"text": "\n".join(current_chunk),
                        "metadata": {"candidate_id": candidate_id, "section": current_section}})
    return chunks

# Embedding + Storage
def add_chunks(chunks: list[dict]):
    texts = [c["text"] for c in chunks]
    embeddings = embed_model.encode(texts).tolist()
    ids = [f"{c['metadata']['candidate_id']}_{i}" for i, c in enumerate(chunks)]
    metadatas = [c["metadata"] for c in chunks]

    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)

# Retrieval
def query_chunks(query: str, candidate_id: str = None, top_k: int = 5):
    query_embedding = embed_model.encode([query]).tolist()[0]
    where_filter = {"candidate_id": candidate_id} if candidate_id else None

    results = collection.query(query_embeddings=[query_embedding], n_results=top_k, where=where_filter)
    return results["documents"][0] if results["documents"] else []

# Chatbot
CHAT_SYSTEM = """You are an HR assistant. Use ONLY the provided resume context
to answer the question. If the answer isn't in the context, say so."""

def chat_with_resumes(user_query: str, candidate_id: str = None) -> str:
    chunks = query_chunks(user_query, candidate_id=candidate_id, top_k=5)
    context = "\n\n---\n\n".join(chunks)
    prompt = f"CONTEXT:\n{context}\n\nQUESTION:\n{user_query}"
    return call_mistral(CHAT_SYSTEM, prompt, json_mode=False)