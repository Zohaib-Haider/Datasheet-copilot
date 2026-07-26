import requests
from sentence_transformers import SentenceTransformer
import chromadb

from ingest import DOCS_DIR, DB_DIR, COLLECTION_NAME, EMBEDDING_MODEL_NAME

ollama_url = "http://localhost:11434/api/generate"
ollama_model_name = "llama3.2"
TOP_K = 4  # Number of top documents to retrieve

_embedding_model = None
_collection = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}'...")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model

def get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(DB_DIR))
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection

def retrieve(question: str, top_k: int = TOP_K):
    collection = get_collection()
    embedding_model = get_embedding_model()
    querry_embedding = embedding_model.encode(question).tolist()

    results = collection.query(
        query_embeddings=[querry_embedding],
        n_results=top_k
    )
    chunks = []
    for doc, meta, dist in zip(
        results['documents'][0],results['metadatas'][0], results['distances'][0]
    ):
        chunks.append({
            "text": doc,
            "source": meta["source"],
            "page": meta["page"],
            "distance": dist
        })
    return chunks

def build_promt(question: str, chunks: list) -> str:
    context_blocks = []
    for i, c in enumerate(chunks, start=1):
        context_blocks.append(f"[source [i]: {c['source']}, page {c['page']}] {c['text']}")
    context = "\n\n".join(context_blocks)

    prompt = f"""you are a technical assistant answering questions about electronics datasheets. Answer the question using only the context below. if the context dose not contain the answer, say "I couldnt find that in the provided datasheets" - do not guess or use outside knowledge. Site which source mnumber[s] you used.

Context:
{context}

question: {question}
answer:"""
    return prompt
def ask_ollama(prompt: str, model: str = ollama_model_name) -> str:
    response = requests.post(
        ollama_url,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        },
        timeout=120
    )
    response.raise_for_status()
    return response.json()["response"].strip()

def answer_question(question: str, top_k: int = TOP_K):
    chunks = retrieve(question, top_k= top_k)
    if not chunks:
        return "I couldnt find that in the provided datasheets"
    prompt = build_promt(question, chunks)
    answer = ask_ollama(prompt)
    sources= [f"{c['source']} (page {c['page']})" for c in chunks]
    return answer, sources

if __name__ == "__main__":
    question = input("Enter your question: ")
    answer, sources = answer_question(question)
    print(f"Answer: {answer}")
    print(f"Sources: {', '.join(sources)}")
    for s in sources:
        print(f"Source: {s}")
