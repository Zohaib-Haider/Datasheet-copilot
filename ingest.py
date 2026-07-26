import re
import sys
from pathlib import Path

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb

DOCS_DIR = Path(__file__).parent / "docs"
DB_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "datasheets"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

chunk_size = 1200  # Number of characters per chunk
chunk_overlap = 200  # Number of overlapping characters between chunks


def extract_pages(pdf_path: Path):
    reader = PdfReader(str(pdf_path))
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        yield page_num, text

def clean_text(text: str) -> str:
    # Remove multiple spaces and newlines
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text: str, chunk_size: int=chunk_size, chunk_overlap: int=chunk_overlap):
    if len(text) <= chunk_size:
        return [text] if text.strip() else []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        if end == len(text):
            break
        start = end - chunk_overlap
    return chunks

def build_index(docs_dir: Path = DOCS_DIR, db_dir: Path = DB_DIR, reset: bool = True): 
    pdf_files = list(docs_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {docs_dir}.")
        sys.exit(1)

    print(f"Found {len(pdf_files)} PDF files: {[p.name for p in pdf_files]}")
    print(f"loading embedding model '{EMBEDDING_MODEL_NAME}'...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    client = chromadb.PersistentClient(path=str(db_dir))
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"Deleted existing collection '{COLLECTION_NAME}'.")
        except Exception:
            pass
        collection = client.get_or_create_collection(
            COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
    all_ids, all_docs, all_metas = [], [], []
    for pdf_path in pdf_files:
        source_name = pdf_path.stem
        print(f"Processing '{pdf_path.name}'...")
        
        chunk_idx = 0
        for page_num, raw_text in extract_pages(pdf_path):
            text= clean_text(raw_text)
            if not text:
                continue
            for chunk in chunk_text(text):
                chunk_id = f"{source_name}::p{page_num}::c{chunk_idx}"
                all_ids.append(chunk_id)
                all_docs.append(chunk)
                all_metas.append({"source": pdf_path.name, "page": page_num})
                chunk_idx += 1
        print(f"Processed {chunk_idx} chunks")

    print(f"Generating embeddings for {len(all_docs)} chunks. This may take a while...")
    embeddings = model.encode(all_docs, show_progress_bar=True, batch_size=32).tolist()

    BATCH = 500
    for i in range (0, len(all_ids), BATCH):
        collection.add(
            ids=all_ids[i:i+BATCH],
            documents=all_docs[i:i+BATCH],
            metadatas=all_metas[i:i+BATCH],
            embeddings=embeddings[i:i+BATCH]
        )
    print(f"Done. Indexed {len(all_ids)} chunks from {len(pdf_files)} PDF files into '{DB_DIR}'.")

if __name__ == "__main__":
    build_index()
