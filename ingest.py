import fitz  # PyMuPDF
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- Config ---
PDF_DIR = "./src"
CHROMA_DB_PATH = "./chroma_db"
EMBEDDING_MODEL_NAME = "./models/bge-base-en-v1.5-f32.gguf"
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 256

# --- PDF Extraction ---
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text()
        doc.close()
    except Exception as e:
        print(f"Error with {pdf_path}: {e}")
    return text

def load_documents(directory):
    docs = []
    for file in os.listdir(directory):
        if file.endswith(".pdf"):
            path = os.path.join(directory, file)
            print(f"Loading {file}")
            text = extract_text_from_pdf(path)
            if text:
                docs.append({
                    "page_content": text,
                    "metadata": {"source": file}
                })
    return docs

# --- Chunking ---
def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = []
    for doc in documents:
        chunks.extend(
            splitter.create_documents(
                [doc["page_content"]],
                metadatas=[doc["metadata"]]
            )
        )
    return chunks

# --- Main ingestion ---
def ingest():
    print("Loading PDFs...")
    docs = load_documents(PDF_DIR)

    if not docs:
        print("No documents found.")
        return

    print("Chunking...")
    chunks = chunk_documents(docs)

    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        # model_name=EMBEDDING_MODEL_NAME,
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={"device": "cpu"},  # cpu/cuda
        encode_kwargs={"normalize_embeddings": True}
    )

    print("Creating Chroma DB...")
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH,
        collection_name="pdf_chunks"
    )

    # db.persist() - deprecated
    print("Done.")

if __name__ == "__main__":
    ingest()