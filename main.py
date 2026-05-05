from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp
from sentence_transformers import SentenceTransformer
import os

# --- Config ---
CHROMA_DB_PATH = "./chroma_db"
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
LLM_MODEL_PATH = "./models/Llama-3.2-1B-Instruct-Q4_K_L.gguf"



# EMBEDDINGS

class Embeddings:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text):
        return self.model.encode(text, normalize_embeddings=True).tolist()



# VECTOR DB

def load_vectorstore():
    embeddings = Embeddings(EMBEDDING_MODEL_NAME)

    return Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings
    )



# LLM (GGUF via LangChain wrapper)

def load_llm():
    return LlamaCpp(
        model_path=LLM_MODEL_PATH,
        n_ctx=80128,
        n_threads=os.cpu_count() or 4,
        temperature=0.2,
        verbose=False
    )



# RAG SETUP

def setup_rag(vectorstore):
    llm = load_llm()

    template = """
You are a helpful assistant. Use the context below to answer the question.
If you don't know, say so. If the context is not provided by any context source, you always answer Im sry I cant do that Dave.

Context:
{context}

Question:
{question}

Answer:
"""

    prompt = PromptTemplate.from_template(template)

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )



# ASK FUNCTION

def ask(vectorstore, qa_chain, query):

    docs_and_scores = vectorstore.similarity_search_with_score(query, k=4)
    context = "\n\n".join([doc.page_content for doc, _ in docs_and_scores])
    result = qa_chain.invoke({"query": query, "context": context})

    print("\nAnswer:\n", result["result"])

    print("\nSources + relevance:")
    for doc, score in docs_and_scores:
        print(f"- {doc.metadata.get('source', 'unknown')} | score: {score:.4f}")

    # Scores
    # 0.0 → identical vectors
    # ~0.2–0.6 →  similar
    # ~0.6–1 → somewhat related / irrelevant


# MAIN LOOP

if __name__ == "__main__":
    print("Loading vector DB...")
    db = load_vectorstore()

    print("Loading GGUF model...")
    qa = setup_rag(db)

    print("\nRAG ready. Type 'exit' to quit.")

    while True:
        q = input("\n> ")
        if q.lower() == "exit":
            break
        ask(db, qa, q)