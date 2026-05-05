import chromadb
import numpy as np
import umap
import matplotlib.pyplot as plt

from langchain_community.embeddings import HuggingFaceEmbeddings

# --- Load DB ---
db = chromadb.PersistentClient(path="./chroma_db")
collection = db.get_collection(name= "pdf_chunks")

# --- Get data ---
data = collection.get(include=["embeddings", "documents", "metadatas"])

embeddings = np.array(data["embeddings"])
docs = data["documents"]
metas = data["metadatas"]

# --- Reduce to 2D ---
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine")
coords = reducer.fit_transform(embeddings)

# --- Plot ---
plt.figure(figsize=(10, 7))

sources = [m["source"] for m in metas]
unique_sources = list(set(sources))
colors = {s: i for i, s in enumerate(unique_sources)}

for i, (x, y) in enumerate(coords):
    plt.scatter(x, y, color=plt.cm.tab10(colors[sources[i]]), s=10)

plt.title("Chunk Embedding Visualization (UMAP)")
plt.show()