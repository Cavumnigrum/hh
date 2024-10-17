import faiss
import numpy as np
from Emb_generation import document_embeddings

max_length = max(embedding.shape[1] for embedding in document_embeddings)
padded_embeddings = [np.pad(embedding, ((0, 0), (0, max_length - embedding.shape[1])), 'constant') for embedding in document_embeddings]

document_embeddings = np.vstack(padded_embeddings)
d = document_embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(document_embeddings)
