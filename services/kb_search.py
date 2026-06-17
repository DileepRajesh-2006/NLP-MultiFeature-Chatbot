# ============================================================
# TASK 3 - KNOWLEDGE BASE SEARCH
# ============================================================

import os
import pickle
import faiss
import numpy as np

from sentence_transformers import (
    SentenceTransformer
)

# ============================================================
# PATHS
# ============================================================

INDEX_PATH = "vector_db/faiss_index.index"

TEXTS_PATH = "vector_db/text_chunks.pkl"

# ============================================================
# SIMILARITY THRESHOLD
# FAISS uses L2 distance — lower = more similar
# Distance > 1.2 means the match is too weak to use
# ============================================================

MAX_DISTANCE = 1.2

# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# ============================================================
# LOAD VECTOR DATABASE
# ============================================================

def load_vector_database():

    if not os.path.exists(INDEX_PATH):
        return None

    if not os.path.exists(TEXTS_PATH):
        return None

    try:

        index = faiss.read_index(INDEX_PATH)

        with open(TEXTS_PATH, "rb") as f:
            text_chunks = pickle.load(f)

        return index, text_chunks

    except Exception as e:

        print(f"Database loading error: {e}")

        return None

# ============================================================
# SEARCH KNOWLEDGE BASE
# ============================================================

def search_knowledge_base(
    query,
    top_k=3
):

    db = load_vector_database()

    if db is None:
        return []

    index, text_chunks = db

    try:

        # Generate query embedding
        query_embedding = model.encode([query])

        query_embedding = np.array(
            query_embedding,
            dtype="float32"
        )

        # Search FAISS
        distances, indices = index.search(
            query_embedding,
            top_k
        )

        results = []

        for i, idx in enumerate(indices[0]):

            # Skip invalid indices
            if idx < 0 or idx >= len(text_chunks):
                continue

            # ================================================
            # THRESHOLD CHECK
            # Skip if distance too high (weak match)
            # ================================================

            if distances[0][i] > MAX_DISTANCE:
                continue

            chunk = text_chunks[idx]

            if isinstance(chunk, dict):

                results.append({
                    "source": chunk.get(
                        "source",
                        "Uploaded Document"
                    ),
                    "content": chunk.get(
                        "content",
                        ""
                    ),
                    "distance": float(distances[0][i])
                })

            else:

                results.append({
                    "source":   "Uploaded Document",
                    "content":  str(chunk),
                    "distance": float(distances[0][i])
                })

        return results

    except Exception as e:

        print(f"Search error: {e}")

        return []