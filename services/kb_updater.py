# ============================================================
# TASK 3 - KNOWLEDGE BASE UPDATER
# Supports TXT, PDF, DOCX
# ============================================================

import os
import pickle
import faiss
import numpy as np

from sentence_transformers import (
    SentenceTransformer
)

from pypdf import PdfReader
from docx import Document

# ============================================================
# PATHS
# ============================================================

DOCUMENTS_PATH = "knowledge_base/documents"

VECTOR_DB_PATH = "vector_db"

INDEX_PATH = os.path.join(
    VECTOR_DB_PATH,
    "faiss_index.index"
)

TEXTS_PATH = os.path.join(
    VECTOR_DB_PATH,
    "text_chunks.pkl"
)

# ============================================================
# EMBEDDING MODEL
# ============================================================

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# ============================================================
# READ FILES
# ============================================================

def read_file(filepath):

    # TXT FILE
    if filepath.endswith(".txt"):

        with open(
            filepath,
            "r",
            encoding="utf-8",
            errors="ignore"
        ) as f:

            return f.read()

    # PDF FILE
    elif filepath.endswith(".pdf"):

        text = ""

        try:

            reader = PdfReader(filepath)

            for page in reader.pages:

                extracted = page.extract_text()

                if extracted:

                    text += extracted + "\n"

        except Exception as e:

            print(f"PDF Error: {e}")

        return text

    # DOCX FILE
    elif filepath.endswith(".docx"):

        try:

            doc = Document(filepath)

            text = "\n".join(

                [p.text for p in doc.paragraphs]
            )

            return text

        except Exception as e:

            print(f"DOCX Error: {e}")

            return ""

    return ""

# ============================================================
# SMART TEXT CHUNKING
# ============================================================

def split_text(
    text,
    chunk_size=500
):

    words = text.split()

    chunks = []

    for i in range(
        0,
        len(words),
        chunk_size
    ):

        chunk = " ".join(
            words[i:i + chunk_size]
        )

        if len(chunk.strip()) > 50:

            chunks.append(chunk)

    return chunks

# ============================================================
# BUILD VECTOR DATABASE
# ============================================================

def build_vector_database():

    os.makedirs(
        VECTOR_DB_PATH,
        exist_ok=True
    )

    all_chunks = []

    # ========================================================
    # READ DOCUMENTS
    # ========================================================

    for filename in os.listdir(
        DOCUMENTS_PATH
    ):

        filepath = os.path.join(
            DOCUMENTS_PATH,
            filename
        )

        text = read_file(filepath)

        if not text.strip():

            print(
                f"Skipped empty file: {filename}"
            )

            continue

        chunks = split_text(text)

        for chunk in chunks:

            all_chunks.append({

                "source": filename,

                "content": chunk
            })

    # ========================================================
    # NO DATA
    # ========================================================

    if not all_chunks:

        return 0

    # ========================================================
    # CREATE EMBEDDINGS
    # ========================================================

    texts = [

        chunk["content"]

        for chunk in all_chunks
    ]

    embeddings = model.encode(
        texts,
        show_progress_bar=True
    )

    embeddings = np.array(
        embeddings,
        dtype="float32"
    )

    # ========================================================
    # CREATE FAISS INDEX
    # ========================================================

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(embeddings)

    # ========================================================
    # SAVE INDEX
    # ========================================================

    faiss.write_index(
        index,
        INDEX_PATH
    )

    with open(TEXTS_PATH, "wb") as f:

        pickle.dump(
            all_chunks,
            f
        )

    return len(all_chunks)