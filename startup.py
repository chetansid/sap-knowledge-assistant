from ingest import ingest_documents
import os

if not os.path.exists("chroma_db"):
    print("No ChromaDB found — running ingestion...")
    ingest_documents()
else:
    print("ChromaDB already exists — skipping ingestion.")