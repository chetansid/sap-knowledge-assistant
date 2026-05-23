from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os
import glob

load_dotenv()

def ingest_documents():
    # Step 1 — Load all PDFs from docs folder
    pdf_files = glob.glob("docs/*.pdf")
    
    if not pdf_files:
        print("No PDFs found in docs/ folder")
        return
    
    all_documents = []
    for pdf_path in pdf_files:
        print(f"Loading: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        all_documents.extend(documents)
    
    print(f"Loaded {len(all_documents)} pages total")

    # Step 2 — Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(all_documents)
    print(f"Split into {len(chunks)} chunks")

    # Step 3 — Convert to embeddings and store in ChromaDB
    print("Creating embeddings and storing in ChromaDB...")
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )
    print(f"Done. {len(chunks)} chunks stored in ChromaDB.")

if __name__ == "__main__":
    ingest_documents()