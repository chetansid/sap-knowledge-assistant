from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from query import ask_question
from ingest import ingest_documents
import shutil
import os

app = FastAPI(
    title="SAP Knowledge Assistant",
    description="Ask questions about SAP documentation using AI",
    version="1.0.0"
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/ui")
def ui():
    return FileResponse("static/index.html")

class QuestionRequest(BaseModel):
    question: str

class QuestionResponse(BaseModel):
    question: str
    answer: str
    source_pages: list

@app.get("/")
def root():
    return {"message": "SAP Knowledge Assistant is running"}

@app.post("/ask", response_model=QuestionResponse)
def ask(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    result = ask_question(request.question)
    return result

@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    # Save uploaded file to docs folder
    os.makedirs("docs", exist_ok=True)
    file_path = f"docs/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Re-ingest all documents
    ingest_documents()
    
    return {"message": f"{file.filename} uploaded and ingested successfully"}

@app.get("/health")
def health():
    return {"status": "healthy"}