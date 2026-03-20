from fastapi import FastAPI, UploadFile, File
from rag_pipeline import load_document, split_text, create_vectorstore, ask_question
import shutil
import os

app = FastAPI()
vectorstore = None


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_path = f"temp_{file.filename}"
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        docs = load_document(file_path)
        chunks = split_text(docs)
        
        global vectorstore
        vectorstore = create_vectorstore(chunks)
        return {"message": "File uploaded and processed"}
    
@app.post("/chat")
async def chat(query: str):
    
    if vectorstore is None:
        return {"error": "No document uploaded yet"}
    answer = ask_question(vectorstore,query)
    return {"answer": answer}

@app.get("/")
def root():
    return {"message": "hello"}