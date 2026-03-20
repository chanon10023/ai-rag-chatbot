from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import requests

def load_document(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load()

def split_text(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50
    )
    return splitter.split_documents(documents)

def create_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

def retrieve_context(vectorstore, query):
    docs = vectorstore.similarity_search(query, k=3)
    return "\n".join([doc.page_content for doc in docs])

def ask_llm(prompt):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json = {
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json()["response"]

def ask_question(vectorstore, question):
    context = retrieve_context(vectorstore, question)
    prompt = f"""
                You are a company assistant.
                Answer only from the context.

                Context:
                {context}

                Question:
                {question}
              """
    return ask_llm(prompt)