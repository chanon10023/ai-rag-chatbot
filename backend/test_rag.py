from rag_pipeline import load_document, split_text, create_vectorstore, ask_question
docs = load_document("../test.pdf")
chunks = split_text(docs)
vectorstore = create_vectorstore(chunks)
question = "Summarize the information from this document"
answer = ask_question(vectorstore, question)
print("\nANSWER:\n", answer)