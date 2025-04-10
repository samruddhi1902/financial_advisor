# rag_engine.py
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import docx
from io import BytesIO
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document as LangchainDocument
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.in_memory import InMemoryDocstore
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_together import ChatTogether
import faiss

# Load and preprocess the fixed file once
def load_fixed_document(filepath):
    document = docx.Document(filepath)
    full_text = "\n\n".join(p.text.strip() for p in document.paragraphs if p.text.strip())
    doc = LangchainDocument(page_content=full_text.strip())
    splitter = RecursiveCharacterTextSplitter(separators=["\n\n"], chunk_size=1500, chunk_overlap=500)
    return splitter.split_documents([doc])

# Build index once
def init_engine():
    chunks = load_fixed_document("/Users/chintamaniborhade/financial_advisor/FinancialData (1) (1).docx")  # path to your fixed file
    embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    texts = [doc.page_content for doc in chunks]
    embeddings = np.array(embedding_model.embed_documents(texts), dtype=np.float32)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    docstore = InMemoryDocstore({i: chunks[i] for i in range(len(chunks))})
    vector_store = FAISS(index, docstore, {}, embedding_model.embed_query)

    return index, docstore, embedding_model

index, docstore, embedding_model = init_engine()

def query_chatbot(question, chat_history):
    query_embedding = embedding_model.embed_query(question)
    D, I = index.search(np.array([query_embedding]), k=3)

    contexts = [docstore.search(idx).page_content for idx in I[0] if idx != -1]
    if not contexts:
        return "No relevant context found."

    context = "\n\n---\n\n".join(contexts)
    history_context = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history])

    chat_model = ChatTogether(
        together_api_key=os.getenv("TOGETHER_API_KEY"), 
        model="mistralai/Mixtral-8x7B-Instruct-v0.1"
    )

    prompt = PromptTemplate(
        input_variables=["history", "context", "question"],
        template="""You are a financial advisor specializing in suggesting investment schemes. Use the following context to answer questions accurately:

Context: {context}

Chat History: {history}

Question: {question}

Strict rules:
1. Use only the provided context.
2. Quote interest rates exactly.
3. Specify whether data is for senior/general citizens.
4. Keep responses concise.

Response:"""
    )

    qa_chain = LLMChain(llm=chat_model, prompt=prompt)
    return qa_chain.run(history=history_context, context=context, question=question)
