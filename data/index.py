import os
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from config import settings

embedding_model = OpenAIEmbeddings(model=settings.EMBEDDING_MODEL_NAME)

def get_or_create_store(docs: List[Document], path: str) -> FAISS:
    if os.path.exists(path):
        print(f"Loading FAISS index from {path}")
        return FAISS.load_local(path, embedding_model, allow_dangerous_deserialization=True)
    else:
        print(f"Creating new FAISS index at {path}")
        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
        store = FAISS.from_documents(chunks, embedding_model)
        store.save_local(path)
        return store