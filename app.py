import os
import openai
import gradio as gr
from typing import List
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.schema import Document
from dotenv import load_dotenv


# Setup env variables and models
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
MODEL = "gpt-4o-mini"
SYSTEM_MESSAGE = f"""
You are a helpful assistant that answers user questions politely. You dont have to use internal context all the time, only when relevant question is asked.
And when asked about code, security or deployment practices, please only use internal documentation/code and provided context.
Answer clearly and concisely. If unsure, say so politely.
"""


# Sample data to work with
confluence_docs = [
    Document(page_content="The standard deployment process requires code reviews, CI pipeline execution, and approval from QA before release."),
    Document(page_content="All production deployments must be approved by the release manager and require a rollback plan."),
    Document(page_content="Security processes mandate that all dependencies are scanned weekly using Snyk and reported to the security team."),
]

codebase_docs = [
    Document(page_content="def calculate_discount(price, percentage): return price - (price * percentage/100)"),
    Document(page_content="class ShoppingCart: def __init__(self): self.items = []"),
    Document(page_content="def connect_to_db(uri): # establishes a PostgreSQL connection using psycopg2.connect"),
]


DOC_INDEX_PATH = "./faiss_index/faiss_confluence_index"
CODEBASE_INDEX_PATH = "./faiss_index/faiss_codebase_index"


# Load Vector Stores
def get_or_create_store(docs: List[Document], path: str) -> FAISS:
    if os.path.exists(path):
        print(f"Loading FAISS index from {path}")
        return FAISS.load_local(path, embedding_model, allow_dangerous_deserialization=True)
    else:
        print(f"Creating new FAISS index at {path}")
        splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)
        chunks = splitter.split_documents(docs)
        store = FAISS.from_documents(chunks, embedding_model)
        store.save_local(path)
        return store

confluence_store = get_or_create_store(confluence_docs, DOC_INDEX_PATH)
codebase_store = get_or_create_store(codebase_docs, CODEBASE_INDEX_PATH)

def rag_query(query: str) -> str:
    # skip casual greetings
    casual_keywords = {"hi", "hello", "hey", "thanks", "thank you", "good morning", "good evening"}
    if query.strip().lower() in casual_keywords:
        return ""

    # Define a relevance score threshold
    threshold = 2.0

    # Perform similarity search with scores
    confluence_results = confluence_store.similarity_search_with_score(query, k=2)
    codebase_results = codebase_store.similarity_search_with_score(query, k=2)

    # Filter results based on threshold
    filtered_docs = [
        doc for doc, score in (confluence_results + codebase_results) if score < threshold
    ]

    # If no relevant docs, return blank
    if not filtered_docs:
        return ""

    # Otherwise, return concatenated context
    context = "\n".join([doc.page_content for doc in filtered_docs])
    return context


# UI code with Gradio
def chat_interface(query):
    return rag_query(query)

def chat(message, history):    
    messages = [{"role": "system", "content": SYSTEM_MESSAGE}]
    
    # Add history if exists
    if history:
        messages = [{"role": "system", "content": SYSTEM_MESSAGE}] + history
    
    # Add use prompt
    messages.append({"role": "user", "content": message})
    
    # add context from Vector DB
    context = rag_query(message)
    messages.append({"role": "user", "content": context})

    print(f"Context sent to model: {context}")

    # Create chat completion with explicit model name
    stream = openai.chat.completions.create(
        model=MODEL,
        messages=messages,
        stream=True
    )

    response = ""
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            response += chunk.choices[0].delta.content
            yield response

if __name__ == "__main__":    
    gr.ChatInterface(fn=chat, type="messages").launch(server_name="0.0.0.0", server_port=7860)
    
