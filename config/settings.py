from langchain_openai import OpenAIEmbeddings
import os
import openai
from dotenv import load_dotenv

load_dotenv()

API_KEY_NAME = "OPENAI_API_KEY"
EMBEDDING_MODEL_NAME = "text-embedding-3-small"

openai.api_key = os.getenv(API_KEY_NAME)
embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME)

MODEL = "gpt-4o-mini"
SYSTEM_MESSAGE = f"""
You are a helpful assistant that answers user questions politely.
And when asked about code, security or deployment practices, please only use internal documentation/code and provided context.
Answer clearly and concisely. If unsure, say so politely.
"""

DOC_INDEX_PATH = "../data/faiss_index/faiss_confluence_index"
CODEBASE_INDEX_PATH = "../data/faiss_index/faiss_codebase_index"
BASE_PATH = "./data/__data"