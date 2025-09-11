from config import settings
from data import docs
from data.index import get_or_create_store


confluence_store = get_or_create_store(docs.confluence_docs, settings.DOC_INDEX_PATH)
codebase_store = get_or_create_store(docs.codebase_docs, settings.CODEBASE_INDEX_PATH)

def rag_query(query: str) -> str:
    # skip casual greetings
    casual_keywords = {"hi", "hello", "hey", "thanks", "thank you", "good morning", "good evening"}
    if query.strip().lower() in casual_keywords:
        return ""

    # Define a relevance score threshold
    threshold = 1.5

    # Perform similarity search with scores
    confluence_results = confluence_store.similarity_search_with_score(query, k=1)
    codebase_results = codebase_store.similarity_search_with_score(query, k=1)

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
