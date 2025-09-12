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
    threshold = 2.0  

    results = []

    # Heuristic: if query is about code/functions, search codebase more aggressively
    if any(kw in query.lower() for kw in ["function", "class", "code", "definition"]):
        codebase_results = codebase_store.similarity_search_with_score(query, k=5)
        for doc, score in codebase_results:
            if score < threshold:
                results.append(doc)

        # fallback: if nothing found, include all code functions for LLM to judge
        if not results:
            results = [doc for doc in codebase_docs]

    else:
        # Otherwise search both stores normally
        confluence_results = confluence_store.similarity_search_with_score(query, k=2)
        codebase_results = codebase_store.similarity_search_with_score(query, k=2)
        
        for doc, score in (confluence_results + codebase_results):
            if score < threshold:
                results.append(doc)

    # If still no relevant docs, return blank
    if not results:
        return ""

    # Otherwise, return concatenated context
    context = "\n".join([doc.page_content for doc in results])
    return context

