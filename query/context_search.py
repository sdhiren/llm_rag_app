from config import settings
from data import docs
from data.index import build_or_load_code_store_from_directory

# confluence_store = get_or_create_store(docs.confluence_docs, settings.DOC_INDEX_PATH)
codebase_store = build_or_load_code_store_from_directory(settings.BASE_PATH, settings.CODEBASE_INDEX_PATH)

def _extract_requested_filename(q: str) -> str | None:
    # Find tokens ending with .py
    for token in q.replace("?", " ").replace(",", " ").split():
        t = token.strip().lower()
        if t.endswith(".py"):
            return t
    return None

def rag_query(query: str) -> str:
    # skip casual greetings
    casual_keywords = {"hi", "hello", "hey", "thanks", "thank you", "good morning", "good evening"}
    if query.strip().lower() in casual_keywords:
        return ""

    # Special case: ask for functions in a specific file (e.g., "users.py")
    requested_file = _extract_requested_filename(query)
    if requested_file:
        hits = codebase_store.similarity_search_with_score(query, k=8, filter={"filename": requested_file})
        if not hits:
            return ""
        # Aggregate unique function names from matched file’s chunks
        functions: list[str] = []
        for doc, _ in hits:
            funcs = (doc.metadata or {}).get("functions") or []
            for f in funcs:
                if f not in functions:
                    functions.append(f)
        return "\n".join(functions)

    # Define a relevance score threshold
    threshold = 2.0

    results = []

    # Heuristic: if query is about code/functions, search codebase more aggressively
    if any(kw in query.lower() for kw in ["function", "class", "code", "definition", "method", "def"]):
        codebase_results = codebase_store.similarity_search_with_score(query, k=5)
        for doc, score in codebase_results:
            if score < threshold:
                results.append(doc)
    else:
        # Otherwise search normally
        codebase_results = codebase_store.similarity_search_with_score(query, k=2)
        for doc, score in codebase_results:
            if score < threshold:
                results.append(doc)

    # If still no relevant docs, return blank
    if not results:
        return ""

    # Otherwise, return concatenated context
    context = "\n".join([doc.page_content for doc in results])
    return context