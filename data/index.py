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
        index_docs(docs, path)

def index_docs(docs: List[Document], path: str) -> FAISS:
    print(f"Creating new FAISS index at {path}")
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    store = FAISS.from_documents(chunks, embedding_model)
    store.save_local(path)
    return store

import ast
from typing import Dict, Optional, Tuple

def _extract_python_metadata(code: str) -> Tuple[list, list, list, Optional[str]]:
    """
    Returns: (functions, classes, methods, docstring_summary)
    - functions: list[str] of top-level function names
    - classes: list[str] of class names
    - methods: list[str] like 'Class.method'
    - docstring_summary: first sentence/line of module docstring if present
    """
    try:
        tree = ast.parse(code)
    except Exception:
        return [], [], [], None

    functions = []
    classes = []
    methods = []

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            functions.append(node.name)
        elif isinstance(node, ast.AsyncFunctionDef):
            functions.append(node.name)
        elif isinstance(node, ast.ClassDef):
            classes.append(node.name)
            for cnode in node.body:
                if isinstance(cnode, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    methods.append(f"{node.name}.{cnode.name}")

    doc = ast.get_docstring(tree)
    if doc:
        # Take first non-empty line as summary
        for line in doc.splitlines():
            s = line.strip()
            if s:
                return functions, classes, methods, s[:200]
    return functions, classes, methods, None

def _default_summary(functions: list, classes: list, methods: list) -> str:
    return (
        f"Python module with {len(functions)} function(s), "
        f"{len(classes)} class(es), and {len(methods)} method(s)."
    )

def _rel_module_from_path(base_dir: str, file_path: str) -> str:
    rel = os.path.relpath(file_path, base_dir)
    no_ext = os.path.splitext(rel)[0]
    parts = []
    for p in no_ext.split(os.sep):
        if p == "__init__":
            continue
        parts.append(p)
    return ".".join(parts)

def load_python_docs_from_directory(base_dir: str) -> List[Document]:
    """
    Recursively load all .py files under base_dir into Document objects with rich metadata:
    - filename: basename
    - path: absolute file path
    - module: dotted path relative to base_dir
    - functions: list[str]
    - classes: list[str]
    - methods: list[str] ('Class.method')
    - summary: short description from module docstring or synthesized fallback
    The raw file content becomes page_content; metadata is preserved during chunking.
    """
    documents: List[Document] = []
    base_dir = os.path.abspath(base_dir)

    for root, _, files in os.walk(base_dir):
        for fname in files:
            if not fname.endswith(".py"):
                continue
            file_path = os.path.join(root, fname)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    code = f.read()
            except Exception as e:
                print(f"Skipping {file_path}: {e}")
                continue

            functions, classes, methods, docsum = _extract_python_metadata(code)
            summary = docsum or _default_summary(functions, classes, methods)
            module = _rel_module_from_path(base_dir, file_path)

            metadata: Dict[str, object] = {
                "filename": os.path.basename(file_path),
                "path": file_path,
                "module": module,
                "functions": functions,
                "classes": classes,
                "methods": methods,
                "summary": summary,
                "source": "codebase",
                "language": "python",
            }

            documents.append(Document(page_content=code, metadata=metadata))

    return documents

def build_or_load_code_store_from_directory(base_dir: str, index_path: str) -> Optional[FAISS]:
    """
    Build a FAISS index from all Python files under base_dir, or load it if present.
    """
    if os.path.exists(index_path):
        print(f"Loading FAISS index from {index_path}")
        return FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)

    docs = load_python_docs_from_directory(base_dir)
    if not docs:
        print(f"No Python files found under {base_dir}")
        return None
    return index_docs(docs, index_path)