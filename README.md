# LLM RAG Application

A question-answering application that uses RAG (Retrieval Augmented Generation) to provide context-aware responses by searching through internal documentation and codebase.

## Features
- RAG-powered responses using FAISS vector store
- Searches through Confluence docs and codebase
- Streaming responses from OpenAI API
- Interactive chat interface using Gradio

## Prerequisites
- Python 3.8+
- OpenAI API key
- Git

## Installation & Setup

1. Clone the repository
```bash
git clone git@github.com:sdhiren/llm_rag_app.git
cd llm_rag_app
```
2. Create .env file with your OpenAI API key
```bash
echo "OPENAI_API_KEY=your-api-key-here" > .env
```
3. Setup virtual environment and install dependencies
```bash
make activate_venv
```

## Usage
Run the application
```bash
make run
```

The application will start a Gradio web interface where you can:
Ask questions about internal documentation
Query codebase information
Get context-aware responses


## Environment Variables
Create a .env file in the project root with:
```bash
OPENAI_API_KEY=your-api-key-here
```

