from query import context_search
from config import settings
import openai


def chat_interface(query):
    return context_search.rag_query(query)

def chat(message, history):    
    messages = [{"role": "system", "content": settings.SYSTEM_MESSAGE}]
    
    # Add history if exists
    if history:
        messages = [{"role": "system", "content": settings.SYSTEM_MESSAGE}] + history
    
    # Add use prompt
    messages.append({"role": "user", "content": message})
    
    # add context from Vector DB
    context = context_search.rag_query(message)
    messages.append({"role": "user", "content": context})

    print(f"Context sent to model: {context}")

    # Create chat completion with explicit model name
    stream = openai.chat.completions.create(
        model=settings.MODEL,
        messages=messages,
        stream=True
    )

    response = ""
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            response += chunk.choices[0].delta.content
            yield response