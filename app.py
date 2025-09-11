from chat import chat
import gradio as gr

if __name__ == "__main__":
    gr.ChatInterface(fn=chat.chat, type="messages").launch(server_name="0.0.0.0", server_port=7860)
    
