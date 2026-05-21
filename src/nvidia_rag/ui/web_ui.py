import gradio as gr
from nvidia_rag.core.engine import RAGEngine

class WebUI:
    """Gradio web interface for the RAG system."""
    
    def __init__(self, engine: RAGEngine):
        self.engine = engine

    def _chat_handler(self, query: str):
        response, tokens, _ = self.engine.generate_response(query, use_rag=True)
        return response

    def build(self) -> gr.Interface:
        return gr.Interface(
            fn=self._chat_handler,
            inputs=gr.Textbox(label="Ask a question", placeholder="e.g., What is the latest news in AI?"),
            outputs=gr.Textbox(label="Hybrid RAG Response"),
            title="NVIDIA Llama-3 Hybrid RAG",
            description="Production-ready hybrid RAG system with real-time web grounding."
        )

    def launch(self, **kwargs):
        demo = self.build()
        demo.launch(**kwargs)
