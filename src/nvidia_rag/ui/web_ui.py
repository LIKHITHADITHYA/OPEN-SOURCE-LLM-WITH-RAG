"""
Gradio Web Interface.
Provides a user-friendly frontend for the RAG system.
"""

import gradio as gr
from nvidia_rag.core.engine import RAGEngine


class WebUI:
    """
    Advanced Gradio web interface implementation.
    Features a chat interface with additional system controls and
    status indicators.
    """

    def __init__(self, engine: RAGEngine):
        """
        Initialize the Web UI with the core RAG engine.

        Args:
            engine: The RAGEngine instance to handle queries.
        """
        self.engine = engine

    def _chat_handler(self, message, history, use_rag, temperature):
        """
        Internal handler for the ChatInterface component.

        Args:
            message: The current user message.
            history: The chat history (provided by Gradio, but we use
                     engine's memory).
            use_rag: State of the 'Enable RAG' checkbox.
            temperature: Current value of the temperature slider.

        Returns:
            The formatted response from the RAG engine.
        """
        # Execute the query through the engine
        response, tokens, source = self.engine.generate_response(
            message,
            use_rag=use_rag,
            temperature=temperature
        )

        # Format the final output with metadata
        return (
            f"{response}\n\n"
            f"--- Metadata ---\n"
            f"[Tokens: {tokens} | Source: {source}]"
        )

    def build(self) -> gr.Blocks:
        """
        Constructs the Gradio UI layout using Blocks.

        Returns:
            A gr.Blocks object containing the full UI structure.
        """
        with gr.Blocks(title="NVIDIA Llama-3 Advanced RAG") as demo:
            gr.Markdown("# 🚀 NVIDIA Llama-3 Hybrid RAG")
            gr.Markdown(
                "A modular agentic RAG system. It automatically routes "
                "your query to **Web Search** or **Local Documents** "
                "based on intent."
            )

            with gr.Row():
                # Main Chat Column
                with gr.Column(scale=4):
                    gr.ChatInterface(
                        fn=self._chat_handler,
                        additional_inputs=[
                            gr.Checkbox(
                                label="Enable RAG Grounding",
                                value=True
                            ),
                            gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                value=0.6,
                                step=0.1,
                                label="Sampling Temperature"
                            )
                        ],
                        examples=[
                            "What are the latest AI hardware trends?",
                            "Tell me about internal project setup.",
                            "Hello! How can you help me today?"
                        ],
                    )

                # Sidebar/Status Column
                with gr.Column(scale=1):
                    gr.Markdown("### 📄 Document Ingestion")

                    file_input = gr.File(
                        label="Upload Document (TXT)",
                        file_types=[".txt"]
                    )
                    upload_btn = gr.Button("Upload to Vector DB")
                    upload_status = gr.Markdown()

                    def process_file(file_obj):
                        if file_obj is None:
                            return "No file selected."
                        try:
                            with open(
                                file_obj.name, "r", encoding="utf-8"
                            ) as f:
                                text = f.read()
                            # Naive chunking: splits by ~1000 characters
                            chunks = [
                                text[i:i+1000]
                                for i in range(0, len(text), 1000)
                            ]
                            self.engine.vector_tool.add_documents(chunks)
                            return (
                                f"✅ Successfully ingested "
                                f"{len(chunks)} chunks!"
                            )
                        except Exception as e:  # noqa: BLE001
                            return f"❌ Error: {e}"

                    upload_btn.click(
                        fn=process_file,
                        inputs=[file_input],
                        outputs=[upload_status]
                    )

                    gr.Markdown("---")
                    gr.Markdown("### ⚙️ System Controls")

                    # Button to reset the conversation history in the engine
                    clear_btn = gr.Button(
                        "🗑️ Clear Session Memory",
                        variant="stop"
                    )
                    clear_btn.click(fn=lambda: self.engine.memory.clear())

                    gr.Markdown("---")
                    gr.Markdown("### 📊 Backend Info")
                    gr.Markdown("- **Model:** Llama-3 (NVIDIA NIM)")
                    gr.Markdown("- **Web Retrieval:** SerpApi")
                    gr.Markdown("- **Local Retrieval:** ChromaDB")

        return demo

    def launch(self, **kwargs):
        """
        Starts the Gradio web server.

        Args:
            **kwargs: Arguments passed directly to gr.Blocks.launch().
        """
        demo = self.build()
        demo.launch(**kwargs)
