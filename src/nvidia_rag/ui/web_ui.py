"""
Gradio Web Interface.
Provides a user-friendly frontend for the RAG system.
"""

import hashlib
import os
import uuid

import gradio as gr
import pypdf  # pylint: disable=import-error

from nvidia_rag.core.engine import RAGEngine
from nvidia_rag.tools.splitter import split_text_recursively


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

    def _chat_handler(self, message, _history, use_rag, temperature):
        """
        Internal handler for the ChatInterface component.

        Args:
            message: The current user message.
            _history: The chat history (provided by Gradio, but we use
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

    def _process_file(self, file_obj):
        if file_obj is None:
            return "No file selected."
        try:
            # Detect file extension
            if file_obj.name.lower().endswith(".pdf"):
                reader = pypdf.PdfReader(file_obj.name)
                text_parts = []
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
                text = "\n".join(text_parts)
            else:
                with open(
                    file_obj.name, "r", encoding="utf-8"
                ) as f:
                    text = f.read()

            if not text.strip():
                return "❌ Error: The uploaded file has no readable text content."

            file_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if self.engine.vector_tool.is_document_ingested(file_hash):
                return "⚠️ Document already ingested. Skipping."

            chunks = split_text_recursively(text, chunk_size=1000, chunk_overlap=200)
            if not chunks:
                return "❌ Error: The split text generated no chunks."

            filename = os.path.basename(file_obj.name)
            metadatas = [{"source": filename, "doc_hash": file_hash} for _ in chunks]
            ids = [
                f"{filename}_{file_hash}_{i}_{uuid.uuid4().hex[:8]}"
                for i in range(len(chunks))
            ]
            self.engine.vector_tool.add_documents(chunks, metadatas=metadatas, ids=ids)
            return (
                f"✅ Successfully ingested "
                f"{len(chunks)} chunks!"
            )
        except Exception as e:  # pylint: disable=broad-except
            return f"❌ Error: {e}"

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
                            ["What are the latest AI hardware trends?", True, 0.6],
                            ["Tell me about internal project setup.", True, 0.6],
                            ["Hello! How can you help me today?", True, 0.6]
                        ],
                    )

                # Sidebar/Status Column
                with gr.Column(scale=1):
                    gr.Markdown("### 📄 Document Ingestion")

                    file_input = gr.File(
                        label="Upload Document (TXT, PDF)",
                        file_types=[".txt", ".pdf"]
                    )
                    upload_btn = gr.Button("Upload to Vector DB")
                    upload_status = gr.Markdown()

                    def process_file_wrapper(file_obj):
                        return self._process_file(file_obj)

                    # pylint: disable=no-member
                    upload_btn.click(
                        fn=process_file_wrapper,
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
                    # pylint: disable=no-member
                    clear_btn.click(fn=self.engine.memory.clear)

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
