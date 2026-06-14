"""
Unit tests for the RAG system components with mocked network calls.
"""

# pylint: disable=import-error,missing-function-docstring,protected-access,import-outside-toplevel,unused-argument,unused-variable

import sys
import threading
from unittest.mock import MagicMock

# Mock chromadb globally to prevent lock contention on the database file
mock_chroma_client = MagicMock()
mock_chroma_collection = MagicMock()
mock_chroma_client.get_or_create_collection.return_value = mock_chroma_collection

class DummyEmbeddingFunction:
    """Plain class wrapper to replace Mock class inheritance issues in tests."""
    pass

sys.modules["chromadb"] = MagicMock()
import chromadb
chromadb.PersistentClient = MagicMock(return_value=mock_chroma_client)
chromadb.EmbeddingFunction = DummyEmbeddingFunction

# Mock pypdf globally to prevent actual file parsing and import errors
mock_pypdf = MagicMock()
mock_page1 = MagicMock()
mock_page1.extract_text.return_value = "Page one content."
mock_page2 = MagicMock()
mock_page2.extract_text.return_value = "Page two content."
mock_pypdf.PdfReader.return_value.pages = [mock_page1, mock_page2]
sys.modules["pypdf"] = mock_pypdf

import unittest
from unittest.mock import patch

from nvidia_rag.core.memory import ConversationMemory
from nvidia_rag.core.router import QueryRouter
from nvidia_rag.core.engine import RAGEngine
from nvidia_rag.tools.search import SearchTool
from nvidia_rag.tools.splitter import split_text_recursively
from nvidia_rag.ui.web_ui import WebUI
from nvidia_rag.api.server import app


class ThreadSafeMockRecorder:
    """A thread-safe mock recorder to prevent deadlocks in ThreadPoolExecutor."""
    def __init__(self, return_value):
        self.return_value = return_value
        self.calls = []
        self.lock = threading.Lock()

    def __call__(self, *args, **kwargs):
        with self.lock:
            self.calls.append(args)
        return self.return_value


class TestConversationMemory(unittest.TestCase):
    """Verifies that conversation memory limits history correctly."""

    def test_add_and_prune(self):
        memory = ConversationMemory(max_turns=2)
        memory.add_turn("user", "Hello")
        memory.add_turn("assistant", "Hi there!")
        
        # Check initial state
        self.assertEqual(len(memory.get_context()), 2)
        self.assertEqual(memory.get_context()[0]["content"], "Hello")
        
        # Add more turns exceeding limit (2 turns = 4 messages)
        memory.add_turn("user", "What is the weather?")
        memory.add_turn("assistant", "It is sunny.")
        memory.add_turn("user", "Thanks")
        memory.add_turn("assistant", "You're welcome.")
        
        # Verify pruning
        self.assertEqual(len(memory.get_context()), 4)
        self.assertEqual(memory.get_context()[0]["content"], "What is the weather?")
        self.assertEqual(memory.get_context()[-1]["content"], "You're welcome.")

    def test_format_for_prompt(self):
        memory = ConversationMemory()
        self.assertEqual(memory.format_for_prompt(), "")
        
        memory.add_turn("user", "A")
        memory.add_turn("assistant", "B")
        formatted = memory.format_for_prompt()
        self.assertIn("User: A", formatted)
        self.assertIn("Assistant: B", formatted)


class TestRecursiveSplitter(unittest.TestCase):
    """Verifies that split_text_recursively splits texts cleanly."""

    def test_splitting_boundaries(self):
        text = (
            "Paragraph one is here.\n\n"
            "Paragraph two is also here.\n"
            "Paragraph three has some lines."
        )
        chunks = split_text_recursively(text, chunk_size=30, chunk_overlap=5)
        # Should split at paragraph breaks or newlines and not break words
        self.assertTrue(len(chunks) >= 2)
        for chunk in chunks:
            self.assertTrue(len(chunk) <= 35)  # accounting for overlap buffer range


class TestQueryRouter(unittest.TestCase):
    """Tests the router decision logic with mocked LLM responses."""

    def setUp(self):
        self.mock_client = MagicMock()
        self.router = QueryRouter(self.mock_client)

    def test_route_decisions(self):
        # Setup mock for WEB
        self.mock_client.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(content="WEB"))
        ]
        self.assertEqual(self.router.route_query("Bitcoin price"), ["WEB"])

        # Setup mock for LOCAL
        self.mock_client.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(content="LOCAL"))
        ]
        self.assertEqual(self.router.route_query("internal servers"), ["LOCAL"])

        # Setup mock for NONE
        self.mock_client.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(content="NONE"))
        ]
        self.assertEqual(self.router.route_query("hello"), ["NONE"])

    def test_fallback_on_garbage_or_exception(self):
        # Garbage output
        self.mock_client.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(content="GARBAGE"))
        ]
        self.assertEqual(self.router.route_query("unknown"), ["WEB"])

        # Exception raised
        self.mock_client.chat.completions.create.side_effect = Exception("API down")
        self.assertEqual(self.router.route_query("unknown"), ["WEB"])

        # NoneType content response
        self.mock_client.chat.completions.create.side_effect = None
        self.mock_client.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(content=None))
        ]
        self.assertEqual(self.router.route_query("unknown"), ["WEB"])

    def test_route_with_uploaded_sources(self):
        # Mock LLM response
        self.mock_client.chat.completions.create.side_effect = None
        self.mock_client.chat.completions.create.return_value.choices = [
            MagicMock(message=MagicMock(content="LOCAL"))
        ]
        
        sources = ["doc1.pdf", "doc2.txt"]
        self.router.route_query("Explain doc1.pdf specs", uploaded_sources=sources)
        
        # Verify that create was called and the prompt included the sources
        call_args = self.mock_client.chat.completions.create.call_args[1]
        prompt = call_args["messages"][0]["content"]
        self.assertIn("doc1.pdf, doc2.txt", prompt)


class TestSearchTool(unittest.TestCase):
    """Tests the caching and formatting of the search tool."""

    @patch("nvidia_rag.tools.search.SerpAPIWrapper")
    def test_caching_and_formatting(self, mock_serp_wrapper):
        mock_instance = mock_serp_wrapper.return_value
        mock_instance.run.return_value = "Search result body"

        tool = SearchTool(api_key="mock_key")
        
        # First call should call mock
        res1 = tool.search_query("test query")
        self.assertEqual(res1, "Search result body")
        self.assertEqual(mock_instance.run.call_count, 1)

        # Second call with same query should hit cache
        res2 = tool.search_query("test query")
        self.assertEqual(res2, "Search result body")
        self.assertEqual(mock_instance.run.call_count, 1)


class TestVectorStoreTool(unittest.TestCase):
    """Tests VectorStoreTool functions."""

    def test_get_uploaded_sources(self):
        # Override mock collection return value specifically for this test
        mock_chroma_collection.get.return_value = {
            "metadatas": [
                {"source": "doc1.pdf"},
                {"source": "doc2.txt"},
                {"source": "doc1.pdf"}, # duplicate
                None,                  # edge case
                {"other": "metadata"}  # missing source key
            ]
        }

        from nvidia_rag.tools.vector_store import VectorStoreTool
        tool = VectorStoreTool(persist_directory="dummy_dir")

        sources = tool.get_uploaded_sources()
        self.assertEqual(sources, ["doc1.pdf", "doc2.txt"])


class TestRAGEngine(unittest.TestCase):
    """Tests the integration of memory, routing, and generator in RAGEngine."""

    @patch("nvidia_rag.core.engine.OpenAI")
    @patch("nvidia_rag.core.engine.SearchTool")
    @patch("nvidia_rag.core.engine.VectorStoreTool")
    def test_generate_response_web_flow(self, mock_vector, mock_search, mock_openai):
        # Setup instances
        mock_search_inst = mock_search.return_value
        web_recorder = ThreadSafeMockRecorder("Mocked web results")
        mock_search_inst.search_query = web_recorder

        mock_client_inst = mock_openai.return_value
        # Mocking router completion
        mock_router_resp = MagicMock()
        mock_router_resp.choices = [MagicMock(message=MagicMock(content="WEB"))]
        # Mocking main engine generation completion
        mock_engine_resp = MagicMock()
        mock_engine_resp.choices = [MagicMock(message=MagicMock(content="LLM Answer"))]
        mock_engine_resp.usage.total_tokens = 42

        mock_client_inst.chat.completions.create.side_effect = [
            mock_router_resp,  # call 1: router
            mock_engine_resp   # call 2: engine response generator
        ]

        engine = RAGEngine(client=mock_client_inst)
        resp, tokens, source = engine.generate_response("Test query", use_rag=True)

        # Verify flow
        self.assertEqual(resp, "LLM Answer")
        self.assertEqual(tokens, 42)
        self.assertEqual(source, "WEB")
        self.assertEqual(len(web_recorder.calls), 1)
        self.assertEqual(web_recorder.calls[0][0], "Test query")

        # Check memory was updated
        self.assertEqual(len(engine.memory.get_context()), 2)
        self.assertEqual(engine.memory.get_context()[0]["content"], "Test query")
        self.assertEqual(engine.memory.get_context()[1]["content"], "LLM Answer")

    @patch("nvidia_rag.core.engine.OpenAI")
    @patch("nvidia_rag.core.engine.SearchTool")
    @patch("nvidia_rag.core.engine.VectorStoreTool")
    def test_generate_response_none_content(self, mock_vector, mock_search, mock_openai):
        mock_search_inst = mock_search.return_value
        mock_search_inst.search_query = ThreadSafeMockRecorder("Mocked web results")

        mock_client_inst = mock_openai.return_value
        
        # Router returns WEB, Generator returns None content
        mock_router_resp = MagicMock()
        mock_router_resp.choices = [MagicMock(message=MagicMock(content="WEB"))]
        
        mock_engine_resp = MagicMock()
        mock_engine_resp.choices = [MagicMock(message=MagicMock(content=None))]
        mock_engine_resp.usage.total_tokens = 0
        
        mock_client_inst.chat.completions.create.side_effect = [
            mock_router_resp,
            mock_engine_resp
        ]
        
        engine = RAGEngine(client=mock_client_inst)
        resp, tokens, source = engine.generate_response("Test query")
        
        self.assertEqual(resp, "")
        self.assertEqual(tokens, 0)
        self.assertEqual(source, "WEB")

    @patch("nvidia_rag.core.engine.OpenAI")
    @patch("nvidia_rag.core.engine.SearchTool")
    @patch("nvidia_rag.core.engine.VectorStoreTool")
    def test_generate_response_web_validation_fallback(self, mock_vector, mock_search, mock_openai):
        mock_search_inst = mock_search.return_value
        mock_search_inst.search_query = ThreadSafeMockRecorder("Real-time search results could not be retrieved due to an error.")

        mock_client_inst = mock_openai.return_value
        
        mock_router_resp = MagicMock()
        mock_router_resp.choices = [MagicMock(message=MagicMock(content="WEB"))]
        
        mock_engine_resp = MagicMock()
        mock_engine_resp.choices = [MagicMock(message=MagicMock(content="Fallback answer"))]
        mock_engine_resp.usage.total_tokens = 10
        
        mock_client_inst.chat.completions.create.side_effect = [
            mock_router_resp,
            mock_engine_resp
        ]
        
        engine = RAGEngine(client=mock_client_inst)
        resp, tokens, source = engine.generate_response("Test query")
        
        self.assertEqual(resp, "Fallback answer")
        self.assertEqual(source, "NONE")

    @patch("nvidia_rag.core.engine.OpenAI")
    @patch("nvidia_rag.core.engine.SearchTool")
    @patch("nvidia_rag.core.engine.VectorStoreTool")
    def test_generate_response_local_validation_fallback(self, mock_vector, mock_search, mock_openai):
        mock_vector_inst = mock_vector.return_value
        mock_vector_inst.search = ThreadSafeMockRecorder("No relevant local documents found.")

        mock_client_inst = mock_openai.return_value
        
        mock_router_resp = MagicMock()
        mock_router_resp.choices = [MagicMock(message=MagicMock(content="LOCAL"))]
        
        mock_engine_resp = MagicMock()
        mock_engine_resp.choices = [MagicMock(message=MagicMock(content="Fallback answer"))]
        mock_engine_resp.usage.total_tokens = 10
        
        mock_client_inst.chat.completions.create.side_effect = [
            mock_router_resp,
            mock_engine_resp
        ]
        
        engine = RAGEngine(client=mock_client_inst)
        resp, tokens, source = engine.generate_response("Test query")
        
        self.assertEqual(resp, "Fallback answer")
        self.assertEqual(source, "NONE")

    @patch("nvidia_rag.core.engine.OpenAI")
    @patch("nvidia_rag.core.engine.SearchTool")
    @patch("nvidia_rag.core.engine.VectorStoreTool")
    def test_generate_response_multi_route(self, mock_vector, mock_search, mock_openai):
        # Setup search tool results
        mock_search_inst = mock_search.return_value
        web_recorder = ThreadSafeMockRecorder("Mocked web results for H100")
        mock_search_inst.search_query = web_recorder

        # Setup vector store results
        mock_vector_inst = mock_vector.return_value
        local_recorder = ThreadSafeMockRecorder("Mocked local doc info about H100 specs")
        mock_vector_inst.search = local_recorder

        mock_client_inst = mock_openai.return_value
        
        # Router returns both WEB and LOCAL
        mock_router_resp = MagicMock()
        mock_router_resp.choices = [MagicMock(message=MagicMock(content="WEB, LOCAL"))]
        
        # Engine generator returns answer
        mock_engine_resp = MagicMock()
        mock_engine_resp.choices = [MagicMock(message=MagicMock(content="Merged RAG Answer"))]
        mock_engine_resp.usage.total_tokens = 100
        
        mock_client_inst.chat.completions.create.side_effect = [
            mock_router_resp,
            mock_engine_resp
        ]
        
        engine = RAGEngine(client=mock_client_inst)
        resp, tokens, source = engine.generate_response("H100 specs and cost", use_rag=True)
        
        self.assertEqual(resp, "Merged RAG Answer")
        self.assertIn("WEB", source)
        self.assertIn("LOCAL", source)
        
        # Verify both search and vector store were queried via thread-safe recorder
        self.assertEqual(len(web_recorder.calls), 1)
        self.assertEqual(web_recorder.calls[0][0], "H100 specs and cost")
        self.assertEqual(len(local_recorder.calls), 1)
        self.assertEqual(local_recorder.calls[0][0], "H100 specs and cost")

    @patch("nvidia_rag.core.engine.OpenAI")
    @patch("nvidia_rag.core.engine.SearchTool")
    @patch("nvidia_rag.core.engine.VectorStoreTool")
    def test_condense_query_with_history(self, mock_vector, mock_search, mock_openai):
        mock_client_inst = mock_openai.return_value
        mock_condensed_resp = MagicMock()
        mock_condensed_resp.choices = [MagicMock(message=MagicMock(content="What is the GPU cost?"))]
        mock_client_inst.chat.completions.create.return_value = mock_condensed_resp

        engine = RAGEngine(client=mock_client_inst)
        engine.memory.add_turn("user", "What is the H100 GPU?")
        engine.memory.add_turn("assistant", "It is an enterprise GPU.")

        condensed = engine._condense_query("How much does it cost?")
        self.assertEqual(condensed, "What is the GPU cost?")
        
        # Verify the LLM was called with the history
        call_args = mock_client_inst.chat.completions.create.call_args[1]
        prompt = call_args["messages"][0]["content"]
        self.assertIn("What is the H100 GPU?", prompt)
        self.assertIn("How much does it cost?", prompt)


class TestPDFIngestion(unittest.TestCase):
    """Verifies that PDF upload and text extraction works in both Web UI and API."""

    def test_web_ui_pdf_ingestion(self):
        # Setup mock engine and vector store
        mock_engine = MagicMock()
        mock_vector_tool = MagicMock()
        mock_vector_tool.is_document_ingested.return_value = False
        mock_engine.vector_tool = mock_vector_tool

        ui = WebUI(mock_engine)

        # Call _process_file
        mock_file = MagicMock()
        mock_file.name = "test_doc.pdf"
        
        res = ui._process_file(mock_file)

        # Assert results
        self.assertIn("Successfully ingested", res)
        mock_vector_tool.add_documents.assert_called_once()
        called_args = mock_vector_tool.add_documents.call_args[0][0]
        self.assertIn("Page one content.\nPage two content.", "".join(called_args))

    def test_web_ui_pdf_ingestion_duplicate(self):
        mock_engine = MagicMock()
        mock_vector_tool = MagicMock()
        mock_vector_tool.is_document_ingested.return_value = True
        mock_engine.vector_tool = mock_vector_tool

        ui = WebUI(mock_engine)

        mock_file = MagicMock()
        mock_file.name = "test_doc.pdf"
        
        res = ui._process_file(mock_file)
        self.assertIn("already ingested", res)
        mock_vector_tool.add_documents.assert_not_called()

    @patch("nvidia_rag.api.server.engine")
    def test_api_server_pdf_ingestion(self, mock_engine):
        mock_engine.vector_tool.is_document_ingested.return_value = False

        from fastapi.testclient import TestClient
        client = TestClient(app)
        
        # Test PDF upload
        pdf_content = b"%PDF-1.4 mock pdf bytes"
        response = client.post(
            "/upload",
            files={"file": ("test.pdf", pdf_content, "application/pdf")}
        )

        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        self.assertEqual(json_data["status"], "success")
        self.assertEqual(json_data["chunks_added"], 1)  # Two pages fit in one chunk
        mock_engine.vector_tool.add_documents.assert_called_once()

    @patch("nvidia_rag.api.server.engine")
    def test_api_server_pdf_ingestion_duplicate(self, mock_engine):
        mock_engine.vector_tool.is_document_ingested.return_value = True

        from fastapi.testclient import TestClient
        client = TestClient(app)
        
        # Test PDF upload
        pdf_content = b"%PDF-1.4 mock pdf bytes"
        response = client.post(
            "/upload",
            files={"file": ("test.pdf", pdf_content, "application/pdf")}
        )

        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        self.assertEqual(json_data["status"], "skipped")
        self.assertEqual(json_data["detail"], "Document already ingested. Skipping.")
        mock_engine.vector_tool.add_documents.assert_not_called()


if __name__ == "__main__":
    unittest.main()
