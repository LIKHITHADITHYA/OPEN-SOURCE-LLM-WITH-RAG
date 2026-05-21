import unittest
from unittest.mock import MagicMock
from nvidia_rag.core.engine import RAGEngine

class TestRAGEngine(unittest.TestCase):
    def setUp(self):
        self.engine = RAGEngine(client=MagicMock())

    def test_build_rag_prompt_with_results(self):
        query = "test query"
        results = "Result 1: content"
        prompt = self.engine._build_rag_prompt(query, results)
        self.assertIn("Search Results:", prompt)
        self.assertIn(results, prompt)

    def test_build_rag_prompt_no_results(self):
        query = "test query"
        results = "No search results found."
        prompt = self.engine._build_rag_prompt(query, results)
        self.assertIn("unable to retrieve", prompt)

if __name__ == "__main__":
    unittest.main()
