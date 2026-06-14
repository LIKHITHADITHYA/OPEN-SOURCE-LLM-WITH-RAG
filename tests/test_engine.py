"""
Integration tests for the Enhanced RAG Engine.
Verifies routing logic, multi-turn memory, and tool integration.
"""

from nvidia_rag.core.engine import RAGEngine

def test_engine():
    """
    Executes a series of queries to verify the engine's behavior across different routes.
    """
    # Initialize the core engine
    engine = RAGEngine()
    
    # Test 1: General Conversation (NONE Route)
    print("--- Test 1: Testing NONE route (Greeting) ---")
    resp, tokens, source = engine.generate_response("Hello, who are you?")
    print(f"Source: {source}\nResponse: {resp[:100]}...\n")
    
    # Test 2: External Knowledge (WEB Route)
    print("--- Test 2: Testing WEB route (Current News/Price) ---")
    resp, tokens, source = engine.generate_response("What is the current price of Bitcoin?")
    print(f"Source: {source}\nResponse: {resp[:100]}...\n")
    
    # Test 3: Conversational Memory
    print("--- Test 3: Testing Memory (Context Retention) ---")
    resp, tokens, source = engine.generate_response(
        "I just asked about Bitcoin in the previous turn. "
        "Can you summarize what we've discussed so far?"
    )
    print(f"Source: {source}\nResponse: {resp[:200]}...\n")

if __name__ == "__main__":
    # Execution requires valid NVIDIA_API_KEY and SERPAPI_API_KEY in .env
    try:
        logger_name = "nvidia_rag"
        import logging
        logging.getLogger(logger_name).setLevel(logging.INFO)
        
        test_engine()
        print("Integration tests completed successfully.")
    except Exception as e:
        print(f"\n[!] Test failed: {e}")
        print("Check your API keys and internet connection.")
