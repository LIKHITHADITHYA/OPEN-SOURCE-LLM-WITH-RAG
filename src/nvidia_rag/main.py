import sys
import logging
from nvidia_rag.core.engine import RAGEngine
from nvidia_rag.ui.web_ui import WebUI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Main entry point for the application."""
    engine = RAGEngine()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        # Simple CLI mode
        while True:
            query = input("\nYou: ").strip()
            if query.lower() in ['exit', 'quit']:
                break
            resp, tokens, _ = engine.generate_response(query)
            print(f"Bot: {resp}\n(Tokens: {tokens})")
    else:
        # Default to Web UI
        ui = WebUI(engine)
        ui.launch()

if __name__ == "__main__":
    main()
