"""
Application Entry Point.
Provides a unified interface to launch the system in CLI, API, or Web UI mode.
"""

import sys
import logging
from nvidia_rag.core.engine import RAGEngine
from nvidia_rag.ui.web_ui import WebUI

# Configure centralized logging for the entire application
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
# Silence verbose third-party libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

logger = logging.getLogger("nvidia_rag.main")


def main():
    """
    Main execution controller.
    Parses command-line arguments to decide which interface to launch.
    """
    # Instantiate the shared RAG engine
    try:
        engine = RAGEngine()
    except Exception as e:  # noqa: BLE001
        logger.error("Failed to initialize RAGEngine: %s", e)
        sys.exit(1)

    # Check for CLI flags
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()

        if arg == "--cli":
            # Silence all runtime logs in CLI mode for a clean chat experience
            logging.disable(logging.CRITICAL)

            # Launch terminal-based chat interface
            print("\n--- Welcome to NVIDIA RAG CLI ---")
            print("Type 'exit' or 'quit' to end the session.\n")

            while True:
                try:
                    query = input("You: ").strip()
                    if not query:
                        continue
                    if query.lower() in ['exit', 'quit']:
                        break

                    # Generate and display response
                    resp, tokens, source = engine.generate_response(query)
                    print(f"\nBot: {resp}\n")
                except KeyboardInterrupt:
                    break
            print("\nGoodbye!")

        else:
            # Display usage instructions for invalid arguments
            print(f"Error: Unknown argument '{arg}'")
            print("\nUsage:")
            print("  nvidia-rag          : Launch the Gradio Web UI (Default)")
            print("  nvidia-rag --cli    : Launch the Terminal interface")
            sys.exit(1)

    else:
        # Default behavior: Launch the Gradio web interface
        logger.info("Launching Web UI interface...")
        ui = WebUI(engine)
        ui.launch()


if __name__ == "__main__":
    main()
