"""
AstroRAG Python client
Usage:
    # Single question
    python client.py "What are the main Lyman-alpha escape mechanisms?"

    # Interactive mode
    python client.py
"""

import sys
import json
import requests

API_URL = "http://localhost:8000/ask"


def ask(question: str) -> None:
    """Send a question to the AstroRAG API and print the result."""
    try:
        response = requests.post(
            API_URL,
            json={"question": question},
            timeout=30
        )
        response.raise_for_status()
        data = response.json()

        print("\n" + "─" * 60)
        print(f"Q: {data['question']}")
        print("─" * 60)
        print(f"\n{data['answer']}\n")
        print("Sources:")
        for source in data["sources"]:
            print(f"  • {source.split('/')[-1].split(chr(92))[-1]}")
        print("─" * 60 + "\n")

    except requests.exceptions.ConnectionError:
        print("\nError: Could not connect to AstroRAG API.")
        print("Make sure the server is running: uvicorn api:app --reload\n")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"\nAPI error: {e}\n")
        sys.exit(1)


def interactive_mode() -> None:
    """Run an interactive question loop in the terminal."""
    print("\n🔭 AstroRAG Terminal Client")
    print("Type your question and press Enter. Type 'exit' to quit.\n")
    while True:
        try:
            question = input("Ask: ").strip()
            if question.lower() in ("exit", "quit", "q"):
                print("Goodbye!")
                break
            if not question:
                continue
            ask(question)
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Single question passed as argument
        ask(" ".join(sys.argv[1:]))
    else:
        # Interactive mode
        interactive_mode()