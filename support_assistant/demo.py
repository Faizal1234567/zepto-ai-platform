"""Run deterministic, local Module 3 demonstration calls without an HTTP server."""
from __future__ import annotations

from main import AskRequest, ask, service


def main() -> None:
    _, collection = service()
    print(f"ChromaDB collection contains {collection.count()} embedded policy chunks.")
    for query in ["What is the delivery fee for an order below INR 149?", "Tell me a joke."]:
        print(ask(AskRequest(query=query)).model_dump_json())


if __name__ == "__main__":
    main()
