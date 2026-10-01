from app.graph import graph
from app.observability import get_langfuse_handler
from ingest_policy import ingest_document
from sync_documents import (
    get_current_hash,
    save_hash,
    sync_document,
)


def synchronize_document() -> None:
    """
    Check whether the PDF has changed when the application starts.

    If the document changed:
    1. Re-ingest the PDF.
    2. Update Qdrant.
    3. Save the new document hash.

    If the document has not changed:
    - Do not re-ingest.
    """

    print("\nChecking document for changes...")

    document_changed = sync_document()

    if not document_changed:
        print("No changes detected.")
        print("Qdrant is already up to date.")
        return

    print("Document changes detected.")
    print("Calling ingest_policy.py...")
    print("Updating Qdrant...")

    try:
        # Re-ingest the changed PDF
        ingest_document()

        # Save the hash only after successful ingestion
        current_hash = get_current_hash()
        save_hash(current_hash)

        print("Document hash updated successfully.")
        print("Qdrant updated successfully.")

    except Exception as exc:
        print(f"Document update failed: {exc}")
        raise


def ask_questions() -> None:
    """Start the interactive QA session."""

    print("\nEmployee AI QA")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:

        query = input("You: ").strip()

        if query.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not query:
            continue

        try:
            # ----------------------------------------------
            # Create Langfuse handler
            # ----------------------------------------------

            langfuse_handler = get_langfuse_handler()

            # ----------------------------------------------
            # Run QA graph
            # ----------------------------------------------

            result = graph.invoke(
                {
                    "query": query
                },
                config={
                    "callbacks": [langfuse_handler],
                    "run_name": "employee-ai-qa",
                    "metadata": {
                        "application": "employee-ai",
                        "environment": "development",
                    },
                },
            )

            # ----------------------------------------------
            # Display route
            # ----------------------------------------------

            print(
                f"\nRoute: {result.get('route', 'unknown')}"
            )

            # ----------------------------------------------
            # Display answer
            # ----------------------------------------------

            print(
                f"Answer: {result.get('answer', 'No answer generated.')}"
            )

            # ----------------------------------------------
            # Display sources
            # ----------------------------------------------

            sources = result.get("documents", [])

            if sources:

                print("\nSources:")

                for source in sources:
                    print(
                        f"- {source.get('source', 'unknown')} "
                        f"(page {source.get('page', 'unknown')})"
                    )

            print()

        except (
            KeyError,
            TypeError,
            ValueError,
            RuntimeError,
            FileNotFoundError,
        ) as exc:

            print(
                f"\nError: {exc}\n"
            )


def main() -> None:
    """
    Application entry point.

    The document is synchronized FIRST.
    Only after synchronization is complete
    do we start accepting questions.
    """

    # ======================================================
    # STEP 1: Synchronize document
    # ======================================================

    synchronize_document()

    # ======================================================
    # STEP 2: Start QA
    # ======================================================

    ask_questions()


if __name__ == "__main__":
    main()