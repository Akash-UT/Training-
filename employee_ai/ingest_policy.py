import hashlib
from pathlib import Path

import fitz
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient

from app.config import settings


def clean_text(value) -> str:
    """Clean extracted PDF/table cell text."""

    if value is None:
        return ""

    return " ".join(
        str(value)
        .replace("\n", " ")
        .split()
    )


def serialize_table(table, page_number: int) -> str:
    """
    Convert a PDF table into row-oriented text.

    If the PDF provides reliable table headers, they are used.
    If the table has no headers, generic column names are used.
    We do NOT guess what the columns mean.
    """

    rows = table.extract()

    if not rows:
        return ""

    cleaned_rows = []

    for row in rows:
        cleaned_row = [
            clean_text(cell)
            for cell in row
        ]

        if any(cleaned_row):
            cleaned_rows.append(cleaned_row)

    if not cleaned_rows:
        return ""

    header_object = getattr(table, "header", None)

    headers = []

    if header_object is not None:
        header_names = getattr(
            header_object,
            "names",
            None,
        )

        if header_names:
            headers = [
                clean_text(header)
                for header in header_names
            ]

    has_header = bool(headers) and any(headers)

    if not has_header:
        max_columns = max(
            len(row)
            for row in cleaned_rows
        )

        headers = [
            f"Column {index + 1}"
            for index in range(max_columns)
        ]

    output = [
        f"Table from page {page_number}:"
    ]

    if has_header:
        output.append(
            "Columns: " + " | ".join(headers)
        )

        output.append("")

        data_rows = cleaned_rows[1:]

    else:
        output.append(
            "This table does not contain identified headers."
        )

        output.append("")

        data_rows = cleaned_rows

    for row_number, row in enumerate(
        data_rows,
        start=1,
    ):

        output.append(
            f"Table Row {row_number}:"
        )

        for column_number, value in enumerate(row):

            if not value:
                continue

            if column_number < len(headers):
                column_name = headers[column_number]

            else:
                column_name = (
                    f"Column {column_number + 1}"
                )

            output.append(
                f"{column_name}: {value}"
            )

        output.append("")

    return "\n".join(output).strip()


def extract_pdf_documents(
    pdf_path: Path,
) -> list[Document]:
    """
    Extract PDF text and tables.

    Each page is preserved as a complete document.
    Detected tables are stored separately.
    """

    pdf = fitz.open(pdf_path)

    documents = []

    try:

        for page_index in range(len(pdf)):

            page = pdf[page_index]

            page_number = page_index + 1

            # -------------------------------------------------
            # 1. Extract complete page text
            # -------------------------------------------------

            page_text = page.get_text("text")

            if page_text and page_text.strip():

                page_text = page_text.strip()

                documents.append(
                    Document(
                        page_content=(
                            f"Page {page_number}\n\n"
                            f"{page_text}"
                        ),
                        metadata={
                            "source_file": pdf_path.name,
                            "page_number": page_number,
                            "content_type": "page",
                        },
                    )
                )

            # -------------------------------------------------
            # 2. Detect tables
            # -------------------------------------------------

            try:

                table_finder = page.find_tables()

                tables = table_finder.tables

            except Exception as exc:  # noqa: BLE001

                print(
                    f"Warning: Could not detect tables "
                    f"on page {page_number}: {exc}"
                )

                tables = []

            # -------------------------------------------------
            # 3. Store detected tables
            # -------------------------------------------------

            for table_index, table in enumerate(
                tables,
                start=1,
            ):

                table_text = serialize_table(
                    table,
                    page_number=page_number,
                )

                if not table_text:
                    continue

                documents.append(
                    Document(
                        page_content=table_text,
                        metadata={
                            "source_file": pdf_path.name,
                            "page_number": page_number,
                            "content_type": "table",
                            "table_number": table_index,
                        },
                    )
                )

    finally:
        pdf.close()

    return documents

HASH_FILE = Path(".document_hash")


def calculate_file_hash(file_path: Path) -> str:
    """Calculate SHA-256 hash of the document."""

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        while chunk := file.read(1024 * 1024):
            sha256.update(chunk)

    return sha256.hexdigest()


def save_document_hash(file_path: Path) -> None:
    """Save the hash of the successfully ingested document."""

    file_hash = calculate_file_hash(file_path)
    HASH_FILE.write_text(file_hash)

    print("Document hash updated successfully.")


def ingest_document() -> None:
    """
    Extract the document, create LOCAL embeddings,
    and store them in Qdrant.
    """

    document_path = settings.document_path

    if not document_path.exists():

        raise FileNotFoundError(
            f"Document not found: {document_path}"
        )

    print(
        f"Reading document: {document_path}"
    )

    # ---------------------------------------------------------
    # Extract PDF
    # ---------------------------------------------------------

    documents = extract_pdf_documents(
        document_path
    )

    if not documents:

        raise RuntimeError(
            "No text or tables were extracted "
            "from the document."
        )

    page_documents = [
        document
        for document in documents
        if document.metadata.get(
            "content_type"
        ) == "page"
    ]

    table_documents = [
        document
        for document in documents
        if document.metadata.get(
            "content_type"
        ) == "table"
    ]

    print(
        f"Extracted page documents: "
        f"{len(page_documents)}"
    )

    print(
        f"Extracted table documents: "
        f"{len(table_documents)}"
    )

    # ---------------------------------------------------------
    # Split into normal chunks
    # ---------------------------------------------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    chunks = splitter.split_documents(
        documents
    )

    print(
        f"Created {len(chunks)} chunks."
    )


    documents_to_index = (
        documents
        + chunks
    )

    print(
        f"Total documents to index: "
        f"{len(documents_to_index)}"
    )

    

    print(
        "Loading local embedding model..."
    )

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={
            "device": "cpu",
        },
        encode_kwargs={
            "normalize_embeddings": True,
        },
    )

    print(
        "Local embedding model loaded."
    )

    # ---------------------------------------------------------
    # Connect to Qdrant
    # ---------------------------------------------------------

    qdrant_client = QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
    )

    # ---------------------------------------------------------
    # Delete old collection
    # ---------------------------------------------------------

    collections = qdrant_client.get_collections()

    existing_collections = {
        collection.name
        for collection in collections.collections
    }

    if settings.qdrant_collection in existing_collections:

        print(
            f"Deleting existing Qdrant collection: "
            f"{settings.qdrant_collection}"
        )

        qdrant_client.delete_collection(
            collection_name=settings.qdrant_collection
        )

    # ---------------------------------------------------------
    # Store local embeddings in Qdrant
    # ---------------------------------------------------------

    print(
        "Creating Qdrant collection and "
        "storing local embeddings..."
    )

    QdrantVectorStore.from_documents(
        documents=documents_to_index,
        embedding=embeddings,
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        collection_name=settings.qdrant_collection,
    )

    print(
        "\nDocument ingestion completed successfully."
    )

    print(
        f"Total documents stored: "
        f"{len(documents_to_index)}"
    )

    save_document_hash(document_path)

if __name__ == "__main__":
    ingest_document()