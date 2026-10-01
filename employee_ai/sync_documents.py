import hashlib
from pathlib import Path

from app.config import settings


HASH_FILE = Path(__file__).resolve().parent / ".document_hash"


def calculate_file_hash(file_path: Path) -> str:
    """Calculate SHA-256 hash of a file."""

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        while chunk := file.read(1024 * 1024):
            sha256.update(chunk)

    return sha256.hexdigest()


def get_previous_hash() -> str | None:
    """Get the previously saved document hash."""

    if not HASH_FILE.exists():
        return None

    return HASH_FILE.read_text().strip()


def get_current_hash() -> str:
    """Calculate the current document hash."""

    document_path = settings.document_path

    if not document_path.exists():
        raise FileNotFoundError(
            f"Document not found: {document_path}"
        )

    return calculate_file_hash(document_path)


def save_hash(file_hash: str) -> None:
    """Save the latest document hash."""

    HASH_FILE.write_text(file_hash)


def document_has_changed() -> bool:
    """Return True if the document changed."""

    current_hash = get_current_hash()
    previous_hash = get_previous_hash()

    if previous_hash is None:
        return True

    return current_hash != previous_hash


def sync_document() -> bool:
    """Check whether the document has changed."""

    document_path = settings.document_path

    print(f"Checking document: {document_path}")

    return document_has_changed()


if __name__ == "__main__":
    changed = sync_document()

    if changed:
        print("Document changes detected.")
    else:
        print("No changes detected.")