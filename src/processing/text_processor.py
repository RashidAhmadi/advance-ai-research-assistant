import re
import unicodedata
from pathlib import Path


# Default chunking configuration
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150


def clean_text(text: str) -> str:
    """
    Normalize extracted scientific PDF text.

    Preserves original scientific terminology and symbols.
    """

    if not isinstance(text, str):
        raise TypeError("Input text must be a string.")

    # Normalize Unicode characters
    text = unicodedata.normalize("NFKC", text)

    # Remove soft hyphens
    text = text.replace("\u00ad", "")

    # Join words split by a hyphen at a line break
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)

    # Normalize line breaks and whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def create_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> list:
    """
    Split text into overlapping character-based chunks.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be non-negative and smaller than chunk_size."
        )

    text = clean_text(text)

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):

        end = min(start + chunk_size, len(text))

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - chunk_overlap

    return chunks


def process_document(
    extracted_document: dict,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> list:
    """
    Convert an extracted PDF document into page-level chunks.

    Each chunk preserves its original document and page information.
    """

    filename = extracted_document["filename"]

    document_id = Path(filename).stem

    processed_chunks = []

    for page in extracted_document["pages"]:

        page_number = page["page_number"]

        page_chunks = create_chunks(
            page["text"],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        for chunk_index, chunk_text in enumerate(page_chunks, start=1):

            chunk_id = (
                f"{document_id}_p{page_number}_c{chunk_index}"
            )

            processed_chunks.append({
                "chunk_id": chunk_id,
                "document_id": document_id,
                "filename": filename,
                "page_number": page_number,
                "chunk_index": chunk_index,
                "text": chunk_text,
                "character_count": len(chunk_text),
                "word_count": len(chunk_text.split())
            })

    return processed_chunks
