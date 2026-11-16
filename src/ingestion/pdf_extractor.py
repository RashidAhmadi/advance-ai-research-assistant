
from pathlib import Path
import fitz


def extract_pdf(pdf_path: str | Path) -> dict:
    """
    Validate a PDF and extract its metadata and page-level text.

    Args:
        pdf_path: Path to the PDF document.

    Returns:
        Dictionary containing document metadata and extracted pages.
    """

    pdf_path = Path(pdf_path)

    # Validate the input file.
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("The file must have a PDF extension.")

    if pdf_path.stat().st_size == 0:
        raise ValueError("The PDF file is empty.")

    pages = []

    try:
        with fitz.open(pdf_path) as document:

            if document.is_encrypted:
                raise ValueError("Password-protected PDFs are not supported.")

            if document.page_count == 0:
                raise ValueError("The PDF contains no pages.")

            metadata = document.metadata or {}

            for page_index, page in enumerate(document):

                text = page.get_text("text", sort=True)

                pages.append({
                    "page_number": page_index + 1,
                    "text": text,
                    "character_count": len(text),
                    "word_count": len(text.split()),
                    "is_empty": not bool(text.strip())
                })

            return {
                "filename": pdf_path.name,
                "file_size_bytes": pdf_path.stat().st_size,
                "page_count": document.page_count,
                "metadata": metadata,
                "total_characters": sum(
                    page["character_count"] for page in pages
                ),
                "total_words": sum(
                    page["word_count"] for page in pages
                ),
                "pages": pages,
                "status": "success"
            }

    except Exception as error:
        raise RuntimeError(
            f"Could not extract PDF {pdf_path.name}: {error}"
        ) from error
