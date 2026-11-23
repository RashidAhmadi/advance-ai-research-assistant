
from contextlib import asynccontextmanager
from pathlib import Path
import sys
import json
import uuid
import re

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
# ---------------------------------------------------------
# Project imports
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.embeddings.embedding_model import (
    EmbeddingModel,
    ChromaVectorStore
)

from src.retrieval.hybrid_retriever import HybridRetriever

from src.generation.llm_provider import OllamaProvider
from src.generation.rag_pipeline import RAGPipeline

from src.ingestion.pdf_extractor import extract_pdf
from src.processing.text_processor import process_document


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"

CHUNKS_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "document_chunks.jsonl"
)

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB


# ---------------------------------------------------------
# Shared application components
# ---------------------------------------------------------

rag_pipeline = None
hybrid_retriever = None
embedding_model = None
vector_store = None


@asynccontextmanager
async def lifespan(app: FastAPI):

    global rag_pipeline
    global hybrid_retriever
    global embedding_model
    global vector_store

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    embedding_model = EmbeddingModel()

    vector_store = ChromaVectorStore(
        persist_directory=PROJECT_ROOT / "data" / "chroma_db",
        collection_name="research_papers"
    )

    hybrid_retriever = HybridRetriever(
        embedding_model=embedding_model,
        vector_store=vector_store,
        chunks_file=CHUNKS_FILE
    )

    llm_provider = OllamaProvider(
        model_name="qwen2.5:3b"
    )

    rag_pipeline = RAGPipeline(
        retriever=hybrid_retriever,
        llm_provider=llm_provider
    )

    print("Advance AI Research Assistant backend initialized.")

    yield

    rag_pipeline = None
    hybrid_retriever = None
    embedding_model = None
    vector_store = None


# ---------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------

app = FastAPI(
    title="Advance AI Research Assistant",
    description=(
        "Scientific PDF research assistant using "
        "hybrid retrieval and retrieval-augmented generation."
    ),
    version="1.0.0",
    lifespan=lifespan
)
# Allow requests from the Next.js frontend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ---------------------------------------------------------
# Request and response models
# ---------------------------------------------------------

class QueryRequest(BaseModel):

    question: str = Field(
        ...,
        min_length=3,
        max_length=2000
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=10
    )
class RenameDocumentRequest(BaseModel):

    filename: str = Field(
        ...,
        min_length=1,
        max_length=255
    )

class SourceReference(BaseModel):

    source_id: str
    filename: str
    page_number: int
    chunk_id: str


class QueryResponse(BaseModel):

    question: str
    answer: str
    sources: list[SourceReference]
    execution_time: float
    citation_warning: bool


# ---------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "project": "Advance AI Research Assistant",
        "rag_pipeline_initialized": rag_pipeline is not None,
        "retriever_initialized": hybrid_retriever is not None,
        "embedding_model_initialized": embedding_model is not None,
        "vector_store_initialized": vector_store is not None
    }


# ---------------------------------------------------------
# Document management endpoint
# ---------------------------------------------------------

@app.get("/documents")
def list_documents():
    """
    Return all indexed documents and their statistics.

    Document information is reconstructed from the persistent
    JSONL chunk collection. This means the document library
    remains available after restarting the backend.
    """

    if not CHUNKS_FILE.exists():
        return {
            "document_count": 0,
            "total_chunks": 0,
            "total_pages": 0,
            "documents": []
        }

    documents = {}

    # Read the persistent chunk collection.
    with CHUNKS_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            chunk = json.loads(line)

            document_id = chunk["document_id"]

            # Create a document entry the first time we
            # encounter one of its chunks.
            if document_id not in documents:
                documents[document_id] = {
                    "document_id": document_id,
                    "filename": chunk["filename"],
                    "pages": set(),
                    "chunk_count": 0
                }

            # Collect unique page numbers and count chunks.
            page_number = chunk.get("page_number")

            if page_number is not None:
                documents[document_id]["pages"].add(
                    page_number
                )

            documents[document_id]["chunk_count"] += 1

    # Convert the collected information into API records.
    document_list = []

    for document_id, document in documents.items():

        # Uploaded PDFs are stored as:
        # UUID_original_filename.pdf
        stored_files = list(
            UPLOAD_DIR.glob(f"{document_id}_*.pdf")
        )

        page_numbers = sorted(document["pages"])

        document_list.append({
            "document_id": document_id,
            "filename": document["filename"],
            "page_count": len(page_numbers),
            "chunk_count": document["chunk_count"],
            "is_uploaded": len(stored_files) > 0,
            "status": "indexed"
        })

    # Display documents alphabetically.
    document_list.sort(
        key=lambda item: item["filename"].lower()
    )

    return {
        "document_count": len(document_list),
        "total_chunks": sum(
            item["chunk_count"] for item in document_list
        ),
        "total_pages": sum(
            item["page_count"] for item in document_list
        ),
        "documents": document_list
    }
# ---------------------------------------------------------
# Document management helpers
# ---------------------------------------------------------

def read_chunks_file():
    """Load all indexed chunks from the JSONL file."""

    if not CHUNKS_FILE.exists():
        return []

    chunks = []

    with CHUNKS_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                chunks.append(json.loads(line))

    return chunks


def write_chunks_file(chunks):
    """
    Safely replace the JSONL file.

    A temporary file is written first so that an interrupted
    write is less likely to corrupt the existing corpus.
    """

    CHUNKS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    temporary_file = CHUNKS_FILE.with_suffix(".jsonl.tmp")

    with temporary_file.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(
                json.dumps(chunk, ensure_ascii=False) + "\n"
            )

    temporary_file.replace(CHUNKS_FILE)


def validate_pdf_filename(filename: str) -> str:
    """Validate and normalize a display filename."""

    filename = filename.strip()

    if not filename:
        raise HTTPException(
            status_code=400,
            detail="A filename is required."
        )

    # Prevent directory traversal and path-based filenames.
    if "/" in filename or "\\" in filename:
        raise HTTPException(
            status_code=400,
            detail="Please provide a filename without a directory path."
        )

    if Path(filename).suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=400,
            detail="The filename must have a .pdf extension."
        )

    if len(filename) > 255:
        raise HTTPException(
            status_code=400,
            detail="The filename must not exceed 255 characters."
        )

    return filename

# ---------------------------------------------------------
# Rename an indexed document
# ---------------------------------------------------------

@app.patch("/documents/{document_id}")
def rename_document(
    document_id: str,
    request: RenameDocumentRequest
):
    """
    Rename a document in the library.

    The document ID, chunks, embeddings, and original
    uploaded PDF storage name remain unchanged.
    """

    if hybrid_retriever is None or vector_store is None:
        raise HTTPException(
            status_code=503,
            detail="Document management components are not ready."
        )

    new_filename = validate_pdf_filename(request.filename)

    chunks = read_chunks_file()

    document_chunks = [
        chunk
        for chunk in chunks
        if chunk.get("document_id") == document_id
    ]

    if not document_chunks:
        raise HTTPException(
            status_code=404,
            detail="Document not found."
        )

    old_filename = document_chunks[0]["filename"]

    if old_filename == new_filename:
        return {
            "status": "success",
            "message": "The filename has not changed.",
            "document_id": document_id,
            "filename": new_filename
        }

    # Update the filename in the persistent JSONL corpus.
    updated_chunks = []

    for chunk in chunks:
        updated_chunk = dict(chunk)

        if updated_chunk.get("document_id") == document_id:
            updated_chunk["filename"] = new_filename

        updated_chunks.append(updated_chunk)

    try:

        write_chunks_file(updated_chunks)

        # Update the filename metadata in ChromaDB.
        updated_vectors = vector_store.update_document_filename(
            document_id=document_id,
            new_filename=new_filename
        )

        if updated_vectors == 0:
            raise RuntimeError(
                "No matching vectors were found in ChromaDB."
            )

        # Update the in-memory BM25 corpus.
        for chunk in hybrid_retriever.chunks:
            if chunk.get("document_id") == document_id:
                chunk["filename"] = new_filename

        return {
            "status": "success",
            "message": "Document renamed successfully.",
            "document_id": document_id,
            "old_filename": old_filename,
            "new_filename": new_filename,
            "updated_vectors": updated_vectors
        }

    except Exception as error:

        # Restore the original JSONL corpus if the operation fails.
        try:
            write_chunks_file(chunks)
        except Exception as rollback_error:
            print(
                "Rename rollback failed:",
                rollback_error
            )

        raise HTTPException(
            status_code=500,
            detail=f"Document rename failed: {str(error)}"
        ) from error
# ---------------------------------------------------------
# Delete an indexed document
# ---------------------------------------------------------

@app.delete("/documents/{document_id}")
def delete_document(document_id: str):
    """
    Delete a document from:

    1. Persistent JSONL chunks
    2. ChromaDB vector collection
    3. In-memory BM25 index
    4. Uploaded PDF storage, if applicable

    Original PDFs outside data/uploads are not deleted.
    """

    if hybrid_retriever is None or vector_store is None:
        raise HTTPException(
            status_code=503,
            detail="Document management components are not ready."
        )

    chunks = read_chunks_file()

    document_chunks = [
        chunk
        for chunk in chunks
        if chunk.get("document_id") == document_id
    ]

    if not document_chunks:
        raise HTTPException(
            status_code=404,
            detail="Document not found."
        )

    filename = document_chunks[0]["filename"]

    remaining_chunks = [
        chunk
        for chunk in chunks
        if chunk.get("document_id") != document_id
    ]

    removed_chunk_count = len(document_chunks)

    try:

        # 1. Update the persistent JSONL corpus.
        write_chunks_file(remaining_chunks)

        # 2. Delete all corresponding vectors from ChromaDB.
        deleted_vectors = vector_store.delete_document(
            document_id=document_id
        )

        if deleted_vectors == 0:
            raise RuntimeError(
                "No matching vectors were found in ChromaDB."
            )

    except Exception as error:

        # Restore JSONL if vector deletion fails.
        try:
            write_chunks_file(chunks)
        except Exception as rollback_error:
            print(
                "Delete rollback failed:",
                rollback_error
            )

        raise HTTPException(
            status_code=500,
            detail=f"Document deletion failed: {str(error)}"
        ) from error

    # 3. Remove the document from the in-memory BM25 index.
    removed_from_bm25 = hybrid_retriever.remove_document(
        document_id
    )

    # 4. Delete stored uploaded PDF copies, if present.
    deleted_files = 0

    stored_files = list(
        UPLOAD_DIR.glob(f"{document_id}_*.pdf")
    )

    for stored_file in stored_files:
        try:
            stored_file.unlink()
            deleted_files += 1
        except OSError as error:
            print(
                f"Could not delete uploaded PDF {stored_file}:",
                error
            )

    return {
        "status": "success",
        "message": "Document deleted successfully.",
        "document_id": document_id,
        "filename": filename,
        "removed_chunks": removed_chunk_count,
        "deleted_vectors": deleted_vectors,
        "removed_from_bm25": removed_from_bm25,
        "deleted_uploaded_files": deleted_files
    }
# ---------------------------------------------------------
# Research question endpoint
# ---------------------------------------------------------

@app.post("/query", response_model=QueryResponse)
def answer_question(request: QueryRequest):

    if rag_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="The RAG pipeline is not initialized."
        )

    try:

        result = rag_pipeline.run(
            question=request.question,
            final_top_k=request.top_k
        )

        return QueryResponse(
            question=result["question"],
            answer=result["answer"],
            sources=result["sources"],
            execution_time=round(
                result["execution_time"], 3
            ),
            citation_warning=result["citation_warning"]
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Question processing failed: {str(error)}"
        ) from error


# ---------------------------------------------------------
# PDF upload endpoint
# ---------------------------------------------------------

@app.post("/documents/upload")
def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF, extract its text, create chunks,
    generate embeddings, and update both retrieval indexes.
    """

    if any(
        component is None
        for component in (
            hybrid_retriever,
            embedding_model,
            vector_store
        )
    ):
        raise HTTPException(
            status_code=503,
            detail="The document processing components are not ready."
        )

    # Validate the filename and extension.
    original_filename = Path(file.filename or "").name

    if not original_filename:
        raise HTTPException(
            status_code=400,
            detail="A filename is required."
        )

    if Path(original_filename).suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    # Read uploaded file.
    file_content = file.file.read()

    if not file_content:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty."
        )

    if len(file_content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="The PDF exceeds the 50 MB upload limit."
        )

    # Check PDF signature.
    if not file_content.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid PDF."
        )

    # Create a unique internal filename.
    document_id = uuid.uuid4().hex

    safe_name = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        original_filename
    )

    stored_filename = f"{document_id}_{safe_name}"
    saved_path = UPLOAD_DIR / stored_filename

    saved_path.write_bytes(file_content)

    inserted_ids = []
    jsonl_updated = False

    try:

        # 1. Extract PDF text using PyMuPDF.
        extracted_document = extract_pdf(saved_path)

        # 2. Create page-level chunks.
        chunks = process_document(extracted_document)

        if not chunks:
            raise ValueError(
                "No extractable text was found in the PDF."
            )

        # 3. Preserve the original filename in source metadata.
        #    Use a UUID-based document ID to avoid collisions.
        for chunk in chunks:

            chunk["document_id"] = document_id
            chunk["filename"] = original_filename

            chunk["chunk_id"] = (
                f"{document_id}_p"
                f"{chunk['page_number']}_c"
                f"{chunk['chunk_index']}"
            )

        # 4. Generate embeddings.
        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        embeddings = embedding_model.encode(texts)

        # 5. Check chunk IDs before inserting.
        existing_ids = {
            chunk["chunk_id"]
            for chunk in hybrid_retriever.chunks
        }

        new_chunks = [
            chunk for chunk in chunks
            if chunk["chunk_id"] not in existing_ids
        ]

        if not new_chunks:
            raise ValueError(
                "No new chunks were available for indexing."
            )

        new_ids = [
            chunk["chunk_id"]
            for chunk in new_chunks
        ]

        # Keep embeddings aligned with the selected chunks.
        new_embeddings = [
            embeddings[index]
            for index, chunk in enumerate(chunks)
            if chunk["chunk_id"] in set(new_ids)
        ]

        # 6. Persist chunks in the JSONL corpus.
        with CHUNKS_FILE.open("a", encoding="utf-8") as output_file:

            for chunk in new_chunks:
                output_file.write(
                    json.dumps(chunk, ensure_ascii=False) + "\n"
                )

        jsonl_updated = True

        # 7. Insert vectors into ChromaDB.
        vector_store.add_chunks(
            new_chunks,
            new_embeddings
        )

        inserted_ids = new_ids

        # 8. Update the in-memory BM25 index.
        added_count = hybrid_retriever.add_chunks(new_chunks)

        if added_count != len(new_chunks):
            raise RuntimeError(
                "Not all uploaded chunks were added to BM25."
            )

        return {
            "status": "success",
            "message": "PDF uploaded and indexed successfully.",
            "original_filename": original_filename,
            "document_id": document_id,
            "pages": extracted_document["page_count"],
            "chunks_created": len(new_chunks),
            "chunks_added_to_bm25": added_count,
            "vectors_stored": len(new_embeddings)
        }

    except Exception as error:

        # Best-effort rollback of inserted vectors.
        if inserted_ids:
            try:
                vector_store.collection.delete(
                    ids=inserted_ids
                )
            except Exception as rollback_error:
                print(
                    "Vector rollback failed:",
                    rollback_error
                )

        # Restore the JSONL corpus from the current retriever.
        if jsonl_updated:
            try:
                with CHUNKS_FILE.open(
                    "w",
                    encoding="utf-8"
                ) as output_file:

                    for chunk in hybrid_retriever.chunks:
                        output_file.write(
                            json.dumps(
                                chunk,
                                ensure_ascii=False
                            ) + "\n"
                        )

            except Exception as rollback_error:
                print(
                    "JSONL rollback failed:",
                    rollback_error
                )

        if isinstance(error, ValueError):
            raise HTTPException(
                status_code=422,
                detail=str(error)
            ) from error

        raise HTTPException(
            status_code=500,
            detail=f"PDF processing failed: {str(error)}"
        ) from error
