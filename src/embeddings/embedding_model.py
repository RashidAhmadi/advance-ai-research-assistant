from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """
    Generate text embeddings using Sentence Transformers.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None
    ):
        self.model_name = model_name

        # Load the model on the selected device
        self.model = SentenceTransformer(
            model_name,
            device=device
        )

        self.embedding_dimension = self.model.get_embedding_dimension()

    def encode(self, texts: list[str]) -> list[list[float]]:
        """
        Convert a list of texts into normalized embeddings.
        """

        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        return embeddings.tolist()


class ChromaVectorStore:
    """
    Manage persistent vector storage using ChromaDB.
    """

    def __init__(
        self,
        persist_directory: str | Path,
        collection_name: str = "research_papers"
    ):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # Connect to the existing persistent database
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory)
        )

        # Reuse the existing collection if it exists
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(
        self,
        chunks: list[dict],
        embeddings: list[list[float]],
        batch_size: int = 100
    ) -> int:
        """
        Insert or update chunks and their embeddings.

        ChromaDB upsert prevents duplicate IDs.
        """

        if len(chunks) != len(embeddings):
            raise ValueError(
                "Number of chunks and embeddings must match."
            )

        if not chunks:
            return 0

        for start in range(0, len(chunks), batch_size):

            batch_chunks = chunks[start:start + batch_size]
            batch_embeddings = embeddings[start:start + batch_size]

            self.collection.upsert(
                ids=[chunk["chunk_id"] for chunk in batch_chunks],

                embeddings=batch_embeddings,

                documents=[
                    chunk["text"] for chunk in batch_chunks
                ],

                metadatas=[
                    {
                        "document_id": chunk["document_id"],
                        "filename": chunk["filename"],
                        "page_number": chunk["page_number"],
                        "chunk_index": chunk["chunk_index"],
                        "word_count": chunk["word_count"]
                    }
                    for chunk in batch_chunks
                ]
            )

        return len(chunks)

    def count(self) -> int:
        """Return the number of stored vectors."""
        return self.collection.count()

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5
    ) -> dict:
        """Search for semantically similar passages."""

        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

    def update_document_filename(
        self,
        document_id: str,
        new_filename: str
    ) -> int:
        """
        Update the filename metadata of all chunks
        belonging to a document in ChromaDB.
        """
    
        results = self.collection.get(
            where={"document_id": document_id},
            include=["metadatas"]
        )
    
        ids = results.get("ids", [])
        metadatas = results.get("metadatas", [])
    
        if not ids:
            return 0
    
        updated_metadatas = []
    
        for metadata in metadatas:
            updated_metadata = dict(metadata)
            updated_metadata["filename"] = new_filename
            updated_metadatas.append(updated_metadata)
    
        self.collection.update(
            ids=ids,
            metadatas=updated_metadatas
        )
    
        return len(ids)
    
    
    def delete_document(self, document_id: str) -> int:
        """
        Delete all vectors belonging to a document
        from ChromaDB.
        """
    
        results = self.collection.get(
            where={"document_id": document_id},
            include=[]
        )
    
        ids = results.get("ids", [])
    
        if not ids:
            return 0
    
        self.collection.delete(ids=ids)
    
        return len(ids)