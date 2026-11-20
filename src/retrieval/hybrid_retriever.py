import json
import re
from pathlib import Path
from collections import defaultdict

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from src.embeddings.embedding_model import (
    EmbeddingModel,
    ChromaVectorStore
)


# Project configuration
PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHUNKS_FILE = PROJECT_ROOT / "data" / "processed" / "document_chunks.jsonl"
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma_db"

RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

BM25_TOP_K = 10
SEMANTIC_TOP_K = 10
RRF_TOP_K = 10
FINAL_TOP_K = 5
RRF_K = 60


class HybridRetriever:
    """
    Combine BM25, semantic retrieval, RRF fusion,
    and Cross-Encoder reranking.
    """

    def __init__(
        self,
        embedding_model=None,
        vector_store=None,
        reranker=None,
        chunks_file: str | Path = CHUNKS_FILE,
        reranker_model_name: str = RERANKER_MODEL_NAME
    ):

        self.chunks_file = Path(chunks_file)

        if not self.chunks_file.exists():
            raise FileNotFoundError(
                f"Chunk file not found: {self.chunks_file}"
            )

        # Load the existing document chunks
        with self.chunks_file.open("r", encoding="utf-8") as file:
            self.chunks = [
                json.loads(line)
                for line in file
                if line.strip()
            ]

        if not self.chunks:
            raise ValueError("No document chunks were found.")

        # Initialize or reuse the embedding model
        self.embedding_model = (
            embedding_model
            if embedding_model is not None
            else EmbeddingModel()
        )

        # Initialize or reuse the ChromaDB connection
        self.vector_store = (
            vector_store
            if vector_store is not None
            else ChromaVectorStore(CHROMA_DIR)
        )

        # Create the BM25 index
        self.tokenized_corpus = [
            self.tokenize(chunk["text"])
            for chunk in self.chunks
        ]

        self.bm25 = BM25Okapi(self.tokenized_corpus)

        # Initialize or reuse the Cross-Encoder
        self.reranker = (
            reranker
            if reranker is not None
            else CrossEncoder(
                reranker_model_name,
                device="cpu",
                max_length=512
            )
        )

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Convert text into lowercase word tokens."""
        return re.findall(r"\b\w+\b", text.lower())

    def bm25_search(
        self,
        query: str,
        top_k: int = BM25_TOP_K
    ) -> list:
        """Retrieve passages using BM25 keyword matching."""

        query_tokens = self.tokenize(query)

        scores = self.bm25.get_scores(query_tokens)

        ranked_indices = scores.argsort()[::-1][:top_k]

        results = []

        for index in ranked_indices:

            chunk = self.chunks[index]

            results.append({
                "chunk_id": chunk["chunk_id"],
                "score": float(scores[index]),
                "text": chunk["text"],
                "metadata": {
                    "filename": chunk["filename"],
                    "page_number": chunk["page_number"],
                    "document_id": chunk["document_id"],
                    "chunk_index": chunk["chunk_index"]
                }
            })

        return results

    def semantic_search(
        self,
        query: str,
        top_k: int = SEMANTIC_TOP_K
    ) -> list:
        """Retrieve passages using ChromaDB semantic similarity."""

        collection_count = self.vector_store.count()

        if collection_count == 0:
            return []

        query_embedding = self.embedding_model.encode([query])[0]

        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=min(top_k, collection_count)
        )

        retrieved = []

        if not results["ids"] or not results["ids"][0]:
            return retrieved

        for index, chunk_id in enumerate(results["ids"][0]):

            retrieved.append({
                "chunk_id": chunk_id,
                "score": float(results["distances"][0][index]),
                "text": results["documents"][0][index],
                "metadata": results["metadatas"][0][index]
            })

        return retrieved

    @staticmethod
    def reciprocal_rank_fusion(
        result_lists: list,
        k: int = RRF_K,
        top_k: int = RRF_TOP_K
    ) -> list:
        """Combine multiple rankings using Reciprocal Rank Fusion."""

        rrf_scores = defaultdict(float)
        results_by_id = {}

        for results in result_lists:

            for rank, result in enumerate(results, start=1):

                chunk_id = result["chunk_id"]

                rrf_scores[chunk_id] += 1.0 / (k + rank)

                results_by_id[chunk_id] = result

        ranked_ids = sorted(
            rrf_scores,
            key=lambda chunk_id: rrf_scores[chunk_id],
            reverse=True
        )

        fused_results = []

        for chunk_id in ranked_ids[:top_k]:

            result = results_by_id[chunk_id].copy()

            result["rrf_score"] = rrf_scores[chunk_id]

            fused_results.append(result)

        return fused_results

    def rerank_results(
        self,
        query: str,
        results: list,
        top_k: int = FINAL_TOP_K
    ) -> list:
        """Rerank passages using the Cross-Encoder."""

        if not results:
            return []

        pairs = [
            (query, result["text"])
            for result in results
        ]

        scores = self.reranker.predict(
            pairs,
            show_progress_bar=False
        )

        reranked = []

        for result, score in zip(results, scores):

            item = result.copy()
            item["rerank_score"] = float(score)

            reranked.append(item)

        reranked.sort(
            key=lambda item: item["rerank_score"],
            reverse=True
        )

        return reranked[:top_k]

    def hybrid_retrieve(
        self,
        query: str,
        final_top_k: int = FINAL_TOP_K
    ) -> list:
        """Execute the complete hybrid retrieval pipeline."""

        if not query or not query.strip():
            raise ValueError("The search query cannot be empty.")

        # Step 1: BM25 keyword retrieval
        keyword_results = self.bm25_search(
            query,
            top_k=BM25_TOP_K
        )

        # Step 2: Semantic retrieval
        vector_results = self.semantic_search(
            query,
            top_k=SEMANTIC_TOP_K
        )

        # Step 3: Reciprocal Rank Fusion
        fused_results = self.reciprocal_rank_fusion(
            [keyword_results, vector_results],
            top_k=RRF_TOP_K
        )

        # Step 4: Cross-Encoder reranking
        final_results = self.rerank_results(
            query,
            fused_results,
            top_k=final_top_k
        )

        return final_results

    def add_chunks(self, new_chunks: list[dict]) -> int:
        """
        Add newly uploaded document chunks to the BM25 corpus.
    
        Returns:
            Number of newly added chunks.
        """
        if not new_chunks:
            return 0
    
        # Avoid adding chunks that already exist.
        existing_ids = {
            chunk["chunk_id"] for chunk in self.chunks
        }
    
        unique_chunks = [
            chunk for chunk in new_chunks
            if chunk["chunk_id"] not in existing_ids
        ]
    
        if not unique_chunks:
            return 0
    
        # Extend the existing corpus.
        self.chunks.extend(unique_chunks)
    
        # Rebuild BM25 so the new documents become searchable.
        self.tokenized_corpus = [
            self.tokenize(chunk["text"])
            for chunk in self.chunks
        ]
    
        self.bm25 = BM25Okapi(self.tokenized_corpus)
    
        return len(unique_chunks)
