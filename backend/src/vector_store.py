"""
Vector storage using ChromaDB for the RAG system.
Handles embedding storage and similarity search.
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class VectorStore:
    """ChromaDB vector storage manager."""

    def __init__(self, persist_directory: str = "data/chroma_db"):
        """Initialize ChromaDB client."""
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                allow_reset=True,
                anonymized_telemetry=False
            )
        )
        self.collections = {}

    def create_collection(self, name: str, strategy: str) -> chromadb.Collection:
        """Create or get a collection for a chunking strategy."""
        collection_name = f"{name}_{strategy}"

        try:
            collection = self.client.create_collection(
                name=collection_name,
                metadata={"strategy": strategy}
            )
        except ValueError:  # Collection already exists
            collection = self.client.get_collection(name=collection_name)

        self.collections[collection_name] = collection
        return collection

    def add_chunks(
        self,
        collection_name: str,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]]
    ) -> None:
        """Add chunks with embeddings to collection."""
        if collection_name not in self.collections:
            raise ValueError(f"Collection {collection_name} not found")

        collection = self.collections[collection_name]

        # Prepare data for ChromaDB
        ids = [chunk["chunk_id"] for chunk in chunks]
        documents = [chunk["content"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]

        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

    def search(
        self,
        collection_name: str,
        query_embedding: List[float],
        n_results: int = 5
    ) -> Dict[str, Any]:
        """Search for similar chunks."""
        if collection_name not in self.collections:
            raise ValueError(f"Collection {collection_name} not found")

        collection = self.collections[collection_name]

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )

        return {
            "documents": results["documents"][0],
            "distances": results["distances"][0],
            "metadatas": results["metadatas"][0],
            "ids": results["ids"][0]
        }

    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """Get statistics about a collection."""
        if collection_name not in self.collections:
            return {}

        collection = self.collections[collection_name]
        count = collection.count()

        return {
            "name": collection_name,
            "count": count,
            "metadata": collection.metadata
        }

    def list_collections(self) -> List[str]:
        """List all available collections."""
        return list(self.collections.keys())

    def reset_collection(self, collection_name: str) -> None:
        """Reset/delete a collection."""
        try:
            self.client.delete_collection(name=collection_name)
            if collection_name in self.collections:
                del self.collections[collection_name]
        except ValueError:
            pass  # Collection doesn't exist


if __name__ == "__main__":
    # Test the vector store
    store = VectorStore()

    # Create test collection
    collection = store.create_collection("test", "fixed_char_1000")

    # Test data
    test_chunks = [
        {
            "chunk_id": "test_001",
            "content": "This is a test chunk about machine learning.",
            "metadata": {"strategy": "test", "char_count": 45}
        }
    ]

    # Mock embeddings (normally from sentence-transformers)
    test_embeddings = [[0.1, 0.2, 0.3, 0.4, 0.5]]

    store.add_chunks("test_fixed_char_1000", test_chunks, test_embeddings)

    # Test search
    results = store.search("test_fixed_char_1000", [0.1, 0.2, 0.3, 0.4, 0.5])
    print("Search results:", results)

    # Get stats
    stats = store.get_collection_stats("test_fixed_char_1000")
    print("Collection stats:", stats)