"""
Text embedding utilities using sentence-transformers.
Handles text vectorization for the RAG system.
"""

from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import logging
import numpy as np
from pathlib import Path
import torch

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Text embedding generator using sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_dir: Optional[str] = None):
        """
        Initialize embedding generator.

        Args:
            model_name: Name of the sentence-transformer model
            cache_dir: Directory to cache downloaded models
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir) if cache_dir else None

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Loading embedding model: {model_name}")

        # Load model with cache directory if specified
        if self.cache_dir:
            self.model = SentenceTransformer(model_name, cache_folder=str(self.cache_dir))
        else:
            self.model = SentenceTransformer(model_name)

        # Get model properties
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.max_seq_length = self.model.max_seq_length

        logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}, Max sequence length: {self.max_seq_length}")

    def encode_text(self, text: str) -> List[float]:
        """
        Encode single text into embedding.

        Args:
            text: Text to encode

        Returns:
            List of float values representing the embedding
        """
        if not text or not text.strip():
            return [0.0] * self.embedding_dim

        embedding = self.model.encode(text.strip(), convert_to_tensor=False)
        return embedding.tolist()

    def encode_batch(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> List[List[float]]:
        """
        Encode multiple texts into embeddings efficiently.

        Args:
            texts: List of texts to encode
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar

        Returns:
            List of embeddings as lists of floats
        """
        if not texts:
            return []

        # Filter out empty texts and keep track of indices
        valid_texts = []
        valid_indices = []

        for i, text in enumerate(texts):
            if text and text.strip():
                valid_texts.append(text.strip())
                valid_indices.append(i)

        if not valid_texts:
            return [[0.0] * self.embedding_dim] * len(texts)

        # Encode valid texts
        embeddings = self.model.encode(
            valid_texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_tensor=False
        )

        # Create result list with proper ordering
        result = []
        valid_idx = 0

        for i in range(len(texts)):
            if i in valid_indices:
                result.append(embeddings[valid_idx].tolist())
                valid_idx += 1
            else:
                result.append([0.0] * self.embedding_dim)

        return result

    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            Cosine similarity score (-1 to 1)
        """
        # Convert to numpy arrays
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        # Compute cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def find_most_similar(
        self,
        query_embedding: List[float],
        candidate_embeddings: List[List[float]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find most similar embeddings to query.

        Args:
            query_embedding: Query embedding to compare against
            candidate_embeddings: List of candidate embeddings
            top_k: Number of top results to return

        Returns:
            List of dictionaries with index and similarity score
        """
        if not candidate_embeddings:
            return []

        similarities = []

        for i, candidate in enumerate(candidate_embeddings):
            similarity = self.compute_similarity(query_embedding, candidate)
            similarities.append({
                "index": i,
                "similarity": similarity
            })

        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x["similarity"], reverse=True)

        return similarities[:top_k]

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dim,
            "max_sequence_length": self.max_seq_length,
            "device": str(self.model.device),
            "model_type": type(self.model).__name__
        }


def create_embedding_generator(model_name: str = "all-MiniLM-L6-v2") -> EmbeddingGenerator:
    """
    Factory function to create embedding generator.

    Args:
        model_name: Name of the sentence-transformer model

    Returns:
        Configured EmbeddingGenerator instance
    """
    return EmbeddingGenerator(model_name=model_name)


if __name__ == "__main__":
    # Example usage and testing

    # Create embedding generator
    embedder = EmbeddingGenerator()

    # Test single text encoding
    sample_text = "This is a sample text for testing embeddings."
    embedding = embedder.encode_text(sample_text)
    print(f"Single text embedding shape: {len(embedding)}")
    print(f"Embedding preview: {embedding[:5]}...")

    # Test batch encoding
    sample_texts = [
        "This is the first document about machine learning.",
        "The second document discusses natural language processing.",
        "A third document covers computer vision topics.",
        "The fourth document is about data science and analytics."
    ]

    embeddings = embedder.encode_batch(sample_texts)
    print(f"\nBatch encoding results:")
    print(f"Number of embeddings: {len(embeddings)}")
    print(f"Embedding dimension: {len(embeddings[0])}")

    # Test similarity computation
    query_text = "machine learning algorithms and models"
    query_embedding = embedder.encode_text(query_text)

    similarities = embedder.find_most_similar(query_embedding, embeddings, top_k=3)

    print(f"\nSimilarity results for query: '{query_text}'")
    for result in similarities:
        idx = result["index"]
        similarity = result["similarity"]
        print(f"Document {idx + 1}: {similarity:.3f} - {sample_texts[idx]}")

    # Model info
    print(f"\nModel information:")
    model_info = embedder.get_model_info()
    for key, value in model_info.items():
        print(f"{key}: {value}")