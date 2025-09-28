"""
Complete RAG pipeline for processing PDFs, chunking, embedding, and retrieval.
Combines all components into a cohesive system.
"""

from typing import List, Dict, Any, Optional
import logging
from pathlib import Path
import time
import json

from pdf_processor import PDFProcessor
from chunking_strategies import ChunkingManager, TextChunk
from embeddings import EmbeddingGenerator
from vector_store import VectorStore

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Complete RAG pipeline for document processing and retrieval."""

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        vector_store_path: str = "data/chroma_db",
        metadata_path: str = "data/document_metadata.json"
    ):
        """
        Initialize RAG pipeline.

        Args:
            embedding_model: Sentence-transformer model name
            vector_store_path: Path for ChromaDB storage
            metadata_path: Path for document metadata persistence
        """
        self.pdf_processor = PDFProcessor()
        self.chunking_manager = ChunkingManager()
        self.embedding_generator = EmbeddingGenerator(model_name=embedding_model)
        self.vector_store = VectorStore(persist_directory=vector_store_path)

        self.metadata_path = Path(metadata_path)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)

        self.processed_documents = {}
        self.document_stats = {}

        # Load existing document metadata
        self._load_document_metadata()
        self._discover_existing_documents()

    def process_document(
        self,
        pdf_path: str,
        document_name: str,
        chunking_strategies: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Process a PDF document with specified chunking strategies.

        Args:
            pdf_path: Path to PDF file
            document_name: Name identifier for the document
            chunking_strategies: List of strategy names to use

        Returns:
            Processing results and statistics
        """
        start_time = time.time()

        # Extract text from PDF
        logger.info(f"Extracting text from {pdf_path}")
        text_data = self.pdf_processor.extract_text(pdf_path)
        full_text = text_data["full_text"]

        # Use default strategies if none specified
        if not chunking_strategies:
            chunking_strategies = self.chunking_manager.list_strategies()

        results = {
            "document_name": document_name,
            "pdf_path": pdf_path,
            "text_length": len(full_text),
            "word_count": len(full_text.split()),
            "strategies": {},
            "processing_time": 0
        }

        # Process with each chunking strategy
        for strategy_name in chunking_strategies:
            logger.info(f"Processing with strategy: {strategy_name}")

            try:
                # Chunk the text
                chunks = self.chunking_manager.chunk_text(full_text, strategy_name)

                if not chunks:
                    logger.warning(f"No chunks generated for strategy {strategy_name}")
                    continue

                # Generate embeddings for chunks
                chunk_texts = [chunk.content for chunk in chunks]
                embeddings = self.embedding_generator.encode_batch(
                    chunk_texts,
                    show_progress=True
                )

                # Create collection and store chunks
                collection_name = f"{document_name}_{strategy_name}"
                collection = self.vector_store.create_collection(document_name, strategy_name)

                # Prepare chunk data for storage
                chunk_data = []
                for i, chunk in enumerate(chunks):
                    chunk_info = {
                        "chunk_id": chunk.metadata.chunk_id,
                        "content": chunk.content,
                        "metadata": {
                            "strategy": strategy_name,
                            "chunk_index": chunk.metadata.chunk_index,
                            "char_count": chunk.metadata.char_count,
                            "word_count": chunk.metadata.word_count,
                            "start_char": chunk.metadata.start_char,
                            "end_char": chunk.metadata.end_char,
                            "document_name": document_name
                        }
                    }
                    chunk_data.append(chunk_info)

                # Store in vector database
                self.vector_store.add_chunks(collection_name, chunk_data, embeddings)

                # Calculate statistics
                chunking_stats = self.chunking_manager.get_chunking_stats(chunks)

                results["strategies"][strategy_name] = {
                    "chunk_count": len(chunks),
                    "collection_name": collection_name,
                    "stats": chunking_stats
                }

                logger.info(f"Stored {len(chunks)} chunks for strategy {strategy_name}")

            except Exception as e:
                logger.error(f"Error processing strategy {strategy_name}: {e}")
                results["strategies"][strategy_name] = {"error": str(e)}

        processing_time = time.time() - start_time
        results["processing_time"] = processing_time

        # Store results
        self.processed_documents[document_name] = results
        self.document_stats[document_name] = text_data["metadata"]

        # Persist metadata
        self._save_document_metadata()

        logger.info(f"Document processing completed in {processing_time:.2f} seconds")
        return results

    def query(
        self,
        query_text: str,
        document_name: str,
        strategy_name: str,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Query the RAG system for relevant chunks.

        Args:
            query_text: Question or search query
            document_name: Name of the document to search
            strategy_name: Chunking strategy to use
            top_k: Number of top results to return

        Returns:
            Query results with chunks and metadata
        """
        start_time = time.time()

        collection_name = f"{document_name}_{strategy_name}"

        try:
            # Generate query embedding
            query_embedding = self.embedding_generator.encode_text(query_text)

            # Search vector store
            search_results = self.vector_store.search(
                collection_name=collection_name,
                query_embedding=query_embedding,
                n_results=top_k
            )

            # Format results
            results = {
                "query": query_text,
                "document_name": document_name,
                "strategy": strategy_name,
                "chunks": [],
                "query_time": time.time() - start_time
            }

            for i in range(len(search_results["documents"])):
                chunk_result = {
                    "content": search_results["documents"][i],
                    "distance": search_results["distances"][i],
                    "similarity": 1 - search_results["distances"][i],  # Convert distance to similarity
                    "metadata": search_results["metadatas"][i],
                    "chunk_id": search_results["ids"][i]
                }
                results["chunks"].append(chunk_result)

            return results

        except Exception as e:
            logger.error(f"Query error: {e}")
            return {
                "query": query_text,
                "error": str(e),
                "query_time": time.time() - start_time
            }

    def compare_strategies(
        self,
        query_text: str,
        document_name: str,
        strategies: Optional[List[str]] = None,
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Compare query results across multiple chunking strategies.

        Args:
            query_text: Question or search query
            document_name: Name of the document to search
            strategies: List of strategies to compare
            top_k: Number of top results per strategy

        Returns:
            Comparison results across strategies
        """
        if not strategies:
            # Get available strategies for this document
            if document_name in self.processed_documents:
                strategies = list(self.processed_documents[document_name]["strategies"].keys())
            else:
                return {"error": f"Document {document_name} not found"}

        comparison_results = {
            "query": query_text,
            "document_name": document_name,
            "strategies": {},
            "summary": {}
        }

        total_start_time = time.time()

        for strategy in strategies:
            result = self.query(query_text, document_name, strategy, top_k)
            comparison_results["strategies"][strategy] = result

        comparison_results["total_time"] = time.time() - total_start_time

        # Generate summary statistics
        strategy_times = []
        for strategy, result in comparison_results["strategies"].items():
            if "query_time" in result:
                strategy_times.append(result["query_time"])

        if strategy_times:
            comparison_results["summary"] = {
                "fastest_strategy": min(comparison_results["strategies"].items(),
                                      key=lambda x: x[1].get("query_time", float('inf')))[0],
                "slowest_strategy": max(comparison_results["strategies"].items(),
                                      key=lambda x: x[1].get("query_time", 0))[0],
                "avg_query_time": sum(strategy_times) / len(strategy_times),
                "total_strategies": len(strategies)
            }

        return comparison_results

    def get_document_info(self, document_name: str) -> Dict[str, Any]:
        """Get information about a processed document."""
        if document_name not in self.processed_documents:
            return {"error": f"Document {document_name} not found"}

        doc_info = self.processed_documents[document_name].copy()
        doc_info["metadata"] = self.document_stats.get(document_name, {})

        return doc_info

    def list_documents(self) -> List[str]:
        """List all processed documents."""
        return list(self.processed_documents.keys())

    def list_available_strategies(self) -> List[str]:
        """List all available chunking strategies."""
        return self.chunking_manager.list_strategies()

    def get_collection_stats(self, document_name: str, strategy_name: str) -> Dict[str, Any]:
        """Get statistics about a specific collection."""
        collection_name = f"{document_name}_{strategy_name}"
        return self.vector_store.get_collection_stats(collection_name)

    def _load_document_metadata(self) -> None:
        """Load persisted document metadata from file."""
        try:
            if self.metadata_path.exists():
                with open(self.metadata_path, 'r') as f:
                    data = json.load(f)
                    self.processed_documents = data.get('processed_documents', {})
                    self.document_stats = data.get('document_stats', {})
                logger.info(f"Loaded metadata for {len(self.processed_documents)} documents")
            else:
                logger.info("No existing document metadata found")
        except Exception as e:
            logger.warning(f"Error loading document metadata: {e}")
            self.processed_documents = {}
            self.document_stats = {}

    def _save_document_metadata(self) -> None:
        """Save document metadata to file."""
        try:
            metadata = {
                'processed_documents': self.processed_documents,
                'document_stats': self.document_stats
            }
            with open(self.metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            logger.debug("Document metadata saved")
        except Exception as e:
            logger.error(f"Error saving document metadata: {e}")

    def _discover_existing_documents(self) -> None:
        """Discover documents from existing collections that may not be in metadata."""
        try:
            collections = self.vector_store.list_collections()
            discovered_docs = set()

            for collection_name in collections:
                # Parse collection name format: document_name_strategy_name
                # Strategy names are known, so we can match them properly
                strategy_names = self.chunking_manager.list_strategies()
                doc_name = None
                strategy_name = None

                # Try to match against known strategies from the end
                for strategy in strategy_names:
                    if collection_name.endswith(f'_{strategy}'):
                        doc_name = collection_name[:-len(f'_{strategy}')]
                        strategy_name = strategy
                        break

                # If no strategy matched, skip this collection
                if not doc_name or not strategy_name:
                    continue

                discovered_docs.add(doc_name)

                # Initialize document info if not exists
                if doc_name not in self.processed_documents:
                    self.processed_documents[doc_name] = {
                        'document_name': doc_name,
                        'pdf_path': 'unknown',
                        'strategies': {},
                        'text_length': 0,
                        'word_count': 0,
                        'processing_time': 0
                    }

                # Add strategy if not already present
                if strategy_name not in self.processed_documents[doc_name]['strategies']:
                    stats = self.vector_store.get_collection_stats(collection_name)
                    if 'error' not in stats:
                        self.processed_documents[doc_name]['strategies'][strategy_name] = {
                            'chunk_count': stats['count'],
                            'collection_name': collection_name,
                            'stats': {'recovered_from_collection': True}
                        }

            if discovered_docs:
                logger.info(f"Discovered {len(discovered_docs)} documents from existing collections: {discovered_docs}")
                self._save_document_metadata()

        except Exception as e:
            logger.warning(f"Error discovering existing documents: {e}")


if __name__ == "__main__":
    # Example usage
    pipeline = RAGPipeline()

    # Check available strategies
    print("Available chunking strategies:")
    for strategy in pipeline.list_available_strategies():
        print(f"- {strategy}")

    # Example usage would be:
    # pipeline.process_document("path/to/book.pdf", "test_book")
    # result = pipeline.query("What is machine learning?", "test_book", "fixed_char_1000")
    # comparison = pipeline.compare_strategies("What is AI?", "test_book")

    print("\nRAG Pipeline initialized successfully!")
    print("Ready to process documents and handle queries.")