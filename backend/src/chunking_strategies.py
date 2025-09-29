"""
Chunking strategies for RAG system evaluation.
Implements multiple approaches to text chunking for comparison.
"""

import re
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
import tiktoken
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@dataclass
class ChunkMetadata:
    """Metadata for text chunks."""
    chunk_id: str
    strategy: str
    start_char: int
    end_char: int
    char_count: int
    word_count: int
    chunk_index: int
    overlap_start: Optional[int] = None
    overlap_end: Optional[int] = None


@dataclass
class TextChunk:
    """Represents a chunk of text with metadata."""
    content: str
    metadata: ChunkMetadata


class ChunkingStrategy(ABC):
    """Abstract base class for chunking strategies."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def chunk_text(self, text: str) -> List[TextChunk]:
        """Chunk text according to strategy."""
        pass

    def _create_chunk_metadata(
        self,
        chunk_id: str,
        start_char: int,
        end_char: int,
        chunk_index: int,
        text: str,
        overlap_start: Optional[int] = None,
        overlap_end: Optional[int] = None
    ) -> ChunkMetadata:
        """Create metadata for a chunk."""
        char_count = end_char - start_char
        # Rough word count estimation
        word_count = len(re.findall(r'\b\w+\b', text[start_char:end_char]))

        return ChunkMetadata(
            chunk_id=chunk_id,
            strategy=self.name,
            start_char=start_char,
            end_char=end_char,
            char_count=char_count,
            word_count=word_count,
            chunk_index=chunk_index,
            overlap_start=overlap_start,
            overlap_end=overlap_end
        )


class FixedCharacterChunking(ChunkingStrategy):
    """Fixed character-based chunking strategy."""

    def __init__(self, chunk_size: int = 1000, overlap: int = 0):
        super().__init__(f"fixed_char_{chunk_size}")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str) -> List[TextChunk]:
        """Chunk text into fixed character lengths."""
        chunks = []
        text_length = len(text)

        if text_length == 0:
            return chunks

        chunk_index = 0
        start = 0

        while start < text_length:
            # Calculate end position
            end = min(start + self.chunk_size, text_length)

            # Extract chunk content
            chunk_content = text[start:end].strip()

            if chunk_content:  # Only add non-empty chunks
                chunk_id = f"{self.name}_{chunk_index:04d}"

                metadata = ChunkMetadata(
                    chunk_id=chunk_id,
                    strategy=self.name,
                    start_char=start,
                    end_char=end,
                    char_count=len(chunk_content),
                    word_count=len(chunk_content.split()),
                    chunk_index=chunk_index,
                    overlap_start=max(0, start - self.overlap) if self.overlap > 0 else None,
                    overlap_end=min(text_length, end + self.overlap) if self.overlap > 0 else None
                )

                chunks.append(TextChunk(content=chunk_content, metadata=metadata))
                chunk_index += 1

            # Move to next chunk position
            if self.overlap > 0:
                start = max(start + self.chunk_size - self.overlap, start + 1)
            else:
                start = end

            # Prevent infinite loop
            if start >= text_length:
                break

        return chunks


class SentenceChunking(ChunkingStrategy):
    """Sentence-based chunking strategy."""

    def __init__(self, sentences_per_chunk: int = 5, overlap: int = 0):
        super().__init__(f"sentence_{sentences_per_chunk}")
        self.sentences_per_chunk = sentences_per_chunk
        self.overlap = overlap

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex."""
        # Simple sentence splitting - can be improved with spaCy/NLTK
        sentence_pattern = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_pattern, text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk_text(self, text: str) -> List[TextChunk]:
        """Chunk text by sentence count."""
        sentences = self._split_into_sentences(text)
        chunks = []

        if not sentences:
            return chunks

        chunk_index = 0
        start_sentence = 0

        while start_sentence < len(sentences):
            # Calculate end sentence position
            end_sentence = min(start_sentence + self.sentences_per_chunk, len(sentences))

            # Extract chunk sentences
            chunk_sentences = sentences[start_sentence:end_sentence]
            chunk_content = ' '.join(chunk_sentences).strip()

            if chunk_content:
                # Calculate character positions (approximate)
                start_char = len(' '.join(sentences[:start_sentence]))
                if start_sentence > 0:
                    start_char += 1
                end_char = start_char + len(chunk_content)

                chunk_id = f"{self.name}_{chunk_index:04d}"

                metadata = ChunkMetadata(
                    chunk_id=chunk_id,
                    strategy=self.name,
                    start_char=start_char,
                    end_char=end_char,
                    char_count=len(chunk_content),
                    word_count=len(chunk_content.split()),
                    chunk_index=chunk_index
                )

                chunks.append(TextChunk(content=chunk_content, metadata=metadata))
                chunk_index += 1

            # Move to next chunk position
            if self.overlap > 0:
                start_sentence = max(start_sentence + self.sentences_per_chunk - self.overlap, start_sentence + 1)
            else:
                start_sentence = end_sentence

        return chunks


class ParagraphChunking(ChunkingStrategy):
    """Paragraph-based chunking strategy."""

    def __init__(self, paragraphs_per_chunk: int = 3):
        super().__init__(f"paragraph_{paragraphs_per_chunk}")
        self.paragraphs_per_chunk = paragraphs_per_chunk

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs using multiple heuristics for PDF text."""
        lines = text.split('\n')
        paragraphs = []
        current_paragraph = []

        for i, line in enumerate(lines):
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Skip lines that are likely headers/titles (all caps, short lines)
            if len(line) < 100 and (line.isupper() or line.isdigit()):
                # If we have content, save current paragraph
                if current_paragraph:
                    paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = []
                continue

            # Add line to current paragraph
            current_paragraph.append(line)

            # Check if this line ends a paragraph (ends with sentence punctuation
            # and next line starts with capital or is significantly different)
            if line.endswith(('.', '!', '?', '"')) and i + 1 < len(lines):
                next_line = lines[i + 1].strip()

                # If next line starts with capital and current paragraph is substantial
                if (next_line and next_line[0].isupper() and
                    len(' '.join(current_paragraph)) > 50):
                    paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = []

        # Add any remaining content
        if current_paragraph:
            paragraphs.append(' '.join(current_paragraph))

        # Filter out very short paragraphs (likely noise)
        return [p for p in paragraphs if len(p) > 30]

    def chunk_text(self, text: str) -> List[TextChunk]:
        """Chunk text by paragraph count."""
        paragraphs = self._split_into_paragraphs(text)
        chunks = []

        if not paragraphs:
            return chunks

        chunk_index = 0
        start_para = 0

        while start_para < len(paragraphs):
            # Calculate end paragraph position
            end_para = min(start_para + self.paragraphs_per_chunk, len(paragraphs))

            # Extract chunk paragraphs
            chunk_paragraphs = paragraphs[start_para:end_para]
            chunk_content = '\n\n'.join(chunk_paragraphs).strip()

            if chunk_content:
                # Calculate character positions (approximate)
                start_char = sum(len(p) + 2 for p in paragraphs[:start_para])  # +2 for \n\n
                end_char = start_char + len(chunk_content)

                chunk_id = f"{self.name}_{chunk_index:04d}"

                metadata = ChunkMetadata(
                    chunk_id=chunk_id,
                    strategy=self.name,
                    start_char=start_char,
                    end_char=end_char,
                    char_count=len(chunk_content),
                    word_count=len(chunk_content.split()),
                    chunk_index=chunk_index
                )

                chunks.append(TextChunk(content=chunk_content, metadata=metadata))
                chunk_index += 1

            start_para = end_para

        return chunks


class TokenBasedChunking(ChunkingStrategy):
    """Token-based chunking strategy using tiktoken."""

    def __init__(self, chunk_size: int = 512, overlap: int = 0, encoding: str = "cl100k_base"):
        super().__init__(f"token_{chunk_size}")
        self.chunk_size = chunk_size
        self.overlap = overlap
        try:
            self.tokenizer = tiktoken.get_encoding(encoding)
        except Exception:
            # Fallback to a simple tokenizer if tiktoken fails
            self.tokenizer = None

    def _tokenize(self, text: str) -> List[int]:
        """Tokenize text into tokens."""
        if self.tokenizer:
            return self.tokenizer.encode(text)
        else:
            # Simple fallback: split on whitespace and treat each word as a token
            return text.split()

    def _detokenize(self, tokens) -> str:
        """Convert tokens back to text."""
        if self.tokenizer and isinstance(tokens[0], int):
            return self.tokenizer.decode(tokens)
        else:
            # Simple fallback: join words
            return ' '.join(tokens)

    def chunk_text(self, text: str) -> List[TextChunk]:
        """Chunk text into fixed token lengths."""
        chunks = []
        tokens = self._tokenize(text)

        if not tokens:
            return chunks

        chunk_index = 0
        start_token = 0

        while start_token < len(tokens):
            # Calculate end token position
            end_token = min(start_token + self.chunk_size, len(tokens))

            # Extract chunk tokens
            chunk_tokens = tokens[start_token:end_token]
            chunk_content = self._detokenize(chunk_tokens).strip()

            if chunk_content:
                # Approximate character positions
                start_char = len(self._detokenize(tokens[:start_token])) if start_token > 0 else 0
                end_char = start_char + len(chunk_content)

                chunk_id = f"{self.name}_{chunk_index:04d}"

                metadata = ChunkMetadata(
                    chunk_id=chunk_id,
                    strategy=self.name,
                    start_char=start_char,
                    end_char=end_char,
                    char_count=len(chunk_content),
                    word_count=len(chunk_content.split()),
                    chunk_index=chunk_index
                )

                chunks.append(TextChunk(content=chunk_content, metadata=metadata))
                chunk_index += 1

            # Move to next chunk position
            if self.overlap > 0:
                start_token = max(start_token + self.chunk_size - self.overlap, start_token + 1)
            else:
                start_token = end_token

        return chunks


class SlidingWindowChunking(ChunkingStrategy):
    """Sliding window chunking with configurable overlap."""

    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        super().__init__(f"sliding_window_{chunk_size}_{overlap}")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str) -> List[TextChunk]:
        """Chunk text using sliding window with overlap."""
        chunks = []
        text_length = len(text)

        if text_length == 0:
            return chunks

        chunk_index = 0
        start = 0

        while start < text_length:
            # Calculate end position
            end = min(start + self.chunk_size, text_length)

            # Extract chunk content
            chunk_content = text[start:end].strip()

            if chunk_content:
                chunk_id = f"{self.name}_{chunk_index:04d}"

                metadata = ChunkMetadata(
                    chunk_id=chunk_id,
                    strategy=self.name,
                    start_char=start,
                    end_char=end,
                    char_count=len(chunk_content),
                    word_count=len(chunk_content.split()),
                    chunk_index=chunk_index,
                    overlap_start=max(0, start - self.overlap) if chunk_index > 0 else None,
                    overlap_end=min(text_length, end + self.overlap) if end < text_length else None
                )

                chunks.append(TextChunk(content=chunk_content, metadata=metadata))
                chunk_index += 1

            # Move to next chunk position with overlap
            step_size = self.chunk_size - self.overlap
            start = start + max(step_size, 1)  # Ensure we always move forward

            # Stop if we can't make a meaningful chunk
            if start >= text_length:
                break

        return chunks


class SectionBasedChunking(ChunkingStrategy):
    """Section-based chunking using headers and structure."""

    def __init__(self, max_section_size: int = 3000):
        super().__init__(f"section_based_{max_section_size}")
        self.max_section_size = max_section_size

    def _detect_headers(self, lines: List[str]) -> List[int]:
        """Detect header lines that indicate section boundaries."""
        header_indices = []

        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            # Check for common header patterns
            if (
                # All caps lines (but not too long)
                (line.isupper() and len(line) < 100) or
                # Lines starting with numbers (1., 2., Chapter, etc.)
                re.match(r'^(\d+\.|\d+\s|Chapter|CHAPTER)', line) or
                # Lines with specific header keywords
                re.match(r'^(Introduction|Conclusion|Summary|Overview|Background)', line, re.IGNORECASE) or
                # Short lines that might be titles
                (len(line) < 80 and not line.endswith('.') and len(line.split()) <= 8)
            ):
                header_indices.append(i)

        return header_indices

    def chunk_text(self, text: str) -> List[TextChunk]:
        """Chunk text by section boundaries."""
        lines = text.split('\n')
        header_indices = self._detect_headers(lines)

        # Add start and end boundaries
        boundaries = [0] + header_indices + [len(lines)]
        boundaries = sorted(set(boundaries))  # Remove duplicates and sort

        chunks = []
        chunk_index = 0

        for i in range(len(boundaries) - 1):
            start_line = boundaries[i]
            end_line = boundaries[i + 1]

            # Extract section lines
            section_lines = lines[start_line:end_line]
            section_text = '\n'.join(section_lines).strip()

            if not section_text or len(section_text) < 50:  # Skip very short sections
                continue

            # If section is too large, split it
            if len(section_text) > self.max_section_size:
                sub_chunks = self._split_large_section(section_text, chunk_index)
                chunks.extend(sub_chunks)
                chunk_index += len(sub_chunks)
            else:
                # Calculate character positions (approximate)
                start_char = sum(len(line) + 1 for line in lines[:start_line])
                end_char = start_char + len(section_text)

                chunk_id = f"{self.name}_{chunk_index:04d}"

                metadata = ChunkMetadata(
                    chunk_id=chunk_id,
                    strategy=self.name,
                    start_char=start_char,
                    end_char=end_char,
                    char_count=len(section_text),
                    word_count=len(section_text.split()),
                    chunk_index=chunk_index
                )

                chunks.append(TextChunk(content=section_text, metadata=metadata))
                chunk_index += 1

        return chunks

    def _split_large_section(self, text: str, base_index: int) -> List[TextChunk]:
        """Split a large section into smaller chunks."""
        chunks = []
        sentences = re.split(r'(?<=[.!?])\s+', text)

        current_chunk = []
        current_size = 0
        sub_index = 0

        for sentence in sentences:
            sentence_size = len(sentence)

            if current_size + sentence_size > self.max_section_size and current_chunk:
                # Create chunk from current sentences
                chunk_text = ' '.join(current_chunk).strip()
                if chunk_text:
                    chunk_id = f"{self.name}_{base_index:04d}_{sub_index:02d}"

                    metadata = ChunkMetadata(
                        chunk_id=chunk_id,
                        strategy=self.name,
                        start_char=0,  # Approximate
                        end_char=len(chunk_text),
                        char_count=len(chunk_text),
                        word_count=len(chunk_text.split()),
                        chunk_index=base_index + sub_index
                    )

                    chunks.append(TextChunk(content=chunk_text, metadata=metadata))
                    sub_index += 1

                current_chunk = [sentence]
                current_size = sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size

        # Add remaining content
        if current_chunk:
            chunk_text = ' '.join(current_chunk).strip()
            if chunk_text:
                chunk_id = f"{self.name}_{base_index:04d}_{sub_index:02d}"

                metadata = ChunkMetadata(
                    chunk_id=chunk_id,
                    strategy=self.name,
                    start_char=0,
                    end_char=len(chunk_text),
                    char_count=len(chunk_text),
                    word_count=len(chunk_text.split()),
                    chunk_index=base_index + sub_index
                )

                chunks.append(TextChunk(content=chunk_text, metadata=metadata))

        return chunks


class RecursiveChunking(ChunkingStrategy):
    """Recursive chunking that splits by structure, then by size."""

    def __init__(self, max_chunk_size: int = 2000, min_chunk_size: int = 100):
        super().__init__(f"recursive_{max_chunk_size}")
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size

    def chunk_text(self, text: str) -> List[TextChunk]:
        """Recursively chunk text by structure and size."""
        chunks = []
        self._recursive_split(text, chunks, 0)
        return chunks

    def _recursive_split(self, text: str, chunks: List[TextChunk], depth: int) -> None:
        """Recursively split text into chunks."""
        text = text.strip()

        if not text or len(text) < self.min_chunk_size:
            return

        # If text is small enough, create a chunk
        if len(text) <= self.max_chunk_size:
            chunk_id = f"{self.name}_{len(chunks):04d}"

            metadata = ChunkMetadata(
                chunk_id=chunk_id,
                strategy=self.name,
                start_char=0,  # Would need full text context for accurate positions
                end_char=len(text),
                char_count=len(text),
                word_count=len(text.split()),
                chunk_index=len(chunks)
            )

            chunks.append(TextChunk(content=text, metadata=metadata))
            return

        # Try splitting by different separators in order of preference
        separators = [
            '\n\n\n',  # Multiple line breaks
            '\n\n',    # Double line breaks (paragraphs)
            '\n',      # Single line breaks
            '. ',      # Sentence endings
            ' '        # Words
        ]

        for separator in separators:
            if separator in text:
                parts = text.split(separator)

                # Try to combine parts to create balanced chunks
                current_chunk = ""

                for part in parts:
                    test_chunk = current_chunk + separator + part if current_chunk else part

                    if len(test_chunk) <= self.max_chunk_size:
                        current_chunk = test_chunk
                    else:
                        # Current chunk is ready, process it
                        if current_chunk:
                            self._recursive_split(current_chunk, chunks, depth + 1)

                        # Start new chunk with current part
                        current_chunk = part

                # Process remaining chunk
                if current_chunk:
                    self._recursive_split(current_chunk, chunks, depth + 1)

                return

        # If no separator found, force split at max size
        if len(text) > self.max_chunk_size:
            mid_point = self.max_chunk_size
            self._recursive_split(text[:mid_point], chunks, depth + 1)
            self._recursive_split(text[mid_point:], chunks, depth + 1)


class SemanticSimilarityChunking(ChunkingStrategy):
    """Semantic similarity-based chunking using sentence embeddings."""

    def __init__(self, max_chunk_size: int = 2000, similarity_threshold: float = 0.7):
        super().__init__(f"semantic_similarity_{similarity_threshold}")
        self.max_chunk_size = max_chunk_size
        self.similarity_threshold = similarity_threshold
        self.model = None

    def _get_model(self):
        """Lazy load the sentence transformer model."""
        if self.model is None:
            try:
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                logger.warning(f"Could not load sentence transformer: {e}")
                self.model = False
        return self.model

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentence_pattern = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_pattern, text)
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]

    def _calculate_similarity(self, sent1: str, sent2: str) -> float:
        """Calculate semantic similarity between two sentences."""
        model = self._get_model()
        if not model:
            # Fallback to simple word overlap
            words1 = set(sent1.lower().split())
            words2 = set(sent2.lower().split())
            if not words1 or not words2:
                return 0.0
            return len(words1.intersection(words2)) / len(words1.union(words2))

        try:
            embeddings = model.encode([sent1, sent2])
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            return float(similarity)
        except Exception:
            # Fallback to word overlap
            words1 = set(sent1.lower().split())
            words2 = set(sent2.lower().split())
            if not words1 or not words2:
                return 0.0
            return len(words1.intersection(words2)) / len(words1.union(words2))

    def chunk_text(self, text: str) -> List[TextChunk]:
        """Chunk text based on semantic similarity between sentences."""
        sentences = self._split_into_sentences(text)

        if not sentences:
            return []

        chunks = []
        current_chunk_sentences = [sentences[0]]
        chunk_index = 0

        for i in range(1, len(sentences)):
            current_sentence = sentences[i]

            # Calculate similarity with the last sentence in current chunk
            last_sentence = current_chunk_sentences[-1]
            similarity = self._calculate_similarity(last_sentence, current_sentence)

            # Check if we should start a new chunk
            current_chunk_text = ' '.join(current_chunk_sentences + [current_sentence])

            if (similarity < self.similarity_threshold or
                len(current_chunk_text) > self.max_chunk_size):

                # Finalize current chunk
                if current_chunk_sentences:
                    chunk_content = ' '.join(current_chunk_sentences).strip()

                    if chunk_content:
                        chunk_id = f"{self.name}_{chunk_index:04d}"

                        metadata = ChunkMetadata(
                            chunk_id=chunk_id,
                            strategy=self.name,
                            start_char=0,  # Approximate
                            end_char=len(chunk_content),
                            char_count=len(chunk_content),
                            word_count=len(chunk_content.split()),
                            chunk_index=chunk_index
                        )

                        chunks.append(TextChunk(content=chunk_content, metadata=metadata))
                        chunk_index += 1

                # Start new chunk
                current_chunk_sentences = [current_sentence]
            else:
                # Add to current chunk
                current_chunk_sentences.append(current_sentence)

        # Add final chunk
        if current_chunk_sentences:
            chunk_content = ' '.join(current_chunk_sentences).strip()

            if chunk_content:
                chunk_id = f"{self.name}_{chunk_index:04d}"

                metadata = ChunkMetadata(
                    chunk_id=chunk_id,
                    strategy=self.name,
                    start_char=0,
                    end_char=len(chunk_content),
                    char_count=len(chunk_content),
                    word_count=len(chunk_content.split()),
                    chunk_index=chunk_index
                )

                chunks.append(TextChunk(content=chunk_content, metadata=metadata))

        return chunks


class ChunkingManager:
    """Manager for different chunking strategies."""

    def __init__(self):
        self.strategies = {}
        self._register_default_strategies()

    def _register_default_strategies(self):
        """Register default chunking strategies."""
        # Character-based strategy
        self.register_strategy(FixedCharacterChunking(chunk_size=1000))

        # Sentence-based strategy
        self.register_strategy(SentenceChunking(sentences_per_chunk=8))

        # Paragraph-based strategy
        self.register_strategy(ParagraphChunking(paragraphs_per_chunk=3))

        # Token-based strategies
        self.register_strategy(TokenBasedChunking(chunk_size=512))

        # Sliding window strategies with overlap
        self.register_strategy(SlidingWindowChunking(chunk_size=1000, overlap=200))

        # Section-based chunking
        self.register_strategy(SectionBasedChunking(max_section_size=3000))

        # Recursive chunking
        self.register_strategy(RecursiveChunking(max_chunk_size=2000))

        # Semantic similarity chunking
        self.register_strategy(SemanticSimilarityChunking(similarity_threshold=0.7))

    def register_strategy(self, strategy: ChunkingStrategy):
        """Register a chunking strategy."""
        self.strategies[strategy.name] = strategy

    def get_strategy(self, name: str) -> Optional[ChunkingStrategy]:
        """Get a chunking strategy by name."""
        return self.strategies.get(name)

    def list_strategies(self) -> List[str]:
        """List all available strategy names."""
        return list(self.strategies.keys())

    def chunk_text(self, text: str, strategy_name: str) -> List[TextChunk]:
        """Chunk text using specified strategy."""
        strategy = self.get_strategy(strategy_name)
        if not strategy:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        return strategy.chunk_text(text)

    def get_chunking_stats(self, chunks: List[TextChunk]) -> Dict[str, Any]:
        """Get statistics about chunks."""
        if not chunks:
            return {}

        char_counts = [chunk.metadata.char_count for chunk in chunks]
        word_counts = [chunk.metadata.word_count for chunk in chunks]

        return {
            "total_chunks": len(chunks),
            "strategy": chunks[0].metadata.strategy,
            "total_characters": sum(char_counts),
            "total_words": sum(word_counts),
            "avg_chars_per_chunk": sum(char_counts) / len(chunks),
            "avg_words_per_chunk": sum(word_counts) / len(chunks),
            "min_chars": min(char_counts),
            "max_chars": max(char_counts),
            "min_words": min(word_counts),
            "max_words": max(word_counts)
        }


if __name__ == "__main__":
    # Example usage
    sample_text = """
    This is a sample text for testing chunking strategies.
    It contains multiple sentences and paragraphs.

    Each paragraph should be handled differently by different strategies.
    Some strategies focus on character count, others on word count.

    The goal is to evaluate which chunking strategy works best for RAG systems.
    """

    manager = ChunkingManager()

    print("Available strategies:")
    for strategy_name in manager.list_strategies():
        print(f"- {strategy_name}")

    print("\n--- Testing fixed_char_1000 ---")
    chunks = manager.chunk_text(sample_text, "fixed_char_1000")
    stats = manager.get_chunking_stats(chunks)

    print(f"Total chunks: {stats['total_chunks']}")
    print(f"Avg chars per chunk: {stats['avg_chars_per_chunk']:.1f}")
    print(f"Avg words per chunk: {stats['avg_words_per_chunk']:.1f}")

    for i, chunk in enumerate(chunks[:2]):  # Show first 2 chunks
        print(f"\nChunk {i + 1}:")
        print(f"Content: {chunk.content[:100]}...")
        print(f"Metadata: {chunk.metadata}")