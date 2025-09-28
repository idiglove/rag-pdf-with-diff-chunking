"""
Chunking strategies for RAG system evaluation.
Implements multiple approaches to text chunking for comparison.
"""

import re
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging

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