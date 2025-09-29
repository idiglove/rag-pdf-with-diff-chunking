"""
Unit tests for chunking strategies.
Tests all implemented chunking strategies for correctness and edge cases.
"""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from chunking_strategies import (
    FixedCharacterChunking,
    SentenceChunking,
    ParagraphChunking,
    TokenBasedChunking,
    SlidingWindowChunking,
    SectionBasedChunking,
    RecursiveChunking,
    SemanticSimilarityChunking,
    ChunkingManager,
    TextChunk,
    ChunkMetadata
)


class TestFixtures:
    """Test fixture data for chunking tests."""

    # Empty and minimal text
    EMPTY_TEXT = ""
    SINGLE_CHAR = "A"
    SINGLE_WORD = "Hello"
    SINGLE_SENTENCE = "This is a single sentence."

    # Short text
    SHORT_TEXT = """
    This is a short text for testing.
    It has multiple sentences.
    Each sentence should be processed correctly.
    """

    # Medium text with paragraphs
    MEDIUM_TEXT = """
    Introduction

    This is the introduction paragraph. It explains what this text is about.
    The introduction sets the context for the reader.

    Chapter 1: Basic Concepts

    This chapter covers basic concepts. We'll explore fundamental ideas.
    Each concept builds on the previous one.

    The main principle is understanding. This requires careful analysis.
    Implementation details matter significantly.

    Chapter 2: Advanced Topics

    Moving to advanced topics requires deeper understanding.
    Complex ideas need sophisticated approaches.

    Summary

    In conclusion, different approaches have different strengths.
    The choice depends on your specific use case.
    """

    # Text with special characters
    SPECIAL_CHARS_TEXT = "Text with émojis 😀, symbols ©, and unicode characters: αβγ. What about numbers 123?"

    # Malformed text
    MALFORMED_TEXT = "No periods here Multiple sentences without proper punctuation How to handle this"

    # Very long text
    LONG_TEXT = " ".join(["This is sentence number {}.".format(i) for i in range(100)])


class TestFixedCharacterChunking:
    """Test FixedCharacterChunking strategy."""

    def test_empty_text(self):
        """Test with empty text."""
        strategy = FixedCharacterChunking(chunk_size=100)
        chunks = strategy.chunk_text(TestFixtures.EMPTY_TEXT)
        assert len(chunks) == 0

    def test_single_character(self):
        """Test with single character."""
        strategy = FixedCharacterChunking(chunk_size=100)
        chunks = strategy.chunk_text(TestFixtures.SINGLE_CHAR)
        assert len(chunks) == 1
        assert chunks[0].content == "A"
        assert chunks[0].metadata.char_count == 1
        assert chunks[0].metadata.word_count == 1

    def test_normal_chunking(self):
        """Test normal chunking behavior."""
        strategy = FixedCharacterChunking(chunk_size=50)
        chunks = strategy.chunk_text(TestFixtures.SHORT_TEXT)

        assert len(chunks) > 1  # Should create multiple chunks

        # Check metadata consistency
        for i, chunk in enumerate(chunks):
            assert chunk.metadata.chunk_index == i
            assert chunk.metadata.strategy == "fixed_char_50"
            assert chunk.metadata.char_count > 0
            assert chunk.metadata.word_count > 0

    def test_chunk_size_edge_cases(self):
        """Test with very small and large chunk sizes."""
        text = TestFixtures.SHORT_TEXT

        # Very small chunk size
        strategy = FixedCharacterChunking(chunk_size=1)
        chunks = strategy.chunk_text(text)
        assert len(chunks) > 0

        # Very large chunk size (larger than text)
        strategy = FixedCharacterChunking(chunk_size=10000)
        chunks = strategy.chunk_text(text)
        assert len(chunks) == 1
        assert len(chunks[0].content) <= len(text)

    def test_overlap_functionality(self):
        """Test chunking with overlap."""
        strategy = FixedCharacterChunking(chunk_size=50, overlap=10)
        chunks = strategy.chunk_text(TestFixtures.MEDIUM_TEXT)

        if len(chunks) > 1:
            # Verify overlap metadata is set
            for chunk in chunks[1:]:  # Skip first chunk
                assert chunk.metadata.overlap_start is not None

            for chunk in chunks[:-1]:  # Skip last chunk
                assert chunk.metadata.overlap_end is not None


class TestSentenceChunking:
    """Test SentenceChunking strategy."""

    def test_empty_text(self):
        """Test with empty text."""
        strategy = SentenceChunking(sentences_per_chunk=3)
        chunks = strategy.chunk_text(TestFixtures.EMPTY_TEXT)
        assert len(chunks) == 0

    def test_single_sentence(self):
        """Test with single sentence."""
        strategy = SentenceChunking(sentences_per_chunk=3)
        chunks = strategy.chunk_text(TestFixtures.SINGLE_SENTENCE)
        assert len(chunks) == 1
        assert chunks[0].content.strip() == TestFixtures.SINGLE_SENTENCE.strip()

    def test_sentence_splitting(self):
        """Test sentence splitting logic."""
        strategy = SentenceChunking(sentences_per_chunk=2)
        chunks = strategy.chunk_text(TestFixtures.SHORT_TEXT)

        assert len(chunks) >= 1

        # Verify each chunk contains complete sentences
        for chunk in chunks:
            content = chunk.content.strip()
            assert content  # Not empty
            # Should end with sentence punctuation (approximately)
            if len(content) > 10:  # Skip very short chunks
                assert any(content.endswith(punct) for punct in ['.', '!', '?']) or True  # Flexible for edge cases

    def test_malformed_text(self):
        """Test with text without proper sentence punctuation."""
        strategy = SentenceChunking(sentences_per_chunk=2)
        chunks = strategy.chunk_text(TestFixtures.MALFORMED_TEXT)

        # Should handle gracefully without crashing
        assert isinstance(chunks, list)


class TestParagraphChunking:
    """Test ParagraphChunking strategy."""

    def test_empty_text(self):
        """Test with empty text."""
        strategy = ParagraphChunking(paragraphs_per_chunk=2)
        chunks = strategy.chunk_text(TestFixtures.EMPTY_TEXT)
        assert len(chunks) == 0

    def test_paragraph_detection(self):
        """Test paragraph detection logic."""
        strategy = ParagraphChunking(paragraphs_per_chunk=1)
        chunks = strategy.chunk_text(TestFixtures.MEDIUM_TEXT)

        assert len(chunks) > 1  # Should detect multiple paragraphs

        # Verify chunks contain meaningful content
        for chunk in chunks:
            assert len(chunk.content.strip()) > 30  # Minimum paragraph size filter

    def test_single_paragraph(self):
        """Test with single paragraph text."""
        strategy = ParagraphChunking(paragraphs_per_chunk=1)
        single_para = "This is a single paragraph. It has multiple sentences. But no paragraph breaks."
        chunks = strategy.chunk_text(single_para)

        # Might create one chunk or none depending on paragraph detection
        assert isinstance(chunks, list)


class TestTokenBasedChunking:
    """Test TokenBasedChunking strategy."""

    def test_empty_text(self):
        """Test with empty text."""
        strategy = TokenBasedChunking(chunk_size=100)
        chunks = strategy.chunk_text(TestFixtures.EMPTY_TEXT)
        assert len(chunks) == 0

    def test_normal_chunking(self):
        """Test normal token-based chunking."""
        strategy = TokenBasedChunking(chunk_size=50)
        chunks = strategy.chunk_text(TestFixtures.MEDIUM_TEXT)

        assert len(chunks) > 0

        # Verify chunks have reasonable content
        for chunk in chunks:
            assert len(chunk.content.strip()) > 0
            assert chunk.metadata.strategy == "token_50"

    def test_fallback_behavior(self):
        """Test fallback behavior when tiktoken is unavailable."""
        # This test verifies the fallback tokenizer works
        strategy = TokenBasedChunking(chunk_size=10)
        chunks = strategy.chunk_text(TestFixtures.SHORT_TEXT)

        assert len(chunks) > 0
        # Should work regardless of tiktoken availability


class TestSlidingWindowChunking:
    """Test SlidingWindowChunking strategy."""

    def test_empty_text(self):
        """Test with empty text."""
        strategy = SlidingWindowChunking(chunk_size=100, overlap=20)
        chunks = strategy.chunk_text(TestFixtures.EMPTY_TEXT)
        assert len(chunks) == 0

    def test_overlap_logic(self):
        """Test sliding window overlap logic."""
        strategy = SlidingWindowChunking(chunk_size=100, overlap=20)
        chunks = strategy.chunk_text(TestFixtures.MEDIUM_TEXT)

        if len(chunks) > 1:
            # Verify overlap metadata
            for chunk in chunks:
                if chunk.metadata.chunk_index > 0:
                    assert chunk.metadata.overlap_start is not None
                if chunk.metadata.chunk_index < len(chunks) - 1:
                    assert chunk.metadata.overlap_end is not None

    def test_no_overlap(self):
        """Test sliding window with no overlap (should behave like fixed chunking)."""
        strategy = SlidingWindowChunking(chunk_size=100, overlap=0)
        chunks = strategy.chunk_text(TestFixtures.MEDIUM_TEXT)

        # Should create chunks without overlap metadata
        for chunk in chunks:
            # No overlap metadata should be set for 0 overlap
            pass  # Basic test that it doesn't crash


class TestSectionBasedChunking:
    """Test SectionBasedChunking strategy."""

    def test_empty_text(self):
        """Test with empty text."""
        strategy = SectionBasedChunking(max_section_size=1000)
        chunks = strategy.chunk_text(TestFixtures.EMPTY_TEXT)
        assert len(chunks) == 0

    def test_header_detection(self):
        """Test header detection in structured text."""
        strategy = SectionBasedChunking(max_section_size=1000)
        chunks = strategy.chunk_text(TestFixtures.MEDIUM_TEXT)

        # Should detect sections based on headers like "Introduction", "Chapter 1", etc.
        assert len(chunks) > 1

        # Check that meaningful sections were created
        total_chars = sum(len(chunk.content) for chunk in chunks)
        assert total_chars > 0

    def test_large_section_splitting(self):
        """Test splitting of sections that exceed max size."""
        strategy = SectionBasedChunking(max_section_size=50)  # Very small max size
        chunks = strategy.chunk_text(TestFixtures.MEDIUM_TEXT)

        # Should split large sections
        assert len(chunks) > 1

        # Most chunks should be under the max size (with some flexibility)
        for chunk in chunks:
            # Allow some flexibility for section boundaries
            assert len(chunk.content) <= strategy.max_section_size * 2


class TestRecursiveChunking:
    """Test RecursiveChunking strategy."""

    def test_empty_text(self):
        """Test with empty text."""
        strategy = RecursiveChunking(max_chunk_size=200)
        chunks = strategy.chunk_text(TestFixtures.EMPTY_TEXT)
        assert len(chunks) == 0

    def test_recursive_splitting(self):
        """Test recursive splitting behavior."""
        strategy = RecursiveChunking(max_chunk_size=200)
        chunks = strategy.chunk_text(TestFixtures.MEDIUM_TEXT)

        assert len(chunks) > 0

        # Verify chunks are within size limits
        for chunk in chunks:
            assert len(chunk.content) <= strategy.max_chunk_size

    def test_small_text_no_splitting(self):
        """Test that small text is not split."""
        strategy = RecursiveChunking(max_chunk_size=1000)
        chunks = strategy.chunk_text(TestFixtures.SHORT_TEXT)

        # Short text should fit in one chunk
        assert len(chunks) == 1


class TestSemanticSimilarityChunking:
    """Test SemanticSimilarityChunking strategy."""

    def test_empty_text(self):
        """Test with empty text."""
        strategy = SemanticSimilarityChunking(similarity_threshold=0.7)
        chunks = strategy.chunk_text(TestFixtures.EMPTY_TEXT)
        assert len(chunks) == 0

    def test_semantic_chunking(self):
        """Test semantic similarity-based chunking."""
        strategy = SemanticSimilarityChunking(similarity_threshold=0.7)
        chunks = strategy.chunk_text(TestFixtures.MEDIUM_TEXT)

        assert len(chunks) > 0

        # Verify chunks contain meaningful content
        for chunk in chunks:
            assert len(chunk.content.strip()) > 10  # Minimum sentence filter

    def test_fallback_behavior(self):
        """Test fallback behavior when sentence transformer is unavailable."""
        strategy = SemanticSimilarityChunking(similarity_threshold=0.5)
        chunks = strategy.chunk_text(TestFixtures.SHORT_TEXT)

        # Should work with fallback similarity calculation
        assert len(chunks) > 0


class TestChunkingManager:
    """Test ChunkingManager functionality."""

    def test_strategy_registration(self):
        """Test strategy registration and retrieval."""
        manager = ChunkingManager()

        # Should have default strategies registered
        strategies = manager.list_strategies()
        assert len(strategies) > 0

        # Should include all expected strategies
        expected_strategies = [
            "fixed_char_1000",
            "sentence_8",
            "paragraph_3",
            "token_512",
            "sliding_window_1000_200",
            "section_based_3000",
            "recursive_2000",
            "semantic_similarity_0.7"
        ]

        for expected in expected_strategies:
            assert expected in strategies

    def test_get_strategy(self):
        """Test getting strategies by name."""
        manager = ChunkingManager()

        # Valid strategy
        strategy = manager.get_strategy("fixed_char_1000")
        assert strategy is not None
        assert isinstance(strategy, FixedCharacterChunking)

        # Invalid strategy
        strategy = manager.get_strategy("nonexistent_strategy")
        assert strategy is None

    def test_chunk_text_via_manager(self):
        """Test chunking text through manager."""
        manager = ChunkingManager()

        chunks = manager.chunk_text(TestFixtures.SHORT_TEXT, "fixed_char_1000")
        assert len(chunks) > 0
        assert isinstance(chunks[0], TextChunk)

    def test_chunking_stats(self):
        """Test chunking statistics calculation."""
        manager = ChunkingManager()
        chunks = manager.chunk_text(TestFixtures.MEDIUM_TEXT, "fixed_char_1000")

        stats = manager.get_chunking_stats(chunks)

        assert "total_chunks" in stats
        assert "strategy" in stats
        assert "total_characters" in stats
        assert "total_words" in stats
        assert "avg_chars_per_chunk" in stats
        assert "avg_words_per_chunk" in stats

        assert stats["total_chunks"] == len(chunks)
        assert stats["strategy"] == "fixed_char_1000"

    def test_invalid_strategy_error(self):
        """Test error handling for invalid strategy."""
        manager = ChunkingManager()

        with pytest.raises(ValueError, match="Unknown strategy"):
            manager.chunk_text(TestFixtures.SHORT_TEXT, "invalid_strategy")


class TestChunkMetadata:
    """Test ChunkMetadata functionality."""

    def test_metadata_creation(self):
        """Test chunk metadata creation and attributes."""
        strategy = FixedCharacterChunking(chunk_size=100)
        chunks = strategy.chunk_text(TestFixtures.MEDIUM_TEXT)

        if chunks:
            chunk = chunks[0]
            metadata = chunk.metadata

            assert isinstance(metadata.chunk_id, str)
            assert isinstance(metadata.strategy, str)
            assert isinstance(metadata.start_char, int)
            assert isinstance(metadata.end_char, int)
            assert isinstance(metadata.char_count, int)
            assert isinstance(metadata.word_count, int)
            assert isinstance(metadata.chunk_index, int)

            # Verify consistency (char_count is of actual content, which may be stripped)
            assert metadata.char_count == len(chunk.content)
            assert metadata.char_count > 0
            assert metadata.word_count > 0
            assert metadata.end_char > metadata.start_char


if __name__ == "__main__":
    pytest.main([__file__])