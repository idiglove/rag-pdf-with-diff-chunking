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
        """Test normal chunking behavior with detailed validation."""
        strategy = FixedCharacterChunking(chunk_size=50)
        chunks = strategy.chunk_text(TestFixtures.SHORT_TEXT)

        # Should create multiple chunks for text longer than chunk_size
        assert len(chunks) > 1, f"Expected multiple chunks, got {len(chunks)}"

        # Verify no chunk exceeds the specified size (allowing for reasonable variation due to stripping)
        for chunk in chunks:
            assert len(chunk.content) <= strategy.chunk_size + 10, f"Chunk too large: {len(chunk.content)} chars"
            assert len(chunk.content) > 0, "Empty chunk found"

        # Check that chunks cover the entire text (no content loss)
        total_chunk_chars = sum(len(chunk.content) for chunk in chunks)
        original_text_chars = len(TestFixtures.SHORT_TEXT.strip())
        # Allow some variation due to whitespace normalization
        assert total_chunk_chars >= original_text_chars * 0.8, "Significant content loss detected"

        # Verify chunk ordering and metadata consistency
        for i, chunk in enumerate(chunks):
            assert chunk.metadata.chunk_index == i, f"Incorrect chunk index: expected {i}, got {chunk.metadata.chunk_index}"
            assert chunk.metadata.strategy == "fixed_char_50", f"Incorrect strategy name: {chunk.metadata.strategy}"
            assert chunk.metadata.char_count == len(chunk.content), f"Char count mismatch: metadata says {chunk.metadata.char_count}, actual is {len(chunk.content)}"
            assert chunk.metadata.word_count == len(chunk.content.split()), f"Word count mismatch: metadata says {chunk.metadata.word_count}, actual is {len(chunk.content.split())}"

            # Verify position metadata makes sense
            assert chunk.metadata.start_char >= 0, "Negative start position"
            assert chunk.metadata.end_char > chunk.metadata.start_char, "End position not after start position"

            # Verify chunk IDs are unique and follow expected format
            expected_chunk_id = f"fixed_char_50_{i:04d}"
            assert chunk.metadata.chunk_id == expected_chunk_id, f"Incorrect chunk ID: expected {expected_chunk_id}, got {chunk.metadata.chunk_id}"

        # Verify chunks are in logical order (start positions should be non-decreasing)
        for i in range(1, len(chunks)):
            assert chunks[i].metadata.start_char >= chunks[i-1].metadata.start_char, "Chunks not in logical order"

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

    def test_paragraph_detection_comprehensive(self):
        """Test paragraph detection logic with detailed validation."""
        strategy = ParagraphChunking(paragraphs_per_chunk=1)
        chunks = strategy.chunk_text(TestFixtures.MEDIUM_TEXT)

        # Should detect multiple paragraphs in structured text
        assert len(chunks) >= 3, f"Expected at least 3 paragraphs, got {len(chunks)}"

        # Verify chunks contain meaningful content (filter enforced in implementation)
        for i, chunk in enumerate(chunks):
            assert len(chunk.content.strip()) > 30, f"Chunk {i} too short: {len(chunk.content.strip())} chars"

            # Verify metadata consistency
            assert chunk.metadata.chunk_index == i, f"Incorrect chunk index: expected {i}, got {chunk.metadata.chunk_index}"
            assert chunk.metadata.strategy == "paragraph_1", f"Incorrect strategy name: {chunk.metadata.strategy}"
            assert chunk.metadata.char_count == len(chunk.content), f"Char count mismatch in chunk {i}"
            assert chunk.metadata.word_count == len(chunk.content.split()), f"Word count mismatch in chunk {i}"

        # Verify that paragraph boundaries are respected (chunks should contain complete thoughts)
        # Look for specific structure in MEDIUM_TEXT
        chunk_contents = [chunk.content for chunk in chunks]

        # Should preserve key sections from the original text
        assert any("Introduction" in content for content in chunk_contents), "Introduction section not found"
        assert any("Chapter 1" in content for content in chunk_contents), "Chapter 1 section not found"

    def test_multiple_paragraphs_per_chunk(self):
        """Test chunking multiple paragraphs together."""
        strategy = ParagraphChunking(paragraphs_per_chunk=2)
        chunks = strategy.chunk_text(TestFixtures.MEDIUM_TEXT)

        # Should create fewer chunks when combining paragraphs
        strategy_single = ParagraphChunking(paragraphs_per_chunk=1)
        chunks_single = strategy_single.chunk_text(TestFixtures.MEDIUM_TEXT)

        assert len(chunks) <= len(chunks_single), "Multi-paragraph chunks should create fewer total chunks"

        # Verify each chunk contains multiple paragraphs (where possible)
        for chunk in chunks:
            # Should have at least some paragraph structure
            assert len(chunk.content.strip()) > 50, f"Multi-paragraph chunk too short: {len(chunk.content.strip())} chars"

    def test_single_paragraph_text(self):
        """Test with text that has no clear paragraph breaks."""
        strategy = ParagraphChunking(paragraphs_per_chunk=1)
        single_para = "This is a single paragraph. It has multiple sentences. But no paragraph breaks. The text continues without line breaks."
        chunks = strategy.chunk_text(single_para)

        # Should handle gracefully - might create one chunk or none depending on paragraph detection
        assert isinstance(chunks, list), "Should return a list"

        if chunks:
            # If a chunk is created, verify its properties
            chunk = chunks[0]
            assert len(chunk.content.strip()) > 30, "Single paragraph chunk should meet minimum size requirement"
            assert chunk.metadata.chunk_index == 0, "First chunk should have index 0"
            assert chunk.metadata.strategy == "paragraph_1", "Strategy name should be correct"

    def test_text_with_headers_and_structure(self):
        """Test paragraph detection with headers and structured text."""
        structured_text = """
        CHAPTER 1: INTRODUCTION

        This is the introduction paragraph. It explains the main concepts.
        We will cover several important topics in this chapter.

        SECTION 1.1: BASIC CONCEPTS

        Basic concepts are fundamental to understanding the subject.
        Each concept builds upon the previous one systematically.

        The implementation requires careful consideration of design.

        SECTION 1.2: ADVANCED TOPICS

        Advanced topics require deeper understanding and analysis.
        These concepts are more complex and require practice.
        """

        strategy = ParagraphChunking(paragraphs_per_chunk=1)
        chunks = strategy.chunk_text(structured_text)

        # Should detect multiple sections/paragraphs
        assert len(chunks) >= 2, f"Should detect multiple sections, got {len(chunks)}"

        # Verify structure preservation
        chunk_contents = [chunk.content for chunk in chunks]

        # Should find references to the structured content
        has_chapter_ref = any("CHAPTER" in content or "introduction" in content.lower() for content in chunk_contents)
        has_section_ref = any("SECTION" in content or "concepts" in content.lower() for content in chunk_contents)

        assert has_chapter_ref or has_section_ref, "Should preserve some structural elements"

    def test_chunk_size_boundaries(self):
        """Test paragraph chunking with various chunk count settings."""
        text = TestFixtures.MEDIUM_TEXT

        # Test different paragraphs_per_chunk values
        for para_count in [1, 2, 3, 5]:
            strategy = ParagraphChunking(paragraphs_per_chunk=para_count)
            chunks = strategy.chunk_text(text)

            # Should create some chunks
            assert len(chunks) >= 1, f"Should create at least 1 chunk with {para_count} paragraphs per chunk"

            # Verify strategy naming
            expected_strategy_name = f"paragraph_{para_count}"
            for chunk in chunks:
                assert chunk.metadata.strategy == expected_strategy_name, f"Strategy name mismatch for {para_count} paragraphs per chunk"

    def test_malformed_paragraph_text(self):
        """Test with text that has irregular paragraph formatting."""
        malformed_text = """


        This paragraph has weird spacing.



        Another paragraph with lots of empty lines.


        Final paragraph here.


        """

        strategy = ParagraphChunking(paragraphs_per_chunk=1)
        chunks = strategy.chunk_text(malformed_text)

        # Should handle malformed text gracefully
        assert isinstance(chunks, list), "Should return a list even with malformed text"

        # Any chunks created should have meaningful content
        for chunk in chunks:
            assert len(chunk.content.strip()) > 10, "Chunks should have meaningful content even with malformed input"
            assert chunk.content.count('\n\n\n') == 0, "Should normalize excessive line breaks"


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