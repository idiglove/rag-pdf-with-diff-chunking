#!/usr/bin/env python3
"""
Command-line interface for testing the RAG pipeline.
Provides simple commands to process documents and run queries.
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from rag_pipeline import RAGPipeline


def process_document_command(pipeline, args):
    """Process a PDF document with chunking strategies."""
    print(f"Processing document: {args.pdf_path}")
    print(f"Document name: {args.name}")

    # Check if PDF file exists
    if not Path(args.pdf_path).exists():
        print(f"Error: PDF file not found: {args.pdf_path}")
        return

    strategies = args.strategies.split(',') if args.strategies else None
    if strategies:
        print(f"Using strategies: {strategies}")
    else:
        print("Using all available strategies")

    try:
        result = pipeline.process_document(args.pdf_path, args.name, strategies)

        print(f"\nProcessing completed in {result['processing_time']:.2f} seconds")
        print(f"Text length: {result['text_length']:,} characters")
        print(f"Word count: {result['word_count']:,} words")

        print("\nChunking results:")
        for strategy, data in result['strategies'].items():
            if 'error' in data:
                print(f"  {strategy}: ERROR - {data['error']}")
            else:
                print(f"  {strategy}: {data['chunk_count']} chunks")
                stats = data['stats']
                print(f"    - Avg chars per chunk: {stats['avg_chars_per_chunk']:.1f}")
                print(f"    - Avg words per chunk: {stats['avg_words_per_chunk']:.1f}")

    except Exception as e:
        print(f"Error processing document: {e}")


def query_command(pipeline, args):
    """Run a query against processed document."""
    print(f"Querying: '{args.query}'")
    print(f"Document: {args.document}")
    print(f"Strategy: {args.strategy}")

    # Check if document exists
    if args.document not in pipeline.list_documents():
        print(f"Error: Document '{args.document}' not found.")
        print("Available documents:")
        documents = pipeline.list_documents()
        if documents:
            for doc in documents:
                print(f"  - {doc}")
        else:
            print("  No documents processed yet. Use 'python cli.py process' to add documents.")
        return

    # Check if strategy exists for this document
    doc_info = pipeline.get_document_info(args.document)
    if 'error' not in doc_info and args.strategy not in doc_info['strategies']:
        print(f"Error: Strategy '{args.strategy}' not found for document '{args.document}'.")
        print("Available strategies for this document:")
        for strategy in doc_info['strategies'].keys():
            print(f"  - {strategy}")
        return

    try:
        result = pipeline.query(args.query, args.document, args.strategy, args.top_k)

        if 'error' in result:
            print(f"Query error: {result['error']}")
            print("This might happen if the collection doesn't exist or contains no data.")
            return

        print(f"\nQuery completed in {result['query_time']:.3f} seconds")
        print(f"Found {len(result['chunks'])} relevant chunks:")

        for i, chunk in enumerate(result['chunks']):
            print(f"\n--- Chunk {i + 1} ---")
            print(f"Similarity: {chunk['similarity']:.3f}")
            print(f"Chunk ID: {chunk['chunk_id']}")
            print(f"Content: {chunk['content'][:200]}...")
            if len(chunk['content']) > 200:
                print("(truncated)")

    except Exception as e:
        print(f"Error during query: {e}")


def compare_command(pipeline, args):
    """Compare query results across strategies."""
    print(f"Comparing strategies for query: '{args.query}'")
    print(f"Document: {args.document}")

    # Check if document exists
    if args.document not in pipeline.list_documents():
        print(f"Error: Document '{args.document}' not found.")
        print("Available documents:")
        documents = pipeline.list_documents()
        if documents:
            for doc in documents:
                print(f"  - {doc}")
        else:
            print("  No documents processed yet. Use 'python cli.py process' to add documents.")
        return

    strategies = args.strategies.split(',') if args.strategies else None
    if strategies:
        print(f"Comparing strategies: {strategies}")
        # Validate strategies exist for this document
        doc_info = pipeline.get_document_info(args.document)
        if 'error' not in doc_info:
            available_strategies = set(doc_info['strategies'].keys())
            invalid_strategies = set(strategies) - available_strategies
            if invalid_strategies:
                print(f"Error: Invalid strategies for document '{args.document}': {invalid_strategies}")
                print("Available strategies for this document:")
                for strategy in available_strategies:
                    print(f"  - {strategy}")
                return

    try:
        result = pipeline.compare_strategies(args.query, args.document, strategies, args.top_k)

        if 'error' in result:
            print(f"Comparison error: {result['error']}")
            return

        print(f"\nComparison completed in {result['total_time']:.3f} seconds")

        if 'summary' in result:
            summary = result['summary']
            print(f"Fastest strategy: {summary['fastest_strategy']}")
            print(f"Average query time: {summary['avg_query_time']:.3f}s")

        print("\nResults by strategy:")
        for strategy, strategy_result in result['strategies'].items():
            if 'error' in strategy_result:
                print(f"\n{strategy}: ERROR - {strategy_result['error']}")
                continue

            print(f"\n{strategy} (query time: {strategy_result['query_time']:.3f}s):")
            for i, chunk in enumerate(strategy_result['chunks'][:2]):  # Show top 2
                print(f"  {i+1}. Similarity: {chunk['similarity']:.3f}")
                print(f"     Content: {chunk['content'][:100]}...")

    except Exception as e:
        print(f"Error during comparison: {e}")


def list_command(pipeline, args):
    """List documents and strategies."""
    print("Processed documents:")
    documents = pipeline.list_documents()
    if documents:
        for doc in documents:
            print(f"  - {doc}")
            doc_info = pipeline.get_document_info(doc)
            strategies = list(doc_info['strategies'].keys())
            print(f"    Strategies: {', '.join(strategies)}")
    else:
        print("  No documents processed yet")

    print(f"\nAvailable chunking strategies:")
    for strategy in pipeline.list_available_strategies():
        print(f"  - {strategy}")


def info_command(pipeline, args):
    """Show detailed information about a document."""
    doc_info = pipeline.get_document_info(args.document)

    if 'error' in doc_info:
        print(f"Error: {doc_info['error']}")
        return

    print(f"Document: {doc_info['document_name']}")
    print(f"PDF path: {doc_info['pdf_path']}")
    print(f"Text length: {doc_info['text_length']:,} characters")
    print(f"Word count: {doc_info['word_count']:,} words")
    print(f"Processing time: {doc_info['processing_time']:.2f} seconds")

    if 'metadata' in doc_info:
        metadata = doc_info['metadata']
        print(f"\nPDF Metadata:")
        for key, value in metadata.items():
            if value:
                print(f"  {key}: {value}")

    print(f"\nChunking strategies ({len(doc_info['strategies'])}):")
    for strategy, data in doc_info['strategies'].items():
        if 'error' in data:
            print(f"  {strategy}: ERROR - {data['error']}")
        else:
            print(f"  {strategy}:")
            print(f"    - Chunks: {data['chunk_count']}")
            print(f"    - Collection: {data['collection_name']}")
            stats = data['stats']
            print(f"    - Avg chars/chunk: {stats['avg_chars_per_chunk']:.1f}")
            print(f"    - Avg words/chunk: {stats['avg_words_per_chunk']:.1f}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="RAG Pipeline CLI for testing chunking strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Typical Workflow:
  1. FIRST: Process a PDF document to create chunks and embeddings
  2. THEN: List, query, or compare the processed documents

Examples:

  # STEP 1: Process a PDF document (REQUIRED FIRST)
  python cli.py process path/to/book.pdf --name my_book

  # Process with specific strategies only
  python cli.py process book.pdf --name test --strategies "fixed_char_1000,sentence_8"

  # STEP 2: Verify processing worked
  python cli.py list

  # STEP 3: Query the processed document
  python cli.py query "What is machine learning?" --document my_book --strategy fixed_char_1000

  # Compare different strategies for the same query
  python cli.py compare "What is AI?" --document my_book

  # Show detailed document information
  python cli.py info --document my_book

Note: You must process a document before you can query or compare it.
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Process command
    process_parser = subparsers.add_parser('process', help='Process a PDF document')
    process_parser.add_argument('pdf_path', help='Path to PDF file')
    process_parser.add_argument('--name', required=True, help='Document name identifier')
    process_parser.add_argument('--strategies', help='Comma-separated list of chunking strategies')

    # Query command
    query_parser = subparsers.add_parser('query', help='Query a processed document')
    query_parser.add_argument('query', help='Search query text')
    query_parser.add_argument('--document', required=True, help='Document name to search')
    query_parser.add_argument('--strategy', required=True, help='Chunking strategy to use')
    query_parser.add_argument('--top_k', type=int, default=5, help='Number of results to return')

    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare strategies for a query')
    compare_parser.add_argument('query', help='Search query text')
    compare_parser.add_argument('--document', required=True, help='Document name to search')
    compare_parser.add_argument('--strategies', help='Comma-separated list of strategies to compare')
    compare_parser.add_argument('--top_k', type=int, default=3, help='Number of results per strategy')

    # List command
    list_parser = subparsers.add_parser('list', help='List processed documents and available strategies')

    # Info command
    info_parser = subparsers.add_parser('info', help='Show detailed document information')
    info_parser.add_argument('--document', required=True, help='Document name')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Initialize pipeline
    print("Initializing RAG pipeline...")
    try:
        pipeline = RAGPipeline()
        print("Pipeline initialized successfully!\n")
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        return

    # Execute command
    if args.command == 'process':
        process_document_command(pipeline, args)
    elif args.command == 'query':
        query_command(pipeline, args)
    elif args.command == 'compare':
        compare_command(pipeline, args)
    elif args.command == 'list':
        list_command(pipeline, args)
    elif args.command == 'info':
        info_command(pipeline, args)


if __name__ == "__main__":
    main()