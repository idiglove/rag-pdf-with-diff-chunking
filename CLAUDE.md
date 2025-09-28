# RAG System with Different Chunking Strategies Evaluation

## Project Overview

A comprehensive RAG (Retrieval-Augmented Generation) system designed to evaluate different chunking strategies for processing a single book. The project aims to measure and compare the effectiveness of various chunking approaches in terms of:

- **Quickness**: Response time and retrieval speed
- **Relevance**: Quality and accuracy of retrieved information
- **Expense**: Computational cost and token usage

## Architecture Components

### 1. Data Processing Pipeline

- PDF/text book ingestion
- Multiple chunking strategy implementations
- Vector embedding generation
- Vector database storage (ChromaDB, Pinecone, or Weaviate)

### 2. Chunking Strategies to Evaluate

#### Fixed-Size Chunking

- **Character-based**: Fixed character count (e.g., 500, 1000, 2000 chars)
- **Token-based**: Fixed token count (e.g., 256, 512, 1024 tokens)
- **Sentence-based**: Fixed number of sentences per chunk

#### Semantic Chunking

- **Paragraph-based**: Natural paragraph boundaries
- **Section-based**: Chapter/section headers as boundaries
- **Topic-based**: Using NLP to identify topic shifts
- **Sentence embedding similarity**: Group sentences with similar embeddings

#### Hybrid Approaches

- **Sliding window**: Overlapping chunks with configurable overlap
- **Hierarchical chunking**: Multiple granularity levels
- **Adaptive chunking**: Size varies based on content complexity
- **Recursive chunking**: Split by structure, then by size if needed

#### Advanced Strategies

- **Semantic similarity clustering**: Group semantically related content
- **Named entity preservation**: Ensure entities aren't split across chunks
- **Context-aware chunking**: Maintain important context within chunks
- **Question-answer focused**: Optimize for common Q&A patterns

### 3. Backend System

- FastAPI/Flask REST API
- Multiple vector stores for comparison
- Chunking strategy manager
- Evaluation metrics collector
- LLM integration (OpenAI, Anthropic, or local models)

### 4. Frontend Interface

- React/Next.js chat interface
- Strategy selection dropdown
- Real-time performance metrics display
- Comparative analysis dashboard
- Export functionality for results

## Evaluation Metrics

### Performance Metrics

- **Retrieval Speed**: Time to find relevant chunks
- **Generation Speed**: Time to generate complete response
- **Total Response Time**: End-to-end latency
- **Throughput**: Queries per second

### Quality Metrics

- **Relevance Score**: Semantic similarity between query and retrieved chunks
- **Answer Accuracy**: Human evaluation or automated scoring
- **Context Completeness**: Whether retrieved chunks contain sufficient context
- **Hallucination Rate**: Frequency of generated false information

### Cost Metrics

- **Token Usage**: Input/output tokens per query
- **API Costs**: Dollar cost per query
- **Storage Requirements**: Vector database size per strategy
- **Computational Resources**: CPU/memory usage

### Specialized Metrics

- **Chunk Utilization**: How often each chunk is retrieved
- **Cross-Reference Accuracy**: Ability to connect related concepts
- **Question Type Performance**: Performance across different query types
- **Book Coverage**: Percentage of book content accessible through queries

## Implementation Plan

### Phase 1: Core Infrastructure

1. Set up project structure
2. Implement basic PDF processing
3. Create simple fixed-size chunking
4. Set up vector database
5. Build basic chat interface

### Phase 2: Chunking Strategy Implementation

1. Implement all planned chunking strategies
2. Create strategy comparison framework
3. Add performance monitoring
4. Build evaluation pipeline

### Phase 3: Advanced Features

1. Add real-time metrics dashboard
2. Implement A/B testing framework
3. Create export/reporting functionality
4. Add strategy recommendation system

### Phase 4: Analysis & Optimization

1. Conduct comprehensive evaluation
2. Analyze trade-offs between strategies
3. Document findings and recommendations
4. Optimize best-performing strategies

## Project Milestones

### Milestone 1: Foundation Setup (Week 1)

**Goal**: Establish basic project infrastructure and simple RAG pipeline

**Todos**:

- [x] Initialize project repository with proper folder structure
- [x] Set up Python virtual environment and requirements.txt
- [x] Create basic FastAPI application with health check endpoint
- [x] Implement PDF text extraction using PyMuPDF or pdfplumber
- [x] Select and download a suitable test book (300+ pages, non-fiction)
- [x] Create basic character-based chunking function (1000 characters)
- [x] Set up ChromaDB for vector storage
- [x] Implement basic text embedding using sentence-transformers
- [x] Create simple storage and retrieval functions
- [x] Test end-to-end pipeline with sample queries

**Deliverables**:

- Working PDF processing pipeline
- Basic vector storage and retrieval
- Simple command-line query interface

### Milestone 1.1: Debug Chunking Strategy Issues (Immediate)

**Goal**: Fix critical bugs discovered in initial chunking strategies

**Issues Identified**:

- paragraph_3 strategy only produces 1 chunk (should be many more)
- Other strategies may have missing chunks or incorrect counts
- Need to validate chunking logic and text preprocessing

**Todos**:

- [x] Debug paragraph_3 strategy - investigate why only 1 chunk generated
- [x] Analyze text preprocessing and paragraph detection logic
- [x] Test sentence_8 strategy for missing chunks
- [x] Verify fixed_char_1000 strategy chunk boundaries
- [x] Fix paragraph splitting regex pattern
- [x] Add better text normalization and cleaning
- [x] Test all strategies with sample text to ensure proper chunking
- [x] Update chunk count validation and logging

**Deliverables**:

- Fixed paragraph chunking strategy with proper multi-chunk output
- Validated chunking strategies with correct chunk counts
- Improved text preprocessing pipeline
- Updated chunking strategy documentation

### Milestone 2: Core Chunking Strategies (Week 2)

**Goal**: Implement and test multiple chunking approaches

**Todos**:

- [ ] Implement fixed-size chunking variants (character, token, sentence)
- [ ] Create paragraph-based semantic chunking
- [ ] Implement section-based chunking using headers
- [ ] Build sliding window chunking with overlap
- [ ] Create recursive chunking strategy
- [ ] Implement basic topic-based chunking using spaCy
- [ ] Add chunking strategy configuration system
- [ ] Create chunk metadata tracking (strategy, size, position)
- [ ] Build chunk visualization tool for debugging
- [ ] Test each strategy with sample book content

**Deliverables**:

- 6+ different chunking strategies implemented
- Strategy configuration and switching system
- Chunk analysis and debugging tools

### Milestone 3: Performance Monitoring (Week 3)

**Goal**: Build comprehensive evaluation and monitoring system

**Todos**:

- [ ] Design and implement metrics collection framework
- [ ] Add response time measurement for retrieval and generation
- [ ] Implement relevance scoring using semantic similarity
- [ ] Create token usage tracking for cost analysis
- [ ] Build chunk utilization analytics
- [ ] Add query type classification system
- [ ] Implement automated evaluation pipeline
- [ ] Create performance comparison utilities
- [ ] Build metrics export functionality (CSV/JSON)
- [ ] Design evaluation query dataset (50+ diverse questions)

**Deliverables**:

- Comprehensive metrics collection system
- Automated evaluation pipeline
- Performance comparison framework

### Milestone 4: Web Interface Development (Week 4)

**Goal**: Create user-friendly frontend for testing and comparison

**Todos**:

- [ ] Set up Next.js project with TypeScript
- [ ] Design and implement chat interface with Tailwind CSS
- [ ] Create chunking strategy selector dropdown
- [ ] Build real-time metrics display components
- [ ] Implement query history and bookmarking
- [ ] Add response quality rating system
- [ ] Create comparative analysis dashboard
- [ ] Build strategy performance visualization charts
- [ ] Implement export functionality for results
- [ ] Add responsive design for mobile devices

**Deliverables**:

- Fully functional web interface
- Real-time performance monitoring
- Comparative analysis dashboard

### Milestone 5: Advanced Chunking & Features (Week 5)

**Goal**: Implement sophisticated chunking strategies and advanced features

**Todos**:

- [ ] Implement named entity preservation chunking
- [ ] Create semantic similarity clustering approach
- [ ] Build context-aware chunking using transformers
- [ ] Implement hierarchical chunking with multiple levels
- [ ] Add adaptive chunking based on content complexity
- [ ] Create question-answer optimized chunking
- [ ] Implement A/B testing framework for strategies
- [ ] Add strategy recommendation system
- [ ] Build automated hyperparameter tuning
- [ ] Create strategy ensemble methods

**Deliverables**:

- Advanced chunking strategies
- A/B testing framework
- Strategy optimization system

### Milestone 6: Comprehensive Evaluation (Week 6)

**Goal**: Conduct thorough analysis and generate insights

**Todos**:

- [ ] Design comprehensive evaluation protocol
- [ ] Create diverse query dataset (200+ questions)
- [ ] Run systematic performance comparison across all strategies
- [ ] Conduct cost-benefit analysis for each approach
- [ ] Analyze performance by query type and complexity
- [ ] Generate statistical significance tests
- [ ] Create detailed performance reports
- [ ] Document trade-offs and recommendations
- [ ] Build strategy selection decision tree
- [ ] Prepare findings presentation

**Deliverables**:

- Comprehensive evaluation report
- Strategy recommendation system
- Performance insights and trade-off analysis

### Milestone 7: Optimization & Polish (Week 7)

**Goal**: Optimize best strategies and finalize project

**Todos**:

- [ ] Optimize top-performing chunking strategies
- [ ] Implement production-ready error handling
- [ ] Add comprehensive logging and monitoring
- [ ] Create deployment configuration (Docker/Docker Compose)
- [ ] Write comprehensive documentation
- [ ] Add unit and integration tests
- [ ] Implement caching for improved performance
- [ ] Create user guide and API documentation
- [ ] Prepare code for open-source release
- [ ] Final testing and bug fixes

**Deliverables**:

- Production-ready RAG system
- Complete documentation
- Deployment package

## Technology Stack

### Backend

- **Language**: Python
- **Framework**: FastAPI
- **Vector DB**: ChromaDB, Pinecone, or Weaviate
- **Embeddings**: OpenAI text-embedding-ada-002 or sentence-transformers
- **LLM**: OpenAI GPT-4, Claude, or local models

### Frontend

- **Framework**: React with Next.js
- **Styling**: Tailwind CSS
- **Charts**: Chart.js or Recharts
- **State Management**: Zustand or Redux Toolkit

### Data Processing

- **PDF Processing**: PyMuPDF, pdfplumber, or langchain
- **Text Processing**: spaCy, NLTK, or transformers
- **Chunking**: Custom implementations + langchain utilities

## Sample Book Selection Criteria

- **Length**: Substantial content (200+ pages)
- **Structure**: Clear chapters/sections
- **Content Type**: Mix of narrative and factual information
- **Complexity**: Varied sentence lengths and concepts
- **Genre**: Non-fiction for objective evaluation

## Success Metrics

- Successfully implement 8+ different chunking strategies
- Demonstrate measurable differences in performance metrics
- Create actionable insights about chunking strategy trade-offs
- Build reusable framework for future RAG projects
- Generate comprehensive analysis report

## Learning Objectives

- Understand impact of chunking on RAG performance
- Learn trade-offs between speed, relevance, and cost
- Gain experience with vector databases and embeddings
- Practice building full-stack ML applications
- Develop evaluation methodologies for RAG systems

## Claude Todos

- Whenever a todo is done, check it off the claude.md
