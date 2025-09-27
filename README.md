# RAG Chunking Strategy Evaluation

A comprehensive system to evaluate different chunking strategies for Retrieval-Augmented Generation (RAG) systems.

## Project Structure

```
rag-pdf-with-diff-chunking/
├── backend/                 # Python FastAPI backend
│   ├── src/
│   │   ├── chunking/       # Chunking strategy implementations
│   │   ├── embedding/      # Text embedding utilities
│   │   ├── storage/        # Vector database operations
│   │   ├── evaluation/     # Performance metrics and evaluation
│   │   └── api/           # FastAPI routes and endpoints
│   ├── tests/             # Backend tests
│   └── config/            # Configuration files
├── frontend/              # Next.js frontend
│   ├── src/
│   │   ├── components/    # React components
│   │   ├── pages/         # Next.js pages
│   │   ├── hooks/         # Custom React hooks
│   │   └── utils/         # Frontend utilities
│   └── public/            # Static assets
├── data/
│   ├── books/             # Source PDF books
│   ├── processed/         # Processed text chunks
│   └── vectors/           # Vector database files
├── docs/                  # Documentation
├── notebooks/             # Jupyter notebooks for analysis
└── scripts/               # Utility scripts
```

## Getting Started

See CLAUDE.md for detailed project planning and milestones.

## Current Status

🔄 **Milestone 1: Foundation Setup** - In Progress
- ✅ Project structure created
- ⏳ Setting up Python environment...