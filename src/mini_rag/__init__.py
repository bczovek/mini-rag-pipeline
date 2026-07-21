"""mini_rag: A minimal RAG pipeline for the Magyar Telekom lakossági ÁSZF."""

from mini_rag.embedder import ChunkVectorStore, build_embeddings
from mini_rag.pdf_parser import parse
from mini_rag.text_splitter import SectionAwareTextSplitter

__all__ = ["ChunkVectorStore", "SectionAwareTextSplitter", "build_embeddings", "parse"]
