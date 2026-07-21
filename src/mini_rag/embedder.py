"""Chunk and query embedding for the mini-RAG pipeline.

Embeds text chunks and queries locally with the multilingual
``sentence-transformers`` model ``paraphrase-multilingual-MiniLM-L12-v2``
(no external API calls) and keeps the resulting vectors in an in-memory
LangChain vector store for similarity search.
"""

import logging
from typing import Final

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

_logger: logging.Logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME: Final[str] = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


def build_embeddings(model_name: str = DEFAULT_MODEL_NAME) -> HuggingFaceEmbeddings:
    """Create a local ``HuggingFaceEmbeddings`` instance for the given model.

    Embeddings are L2-normalized so that similarity search (which uses
    cosine similarity) behaves consistently.
    """
    return HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": True},
    )


class ChunkStore:
    """An in-memory vector store of embedded document chunks.

    Wraps a LangChain ``InMemoryVectorStore`` so chunks from multiple
    source files can be embedded once and queried by similarity, without
    persisting anything to disk.
    """

    def __init__(self, embeddings: HuggingFaceEmbeddings | None = None) -> None:
        self._embeddings: HuggingFaceEmbeddings = embeddings or build_embeddings()
        self._vector_store: InMemoryVectorStore = InMemoryVectorStore(self._embeddings)

    def add_chunks(self, source_file: str, chunks: list[str]) -> None:
        """Embed and store the chunks belonging to one source file."""
        _logger.info("Embedding document: %s (%d chunks)", source_file, len(chunks))
        documents = [
            Document(page_content=chunk, metadata={"source": source_file})
            for chunk in chunks
        ]
        self._vector_store.add_documents(documents)
        _logger.info("Embedded document: %s", source_file)

    def add_documents(self, chunked_documents: dict[str, list[str]]) -> None:
        """Embed and store chunks for every source file in the mapping."""
        for source_file, chunks in chunked_documents.items():
            self.add_chunks(source_file, chunks)

    def search(self, query: str, k: int = 4) -> list[Document]:
        """Return the ``k`` chunks most similar to the query."""
        return self._vector_store.similarity_search(query, k=k)
