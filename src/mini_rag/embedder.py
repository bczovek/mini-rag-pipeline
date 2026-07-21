"""Chunk and query embedding for the mini-RAG pipeline.

Embeds text chunks and queries locally with the multilingual, retrieval-tuned
``sentence-transformers`` model ``intfloat/multilingual-e5-base`` (no
external API calls) and keeps the resulting vectors in an in-memory
LangChain vector store for similarity search.
"""

import logging
from typing import Final

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

_logger: logging.Logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME: Final[str] = "intfloat/multilingual-e5-base"

_QUERY_PREFIX: Final[str] = "query: "
_PASSAGE_PREFIX: Final[str] = "passage: "


class E5Embeddings(HuggingFaceEmbeddings):
    """``HuggingFaceEmbeddings`` for the E5 family of embedding models.

    E5 models (e.g. ``intfloat/multilingual-e5-base``,
    ``intfloat/multilingual-e5-large``) are trained asymmetrically: queries
    and passages must be prefixed with ``"query: "`` and ``"passage: "``
    respectively so the model can distinguish the two roles. This subclass
    adds those prefixes transparently before delegating to the base class.
    """

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query, applying the required ``"query: "`` prefix."""
        return super().embed_query(_QUERY_PREFIX + text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed document chunks, applying the required ``"passage: "`` prefix."""
        return super().embed_documents([_PASSAGE_PREFIX + text for text in texts])


def build_embeddings(model_name: str = DEFAULT_MODEL_NAME) -> HuggingFaceEmbeddings:
    """Create a local ``HuggingFaceEmbeddings`` instance for the given model.

    Embeddings are L2-normalized so that similarity search (which uses
    cosine similarity) behaves consistently. E5 models additionally require
    ``"query: "``/``"passage: "`` prefixes, which :class:`E5Embeddings`
    applies automatically.
    """
    return E5Embeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": True},
    )


class ChunkVectorStore:
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

    def __len__(self) -> int:
        """Return the number of chunks currently held in the store."""
        return len(self._vector_store.store)

    def search(self, query: str, k: int = 4) -> list[tuple[Document, float]]:
        """Return the ``k`` chunks most similar to the query.

        Each result is paired with its cosine similarity score to the
        query, highest first.
        """
        return self._vector_store.similarity_search_with_score(query, k=k)

    def search_all(
        self, query: str, min_similarity: float
    ) -> list[tuple[Document, float]]:
        """Return every stored chunk whose cosine similarity to the query
        is at least ``min_similarity``, most similar first.

        Unlike :meth:`search`, this isn't capped to a fixed count: it
        relies on the underlying vector store always ranking candidates by
        cosine similarity (``InMemoryVectorStore`` does), so scoring every
        stored chunk against the query and filtering by the threshold gives
        every match the collection supports finding, not just the top few.
        """
        all_scored = self._vector_store.similarity_search_with_score(query, k=len(self))
        return [
            (document, score)
            for document, score in all_scored
            if score >= min_similarity
        ]
