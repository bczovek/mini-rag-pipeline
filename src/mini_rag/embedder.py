import logging
from typing import Any, Final

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

_logger: logging.Logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME: Final[str] = "intfloat/multilingual-e5-base"

_QUERY_PREFIX: Final[str] = "query: "
_PASSAGE_PREFIX: Final[str] = "passage: "


class E5Embeddings(HuggingFaceEmbeddings):

    def __init__(self, *, model_name: str, **kwargs: Any) -> None:  # noqa: ANN401
        super().__init__(model_name=model_name, **kwargs)

    def embed_query(self, text: str) -> list[float]:
        return super().embed_query(_QUERY_PREFIX + text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return super().embed_documents([_PASSAGE_PREFIX + text for text in texts])


def build_embeddings(model_name: str = DEFAULT_MODEL_NAME) -> HuggingFaceEmbeddings:
    return E5Embeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": True},
    )


class ChunkVectorStore:

    def __init__(self, embeddings: HuggingFaceEmbeddings | None = None) -> None:
        self._embeddings: HuggingFaceEmbeddings = embeddings or build_embeddings()
        self._vector_store: InMemoryVectorStore = InMemoryVectorStore(self._embeddings)
        self._chunk_count: int = 0

    def add_chunks(self, source_file: str, chunks: list[str]) -> None:
        _logger.info("Embedding document: %s (%d chunks)", source_file, len(chunks))
        documents = [
            Document(page_content=chunk, metadata={"source": source_file})
            for chunk in chunks
        ]
        self._vector_store.add_documents(documents)
        self._chunk_count += len(documents)
        _logger.info("Embedded document: %s", source_file)

    def add_documents(self, chunked_documents: dict[str, list[str]]) -> None:
        for source_file, chunks in chunked_documents.items():
            self.add_chunks(source_file, chunks)

    def __len__(self) -> int:
        return self._chunk_count

    def search(self, query: str, k: int) -> list[tuple[Document, float]]:
        return self._vector_store.similarity_search_with_score(query, k=k)

    def search_all(
        self, query: str, min_similarity: float
    ) -> list[tuple[Document, float]]:
        all_scored = self._vector_store.similarity_search_with_score(query, k=len(self))
        return [
            (document, score)
            for document, score in all_scored
            if score >= min_similarity
        ]
