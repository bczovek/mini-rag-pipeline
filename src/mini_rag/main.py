"""CLI entry point for the mini-RAG PDF ingestion pipeline.

Reads a JSON corpus manifest describing which PDF documents to ingest and
how to parse each one, then runs the PDF parser (see ``mini_rag.pdf_parser``)
against every listed file and splits its text into chunks (see
``mini_rag.text_splitter``). The manifest has common ``footerLinesPatterns``
and ``chunkSize`` settings shared by every document, plus a ``documents``
array listing each document's ``inputFile`` and ``skipPages``. The manifest
and the PDF files it references are expected to live side by side in the
same corpus directory, so the whole directory can be moved or installed
anywhere. The chunks are then embedded locally and held in an in-memory
vector store (see ``mini_rag.embedder``). Once every document has been
embedded, an interactive prompt reads queries from stdin, printing the most
similar chunks for each one until the user types "exit".

Usage:
    python -m mini_rag.main path/to/corpus.json
    python -m mini_rag.main path/to/corpus.json --top-k 8
    python -m mini_rag.main path/to/corpus.json --min-similarity 0.5
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Final

from mini_rag.embedder import ChunkVectorStore
from mini_rag.pdf_parser import parse
from mini_rag.text_splitter import SectionAwareTextSplitter

DEFAULT_CHUNK_SIZE: Final[int] = 512
DEFAULT_SEARCH_RESULTS: Final[int] = 4

_logger: logging.Logger = logging.getLogger(__name__)


def _resolve_input_file(input_file: str, corpus_path: Path) -> Path:
    """Resolve an ``inputFile`` entry from the corpus manifest.

    Relative paths are resolved against the manifest's own directory, since
    the manifest and the PDF files it references are siblings in the same
    corpus directory.
    """
    path = Path(input_file)
    if path.is_absolute():
        return path
    return (corpus_path.parent / path).resolve()


def run(corpus_path: Path) -> dict[str, list[str]]:
    """Parse and chunk every PDF file listed in the corpus manifest.

    Returns a mapping of each document's ``inputFile`` value to its list of
    text chunks.
    """
    manifest = json.loads(corpus_path.read_text(encoding="utf-8"))
    footer_line_patterns = manifest.get("footerLinesPatterns")
    splitter = SectionAwareTextSplitter(
        chunk_size=manifest.get("chunkSize", DEFAULT_CHUNK_SIZE)
    )

    chunked_documents: dict[str, list[str]] = {}
    for document in manifest.get("documents", []):
        input_file = document.get("inputFile", "")
        if not input_file:
            continue

        file_path = _resolve_input_file(input_file, corpus_path)
        text = parse(
            file_path,
            footer_line_patterns=footer_line_patterns,
            skip_pages=document.get("skipPages"),
        )

        _logger.info("Chunking document: %s", input_file)
        chunks = splitter.split(text)
        _logger.info("Chunked document: %s (%d chunks)", input_file, len(chunks))
        chunked_documents[input_file] = chunks

    return chunked_documents


def _print_search_results(
    chunk_store: ChunkVectorStore, query: str, top_k: int, min_similarity: float | None
) -> None:
    if min_similarity is not None:
        results = chunk_store.search_all(query, min_similarity=min_similarity)
    else:
        results = chunk_store.search(query, k=top_k)

    print("=" * 100)
    print(f"QUERY: {query} ({len(results)} results)")
    print("=" * 100)
    for rank, (document, score) in enumerate(results, start=1):
        source = document.metadata.get("source")
        print(
            f"--- result {rank} (source: {source}, cosine similarity: {score:.4f}) ---"
        )
        print(document.page_content)
        print()


def _run_query_loop(
    chunk_store: ChunkVectorStore, top_k: int, min_similarity: float | None
) -> None:
    """Read queries from stdin and print search results until the user
    types "exit" (or stdin is closed)."""
    while True:
        try:
            print("Enter a question to search the corpus, or type 'exit' to quit.")
            query = input("> ").strip()
        except EOFError:
            break

        if not query:
            continue
        if query.lower() == "exit":
            break

        _print_search_results(chunk_store, query, top_k, min_similarity)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )


def _parse_args() -> argparse.Namespace:
    arg_parser = argparse.ArgumentParser(
        description="Parse PDF documents listed in a mini-RAG corpus manifest.",
    )
    arg_parser.add_argument(
        "corpus", type=Path, help="Path to the corpus JSON manifest (e.g. corpus.json)."
    )
    arg_parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_SEARCH_RESULTS,
        help=(
            "Number of nearest chunks to show per query (default: "
            f"{DEFAULT_SEARCH_RESULTS}). Ignored if --min-similarity is given."
        ),
    )
    arg_parser.add_argument(
        "--min-similarity",
        type=float,
        help=(
            "If given, ignore --top-k and instead return every chunk whose "
            "cosine similarity to the query is at least this value (a "
            "float between -1 and 1), most similar first."
        ),
    )
    return arg_parser.parse_args()


def main() -> None:
    _configure_logging()
    args = _parse_args()

    corpus_path: Path = args.corpus
    if not corpus_path.exists():
        print(f"Corpus manifest not found: {corpus_path}", file=sys.stderr)
        sys.exit(1)

    chunked_documents = run(corpus_path)

    chunk_store = ChunkVectorStore()
    chunk_store.add_documents(chunked_documents)
    _run_query_loop(chunk_store, args.top_k, args.min_similarity)


if __name__ == "__main__":
    main()
