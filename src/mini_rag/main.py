"""CLI entry point for the mini-RAG PDF ingestion pipeline.

Reads a JSON corpus manifest describing which PDF documents to ingest and
how to parse each one, then runs the PDF parser (see ``mini_rag.pdf_parser``)
against every listed file and splits its text into chunks (see
``mini_rag.text_splitter``). The manifest and the PDF files it references
are expected to live side by side in the same corpus directory, so the
whole directory can be moved or installed anywhere.

Usage:
    python -m mini_rag.main path/to/corpus.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Final

from mini_rag.pdf_parser import parse
from mini_rag.text_splitter import SectionAwareTextSplitter

DEFAULT_CHUNK_SIZE: Final[int] = 1000


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

    Returns a mapping of each entry's ``inputFile`` value to its list of
    text chunks.
    """
    entries = json.loads(corpus_path.read_text(encoding="utf-8"))

    chunked_documents: dict[str, list[str]] = {}
    for entry in entries:
        input_file = entry.get("inputFile", "")
        if not input_file:
            continue

        file_path = _resolve_input_file(input_file, corpus_path)
        text = parse(
            file_path,
            footer_line_patterns=entry.get("footerLinesPatterns"),
            skip_pages=entry.get("skipPages"),
        )

        splitter = SectionAwareTextSplitter(
            chunk_size=entry.get("chunkSize", DEFAULT_CHUNK_SIZE)
        )
        chunked_documents[input_file] = splitter.split(text)

    return chunked_documents


def main() -> None:
    arg_parser = argparse.ArgumentParser(
        description="Parse PDF documents listed in a mini-RAG corpus manifest.",
    )
    arg_parser.add_argument(
        "corpus", type=Path, help="Path to the corpus JSON manifest (e.g. corpus.json)."
    )
    args = arg_parser.parse_args()

    corpus_path: Path = args.corpus
    if not corpus_path.exists():
        print(f"Corpus manifest not found: {corpus_path}", file=sys.stderr)
        sys.exit(1)

    chunked_documents = run(corpus_path)

    for input_file, chunks in chunked_documents.items():
        print("=" * 100)
        print(f"FILE: {input_file} ({len(chunks)} chunks)")
        print("=" * 100)
        for chunk_number, chunk_text in enumerate(chunks, start=1):
            print(f"--- chunk {chunk_number} ---")
            print(chunk_text)
            print()


if __name__ == "__main__":
    main()
