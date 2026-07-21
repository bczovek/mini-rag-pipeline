"""CLI entry point for the mini-RAG PDF ingestion pipeline.

Reads a JSON config file describing which PDF documents to ingest and how
to parse each one, then runs the PDF parser (see ``mini_rag.pdf_parser``)
against every listed file.

Usage:
    python -m mini_rag.main path/to/config.json
"""

import argparse
import json
import sys
from pathlib import Path

from mini_rag.pdf_parser import parse


def _resolve_input_file(input_file: str, config_path: Path) -> Path:
    """Resolve an ``inputFile`` entry from the config file.

    Relative paths are resolved against the config file's own directory,
    so the config can be run from any working directory.
    """
    path = Path(input_file)
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()


def run(config_path: Path) -> dict[str, str]:
    """Parse every PDF file listed in the config file.

    Returns a mapping of each entry's ``inputFile`` value to its fully
    parsed text.
    """
    entries = json.loads(config_path.read_text(encoding="utf-8"))

    parsed_documents: dict[str, str] = {}
    for entry in entries:
        input_file = entry.get("inputFile", "")
        if not input_file:
            continue

        file_path = _resolve_input_file(input_file, config_path)
        parsed_documents[input_file] = parse(
            file_path,
            footer_line_patterns=entry.get("footerLinesPatterns"),
            skip_pages=entry.get("skipPages"),
        )

    return parsed_documents


def main() -> None:
    arg_parser = argparse.ArgumentParser(
        description="Parse PDF documents listed in a mini-RAG config file.",
    )
    arg_parser.add_argument("config", type=Path, help="Path to the JSON config file.")
    args = arg_parser.parse_args()

    config_path: Path = args.config
    if not config_path.exists():
        print(f"Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    parsed_documents = run(config_path)

    for input_file, text in parsed_documents.items():
        print("=" * 100)
        print(f"FILE: {input_file}")
        print("=" * 100)
        print(text)
        print()


if __name__ == "__main__":
    main()
