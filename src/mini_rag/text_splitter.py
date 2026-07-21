"""Text splitting for the mini-RAG ingestion pipeline.

Splits a parsed PDF's plain-text output into retrieval-sized chunks in two
passes:

  1. A ``RecursiveCharacterTextSplitter`` configured with only the numbered
     section-heading separators (every depth: "1. ", "1.1. ", "1.1.1. ",
     "1.1.1.1. ") and no fallback separators, forced to never merge
     adjacent sections and never recurse further by using a chunk_size of
     1: any piece it produces is, by definition, "too big" for that size,
     so each becomes its own unit with no merging and (since there are no
     further separators to fall back to) no further splitting either. This
     gives a pure, size-unaware hard split on section boundaries alone.
  2. Only the resulting section units that are still larger than the real
     ``chunk_size`` are handed to a ``CharacterTextSplitter`` -- a single
     flat pass with one separator and no recursion tiers -- so
     ``chunk_overlap`` applies consistently across every sub-chunk of that
     unit, unlike a single multi-tier recursive splitter where each
     recursion step starts a fresh buffer and drops overlap across that
     boundary.

``chunk_size`` is measured in tokens of the embedding model's own tokenizer
(see :mod:`mini_rag.embedder`) rather than characters, since the E5 models
used for embedding truncate their input at a fixed token budget (512
tokens): sizing chunks in characters could silently let long or
token-dense text pass that limit and get truncated before embedding.
"""

from collections.abc import Callable
from typing import Final

from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from transformers import AutoTokenizer

from mini_rag.embedder import DEFAULT_MODEL_NAME

_CHUNK_OVERLAP_RATIO: Final[float] = 0.2
_BODY_SEPARATOR: Final[str] = " "

_SECTION_HEADING_PATTERNS: Final[list[str]] = [
    r"\n(?=\d+\.\s[A-ZÁÉÍÓÖŐÚÜŰ0-9])",
    r"\n(?=\d+\.\d+\.?\s[A-ZÁÉÍÓÖŐÚÜŰ0-9])",
    r"\n(?=\d+\.\d+\.\d+\.?\s[A-ZÁÉÍÓÖŐÚÜŰ0-9])",
    r"\n(?=\d+\.\d+\.\d+\.\d+\.?\s[A-ZÁÉÍÓÖŐÚÜŰ0-9])",
]


def _token_length_function(model_name: str) -> Callable[[str], int]:
    """Build a token-counting length function backed by ``model_name``'s
    own tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _token_length(text: str) -> int:
        return len(tokenizer.tokenize(text))

    return _token_length


class SectionAwareTextSplitter:
    """Splits document text into chunks that respect numbered section
    headings, using a section-boundary pass followed by an isolated,
    overlap-preserving pass for any section that doesn't fit chunk_size."""

    def __init__(
        self,
        chunk_size: int,
        length_function: Callable[[str], int] | None = None,
    ) -> None:
        self._chunk_size: int = chunk_size
        self._length_function: Callable[[str], int] = (
            length_function or _token_length_function(DEFAULT_MODEL_NAME)
        )
        self._section_splitter: RecursiveCharacterTextSplitter = (
            RecursiveCharacterTextSplitter(
                separators=_SECTION_HEADING_PATTERNS,
                is_separator_regex=True,
                keep_separator="start",
                chunk_size=1,
                chunk_overlap=0,
            )
        )
        self._body_splitter: CharacterTextSplitter = CharacterTextSplitter(
            separator=_BODY_SEPARATOR,
            is_separator_regex=False,
            chunk_size=chunk_size,
            chunk_overlap=int(chunk_size * _CHUNK_OVERLAP_RATIO),
            length_function=self._length_function,
        )

    def split(self, text: str) -> list[str]:
        """Split the given text into a list of chunks."""
        units = self._section_splitter.split_text(text)
        return [chunk for unit in units for chunk in self._split_unit(unit)]

    def _split_unit(self, unit: str) -> list[str]:
        """Return a single section unit unchanged if it already fits
        chunk_size; otherwise split it with the body splitter so its
        sub-chunks get consistent chunk_overlap between them."""
        if self._length_function(unit) <= self._chunk_size:
            return [unit]
        return self._body_splitter.split_text(unit)
