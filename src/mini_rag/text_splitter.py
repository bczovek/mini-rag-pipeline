"""Text splitting for the mini-RAG ingestion pipeline.

Splits a parsed PDF's plain-text output into retrieval-sized chunks in two
passes:

  1. A ``RecursiveCharacterTextSplitter`` configured with only the numbered
     section-heading separators (every depth: "1. ", "1.1. ", "1.1.1. ",
     "1.1.1.1. ") and no fallback separators, using a small ``chunk_size``
     (``_MIN_SECTION_TOKENS``) so short sections (a bare heading, a
     one-sentence section) are merged with their following same-level
     sibling(s) up to that size instead of becoming their own tiny, mostly
     uninformative chunk, while sections already at or above that size are
     left as their own unit.
  2. Only the resulting section units that are still larger than the real
     ``chunk_size`` are handed to a ``CharacterTextSplitter`` -- a single
     flat pass with one separator and no recursion tiers -- so
     ``chunk_overlap`` applies consistently across every sub-chunk of that
     unit, unlike a single multi-tier recursive splitter where each
     recursion step starts a fresh buffer and drops overlap across that
     boundary.

``chunk_size`` (and ``_MIN_SECTION_TOKENS``) are measured in tokens of the
embedding model's own tokenizer (see :mod:`mini_rag.embedder`) rather than
characters, since the E5 models used for embedding truncate their input at a
fixed token budget (512 tokens): sizing chunks in characters could silently
let long or token-dense text pass that limit and get truncated before
embedding.
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

# Sections at or below this many tokens (e.g. a bare heading like "1.9.
# Szerkesztési célra fenntartva.") are merged with following same-level
# sibling sections instead of staying their own near-empty chunk. Chosen
# from the corpus's own token-length distribution: comfortably above the
# ~50-token "title or one short sentence" band, well below the ~170-token
# median section, so only genuinely tiny sections get merged.
_MIN_SECTION_TOKENS: Final[int] = 96

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
                chunk_size=_MIN_SECTION_TOKENS,
                chunk_overlap=0,
                length_function=self._length_function,
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
