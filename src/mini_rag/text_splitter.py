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
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def _token_length(text: str) -> int:
        return len(tokenizer.tokenize(text))

    return _token_length


class SectionAwareTextSplitter:

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
        units = self._section_splitter.split_text(text)
        return [chunk for unit in units for chunk in self._split_unit(unit)]

    def _split_unit(self, unit: str) -> list[str]:
        if self._length_function(unit) <= self._chunk_size:
            return [unit]
        return self._body_splitter.split_text(unit)
