"""Character-level tokenizer/vocabulary for CTC-based text recognition."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

BLANK_TOKEN = "<blank>"


class CharTokenizer:
    """Builds a character vocabulary from transcriptions and encodes/decodes text.

    Index 0 is always reserved for the CTC blank token.
    """

    def __init__(self, chars: Iterable[str]):
        unique_chars = sorted(set(chars))
        self.itos: List[str] = [BLANK_TOKEN] + unique_chars
        self.stoi = {ch: idx for idx, ch in enumerate(self.itos)}

    @classmethod
    def from_texts(cls, texts: Iterable[str]) -> "CharTokenizer":
        """Build a vocabulary from an iterable of transcription strings."""
        chars = set()
        for text in texts:
            chars.update(text)
        return cls(chars)

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    @property
    def blank_id(self) -> int:
        return 0

    def encode(self, text: str) -> List[int]:
        """Encode a string into a list of character ids (no blanks inserted)."""
        try:
            return [self.stoi[ch] for ch in text]
        except KeyError as exc:
            raise ValueError(f"character {exc.args[0]!r} not in vocabulary") from exc

    def decode(self, ids: Iterable[int]) -> str:
        """Decode a list of character ids back into a string (ids must exclude blanks)."""
        return "".join(self.itos[i] for i in ids)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.itos, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        itos = json.loads(Path(path).read_text(encoding="utf-8"))
        tokenizer = cls.__new__(cls)
        tokenizer.itos = itos
        tokenizer.stoi = {ch: idx for idx, ch in enumerate(itos)}
        return tokenizer
