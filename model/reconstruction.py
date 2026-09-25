"""Dictionary-based post-correction (ReconstructionModule, minimal version).

Builds a word-frequency dictionary from RUKOPYS train transcriptions
(excluding the fixed validation subset) and uses SymSpell to correct
individual out-of-vocabulary words in raw OCR output, leaving words that
are already close enough (or too far) to any known word unchanged.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Iterable

from symspellpy import SymSpell, Verbosity

MAX_EDIT_DISTANCE = 2
_WORD_RE = re.compile(r"\w+", re.UNICODE)


def build_word_frequencies(texts: Iterable[str]) -> Counter:
    """Count word occurrences (whitespace-split, per the task spec) across texts."""
    counts: Counter = Counter()
    for text in texts:
        counts.update(text.split())
    return counts


class ReconstructionModule:
    """Wraps a SymSpell instance built from a word-frequency dictionary."""

    def __init__(self, word_frequencies: Counter, max_edit_distance: int = MAX_EDIT_DISTANCE):
        self.max_edit_distance = max_edit_distance
        self.sym_spell = SymSpell(max_dictionary_edit_distance=max_edit_distance)
        for word, count in word_frequencies.items():
            self.sym_spell.create_dictionary_entry(word, count)

    def reconstruct(self, raw_text: str) -> str:
        """Correct each word in `raw_text` independently; words with no
        suggestion within `max_edit_distance` are left unchanged.
        """
        corrected_words = []
        for token in raw_text.split():
            match = _WORD_RE.search(token)
            if match is None:
                corrected_words.append(token)
                continue
            word = match.group(0)

            suggestions = self.sym_spell.lookup(
                word, Verbosity.CLOSEST, max_edit_distance=self.max_edit_distance,
            )
            if suggestions:
                corrected_words.append(token.replace(word, suggestions[0].term, 1))
            else:
                corrected_words.append(token)
        return " ".join(corrected_words)

    def save(self, path: str | Path) -> None:
        self.sym_spell.save_pickle(str(path))

    @classmethod
    def load(cls, path: str | Path, max_edit_distance: int = MAX_EDIT_DISTANCE) -> "ReconstructionModule":
        instance = cls.__new__(cls)
        instance.max_edit_distance = max_edit_distance
        instance.sym_spell = SymSpell(max_dictionary_edit_distance=max_edit_distance)
        instance.sym_spell.load_pickle(str(path))
        return instance
