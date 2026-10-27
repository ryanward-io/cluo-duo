"""Clue words the AI partners must never use (the list is in blocked_words.py)."""

from __future__ import annotations

from .blocked_words import BLOCKED_WORDS

SUFFIXES = ("S", "ES", "ED", "ING", "ER", "ERS", "Y", "IES")
# Innocent words the suffix rule would otherwise catch (BUTT+ER, BRA+Y, ...).
ALLOWED = frozenset("BUTTER SPICE SPICY SPICES SPICED LUSTER HELLER BRAY COCKER CRAPS WEEDS".split())


def is_blocked(word: str) -> bool:
    """The word, or the word with a common ending removed, is on the list (BITCHES, FUCKING; not GRAPE)."""
    word = word.strip().upper()
    if word in ALLOWED:
        return False
    words = BLOCKED_WORDS
    return word in words or any(word.endswith(s) and word[: -len(s)] in words for s in SUFFIXES)
