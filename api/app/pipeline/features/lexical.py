"""Extração dos 13 atributos lexicais sobre o nome do domínio (Grupo A — Índices 0 a 12)."""

from __future__ import annotations

import math
import re
from collections import Counter

_VOWELS = frozenset("aeiou")
_CONSONANTS = frozenset("bcdfghjklmnpqrstvwxyz")
_DIGITS = frozenset("0123456789")
_CONSONANT_SEQ_RE = re.compile(r"[bcdfghjklmnpqrstvwxyz]+")
_DIGIT_SEQ_RE = re.compile(r"\d+")


def shannon_entropy(text: str) -> float:
    """Calcula a entropia de Shannon (base 2) dos caracteres da string."""
    n = len(text)
    if n == 0:
        return 0.0
    counts = Counter(text)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def extract_lexical_features(domain: str) -> dict[str, float]:
    """
    Calcula os 13 atributos lexicais do domínio normalizado (`LEXICAL_FEATURE_ORDER`):
    0. `lex_length`
    1. `lex_digit_count`
    2. `lex_digit_ratio`
    3. `lex_vowel_ratio`
    4. `lex_consonant_ratio`
    5. `lex_special_char_ratio`
    6. `lex_entropy`
    7. `lex_hyphen_count`
    8. `lex_starts_with_digit`
    9. `lex_has_repeated_chars`
    10. `lex_max_consonant_seq`
    11. `lex_max_digit_seq`
    12. `lex_unique_char_ratio`
    """
    d = (domain or "").strip().lower()
    length = len(d)
    denom = max(length, 1)

    digit_count = sum(1 for ch in d if ch in _DIGITS)
    vowel_count = sum(1 for ch in d if ch in _VOWELS)
    consonant_count = sum(1 for ch in d if ch in _CONSONANTS)
    letter_denom = max(vowel_count + consonant_count, 1)
    special_count = sum(1 for ch in d if not ch.isalnum())
    hyphen_count = d.count("-")

    starts_with_digit = 1.0 if (length > 0 and d[0] in _DIGITS) else 0.0
    has_repeated_chars = 1.0 if any(d[i] == d[i + 1] for i in range(max(length - 1, 0))) else 0.0

    consonant_seqs = _CONSONANT_SEQ_RE.findall(d)
    max_consonant_seq = max((len(seq) for seq in consonant_seqs), default=0)

    digit_seqs = _DIGIT_SEQ_RE.findall(d)
    max_digit_seq = max((len(seq) for seq in digit_seqs), default=0)

    unique_ratio = len(set(d)) / denom if length > 0 else 0.0

    return {
        "lex_length": float(length),
        "lex_digit_count": float(digit_count),
        "lex_digit_ratio": round(digit_count / denom, 6),
        "lex_vowel_ratio": round(vowel_count / letter_denom, 6) if (vowel_count + consonant_count) > 0 else 0.0,
        "lex_consonant_ratio": round(consonant_count / letter_denom, 6) if (vowel_count + consonant_count) > 0 else 0.0,
        "lex_special_char_ratio": round(special_count / denom, 6),
        "lex_entropy": round(shannon_entropy(d), 6),
        "lex_hyphen_count": float(hyphen_count),
        "lex_starts_with_digit": starts_with_digit,
        "lex_has_repeated_chars": has_repeated_chars,
        "lex_max_consonant_seq": float(max_consonant_seq),
        "lex_max_digit_seq": float(max_digit_seq),
        "lex_unique_char_ratio": round(unique_ratio, 6),
    }
