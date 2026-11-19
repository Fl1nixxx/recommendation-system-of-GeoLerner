from __future__ import annotations

import re
import unicodedata
from typing import Any, Iterable


def normalize_author_name(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).casefold()
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.replace("ё", "е")
    text = re.sub(r"[^\w]+", " ", text)
    return " ".join(text.split())


def author_matches(value: Any, aliases: Iterable[str]) -> bool:
    authors = normalize_author_name(value)
    if not authors:
        return False
    padded_authors = f" {authors} "
    for alias in aliases:
        normalized_alias = normalize_author_name(alias)
        if normalized_alias and f" {normalized_alias} " in padded_authors:
            return True
    return False
