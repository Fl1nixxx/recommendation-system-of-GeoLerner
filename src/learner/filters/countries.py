from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from typing import Any

import pandas as pd
import pycountry
from babel import Locale

COUNTRY_COLUMNS = ("country", "countries", "origin_country", "production_countries")
MANUAL_ALIASES = {
    "сша": "US", "америка": "US", "россия": "RU", "рф": "RU",
    "великобритания": "GB", "британия": "GB", "англия": "GB",
    "индия": "IN", "япония": "JP", "южная корея": "KR",
    "корея": "KR", "китай": "CN", "тайвань": "TW", "иран": "IR",
    "сирия": "SY", "турция": "TR", "голландия": "NL",
    "нидерланды": "NL", "чехия": "CZ", "беларусь": "BY",
    "украина": "UA", "ссср": "SU", "usa": "US", "uk": "GB",
    "ussr": "SU",}


def normalize_name(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    text = re.sub(r"[^\w]+", " ", text)
    return " ".join(text.split())


@lru_cache(maxsize=1)
def _country_lookup() -> dict[str, str]:
    lookup: dict[str, str] = {}
    for country in pycountry.countries:
        for field in ("name", "official_name", "common_name"):
            name = getattr(country, field, None)
            if name:
                lookup[normalize_name(name)] = country.alpha_2

    for locale_code in ("ru", "en", "de", "fr", "es", "it", "pt", "zh", "ja", "ko"):
        locale = Locale.parse(locale_code)
        for code, name in locale.territories.items():
            if re.fullmatch(r"[A-Z]{2}", code):
                lookup.setdefault(normalize_name(name), code)

    lookup.update({normalize_name(name): code for name, code in MANUAL_ALIASES.items()})
    return lookup


def country_values_from_cell(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        result: list[str] = []
        for item in value:
            result.extend(country_values_from_cell(item))
        return result
    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass

    text = str(value).strip()
    if not text or text.casefold() in {"none", "null", "nan", "[]"}:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            parsed = [parsed]
        if isinstance(parsed, list):
            result = []
            for item in parsed:
                if isinstance(item, dict):
                    for key in ("iso_3166_1", "iso2", "name"):
                        if item.get(key):
                            result.append(str(item[key]))
                elif item is not None:
                    result.append(str(item))
            return result
    except (json.JSONDecodeError, TypeError):
        pass
    return [part.strip() for part in re.split(r"[,;|]", text) if part.strip()]


def resolve_country_codes(country: str | list[str] | tuple[str, ...] | set[str] | None) -> set[str]:
    if country is None:
        return set()
    if isinstance(country, str):
        values = [part.strip() for part in country.split(",") if part.strip()]
    elif isinstance(country, (list, tuple, set)):
        values = list(country)
    else:
        raise TypeError("country must be a string, a collection of strings, or None")

    lookup = _country_lookup()
    codes: set[str] = set()
    for value in values:
        value = str(value).strip()
        if not value:
            continue
        if re.fullmatch(r"[A-Za-z]{2}", value):
            code = value.upper()
            if code == "SU" or pycountry.countries.get(alpha_2=code):
                codes.add(code)
                continue
            raise ValueError(f"Unknown ISO-3166 alpha-2 code: {code}")
        code = lookup.get(normalize_name(value))
        if code is None:
            raise ValueError(f"Could not resolve country: {value}")
        codes.add(code)
    return codes


def build_catalog_country_codes(metadata: pd.DataFrame) -> list[set[str]]:
    columns = [column for column in COUNTRY_COLUMNS if column in metadata.columns]
    lookup = _country_lookup()
    result: list[set[str]] = []
    for _, row in metadata.iterrows():
        codes: set[str] = set()
        for column in columns:
            for value in country_values_from_cell(row[column]):
                value = str(value).strip()
                if re.fullmatch(r"[A-Za-z]{2}", value):
                    code = value.upper()
                    if code == "SU" or pycountry.countries.get(alpha_2=code):
                        codes.add(code)
                else:
                    code = lookup.get(normalize_name(value))
                    if code:
                        codes.add(code)
        result.append(codes)
    return result


def country_display_values(row: pd.Series) -> str:
    values: list[str] = []
    for column in COUNTRY_COLUMNS:
        if column in row.index:
            values.extend(country_values_from_cell(row[column]))
    return ", ".join(dict.fromkeys(values))
