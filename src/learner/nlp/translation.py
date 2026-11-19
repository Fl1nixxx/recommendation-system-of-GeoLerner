from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd

from learner.nlp.llm import QwenJSONClient

TRANSLATION_PROMPT = r"""
Ты — литературный переводчик. Для каждого названия книги верни естественное
русское название. Если существует общеизвестное русское название произведения,
используй его. Не выдумывай официальность перевода. Если официальное название
неизвестно, предложи аккуратный литературный перевод. Учитывай автора и год,
не переводи имя автора. Сохраняй смысл оригинала.

Верни только JSON:
{"translations":[{"row_key":0,"title_ru":"Русское название"}]}
Не добавляй комментарии и Markdown.
"""


def _contains_cyrillic(value: str) -> bool:
    return bool(re.search(r"[А-Яа-яЁё]", value))


def translate_book_titles_ru(result: pd.DataFrame,llm: QwenJSONClient,*,batch_size: int = 15,) -> pd.DataFrame:
    if result.empty:
        return result.copy()

    if "title" not in result.columns or "content_type" not in result.columns:
        return result.copy()

    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    output = result.copy().reset_index(drop=True)
    output["original_title"] = output["title"].fillna("").astype(str)
    output["title_ru"] = output["original_title"]

    records: list[dict[str, Any]] = []
    for row_key, row in output.iterrows():
        content_type = str(row.get("content_type", "")).casefold()
        title = str(row.get("title", "") or "").strip()
        if content_type != "books" or not title or title.casefold() == "nan":
            continue
        if _contains_cyrillic(title):
            continue
        records.append({
            "row_key": int(row_key),
            "title": title,
            "author": str(row.get("authors", "") or ""),
            "year": str(row.get("year", "") or "")})

    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        try:
            response = llm.ask_json(TRANSLATION_PROMPT,json.dumps(batch, ensure_ascii=False, default=str))
            translations = response.get("translations", [])
            if not isinstance(translations, list):
                continue
            for item in translations:
                if not isinstance(item, dict):
                    continue
                try:
                    row_key = int(item["row_key"])
                    translated = str(item["title_ru"]).strip()
                except (KeyError, TypeError, ValueError):
                    continue
                if 0 <= row_key < len(output) and translated and translated.casefold() != "nan":
                    output.at[row_key, "title_ru"] = translated
        except Exception:
            continue

    output["title"] = output["title_ru"]
    return output
