from __future__ import annotations

import json
import re
import unicodedata
from typing import Any

import pandas as pd

from learner.nlp.llm import QwenJSONClient

DUPLICATE_PROMPT = r"""
Ты обнаруживаешь дубликаты произведений в каталоге Learner.
Сравнивай ИСКЛЮЧИТЕЛЬНО названия. Определи, являются ли два названия
разными языковыми версиями, транслитерациями или известными альтернативными
названиями ОДНОГО И ТОГО ЖЕ произведения.

Не используй авторов, годы, жанры, описания или тематику.
Похожие названия и общие слова не являются доказательством.
Не объединяй самостоятельные произведения. Если нет уверенности — не удаляй.
Сохраняй первый кандидат по порядку. candidate_id может ссылаться только
на более ранний кандидат или на запись из existing_candidates.
Не сравнивай запись с самой собой. Учитывай cross_type: если false,
сравнивай только одинаковые content_type.

Верни только JSON:
{"duplicates":[{"candidate_id":123,"duplicate_of":456,
"reason":"Перевод названия одного произведения"}]}
Если дубликатов нет, верни пустой массив.
"""


def normalize_title(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.casefold().replace("ё", "е")
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def _record(df: pd.DataFrame, position: int) -> dict[str, Any]:
    row = df.iloc[position]
    title = row.get("title", "")
    content_type = row.get("content_type", "")
    return {
        "candidate_id": int(position),
        "title": "" if pd.isna(title) else str(title).strip(),
        "content_type": "" if pd.isna(content_type) else str(content_type).strip()}


def check_candidate_duplicates(candidate_ids: list[int],existing_ids: list[int],df: pd.DataFrame,llm: QwenJSONClient,*,cross_type: bool = False,) -> dict[int, dict[str, Any]]:
    candidate_ids = list(dict.fromkeys(int(x) for x in candidate_ids))
    existing_ids = list(dict.fromkeys(int(x) for x in existing_ids))
    if not candidate_ids:
        return {}

    for position in candidate_ids + existing_ids:
        if position < 0 or position >= len(df):
            raise IndexError(f"Candidate position {position} is outside metadata")

    new_records = [_record(df, idx) for idx in candidate_ids]
    existing_records = [_record(df, idx) for idx in existing_ids]
    duplicates: dict[int, dict[str, Any]] = {}
    known = list(existing_records)

    remaining: list[dict[str, Any]] = []
    for record in new_records:
        title_key = normalize_title(record["title"])
        match = None
        if title_key:
            for previous in known:
                same_type = record["content_type"] == previous["content_type"]
                if not cross_type and not same_type:
                    continue
                if title_key == normalize_title(previous["title"]):
                    match = previous
                    break
        if match:
            duplicates[record["candidate_id"]] = {
                "duplicate_of": match["candidate_id"],
                "duplicate_title": match["title"],
                "reason": "Полное совпадение названия"}
        else:
            remaining.append(record)
            known.append(record)

    if not remaining or (not existing_records and len(remaining) < 2):
        return duplicates

    payload = {
        "cross_type": cross_type,
        "existing_candidates": existing_records,
        "new_candidates": remaining}
    positions = {item["candidate_id"]: i for i, item in enumerate(remaining)}
    existing_by_id = {item["candidate_id"]: item for item in existing_records}
    new_by_id = {item["candidate_id"]: item for item in remaining}

    try:
        response = llm.ask_json(
            DUPLICATE_PROMPT,
            json.dumps(payload, ensure_ascii=False))
        items = response.get("duplicates", [])
        if not isinstance(items, list):
            return duplicates

        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                candidate_id = int(item["candidate_id"])
                duplicate_of = int(item["duplicate_of"])
            except (KeyError, TypeError, ValueError):
                continue

            if candidate_id not in positions or candidate_id == duplicate_of:
                continue
            if duplicate_of not in existing_by_id and duplicate_of not in positions:
                continue
            if duplicate_of in positions and positions[duplicate_of] >= positions[candidate_id]:
                continue

            target = existing_by_id.get(duplicate_of) or new_by_id.get(duplicate_of)
            if target is None:
                continue
            if not cross_type and target["content_type"] != new_by_id[candidate_id]["content_type"]:
                continue

            duplicates[candidate_id] = {
                "duplicate_of": duplicate_of,
                "duplicate_title": target["title"],
                "reason": str(item.get("reason", "Перевод или альтернативное название"))}
    except Exception:
        pass

    return duplicates
