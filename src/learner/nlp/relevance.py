from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from learner.nlp.llm import QwenJSONClient

RELEVANCE_PROMPT = r"""
Ты проверяешь релевантность произведений запросу пользователя в Learner.
Для каждого кандидата выбери ровно один статус:
- relevant — произведение действительно соответствует запросу;
- irrelevant — произведение явно не соответствует запросу;
- uncertain — информации недостаточно для уверенного решения.

Оценивай исходный запрос целиком, а не отдельные совпадающие слова.
Учитывай название, подзаголовок, описание, автора, жанры, темы и атрибуты.
Не считай совпадение по стране или общей теме достаточным доказательством.
Отличай художественную литературу от научных, политических, учебных и
антропологических трудов, если запрос явно просит художественные произведения.
Не выдумывай сведения. Отсутствие описания само по себе не означает
нерелевантность. Не отклоняй незнакомое название только потому, что оно незнакомо.
Проверяй каждого кандидата независимо. Не пытайся сделать всех релевантными.

Верни только JSON:
{"evaluations":[{"candidate_id":0,"decision":"relevant","reason":"Краткая причина"}]}
Используй только relevant, irrelevant, uncertain. Не пропускай кандидатов.
Причина — не более 15 слов.
"""

CANDIDATE_FIELDS = [
    ("title", "Название", 200),
    ("subtitle", "Подзаголовок", 200),
    ("authors", "Авторы", 250),
    ("description", "Описание", 900),
    ("genres", "Жанры", 300),
    ("normalized_genres", "Нормализованные жанры", 300),
    ("topics", "Темы", 350),
    ("normalized_topics", "Нормализованные темы", 350),
    ("unmatched_topics", "Дополнительные темы", 350),
    ("attributes", "Атрибуты", 300),
    ("normalized_attributes", "Нормализованные атрибуты", 300),
    ("subjects", "Предметные категории", 400),
    ("work_type", "Тип произведения", 150),
    ("content_type", "Тип контента", 100),]


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        missing = pd.isna(value)
        return bool(missing) if isinstance(missing, (bool, np.bool_)) else False
    except (TypeError, ValueError):
        return False


def build_candidate_text(row: Mapping[str, Any] | pd.Series, candidate_id: int) -> str:
    """Build a compact metadata payload for Qwen's relevance check."""
    parts = [f"ID: {candidate_id}"]
    for column, label, limit in CANDIDATE_FIELDS:
        if column not in row:
            continue
        value = row[column]
        if _is_missing(value):
            continue
        if isinstance(value, (list, tuple, set, dict)):
            value = json.dumps(value, ensure_ascii=False, default=str)
        value = str(value).strip()
        if not value or value.casefold() in {"none", "null", "nan", "[]"}:
            continue
        parts.append(f"{label}: {value[:limit]}")
    return "\n".join(parts)[:3000]


def check_candidate_relevance(query: str,candidates: Sequence[dict[str, Any]],llm: QwenJSONClient,*,constraints: Mapping[str, Any] | None = None,batch_size: int = 6,) -> dict[int, dict[str, str]]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    decisions: dict[int, dict[str, str]] = {}
    for start in range(0, len(candidates), batch_size):
        batch = candidates[start : start + batch_size]
        payload = {
            "query": query,
            "constraints": dict(constraints or {}),
            "candidates": list(batch)}
        try:
            response = llm.ask_json(RELEVANCE_PROMPT, "Проверь следующие произведения:\n\n" + json.dumps(payload, ensure_ascii=False, default=str))
            evaluations = response.get("evaluations", [])
            valid_ids = {int(item["candidate_id"]) for item in batch}
            if isinstance(evaluations, list):
                for item in evaluations:
                    if not isinstance(item, dict):
                        continue
                    try:
                        candidate_id = int(item["candidate_id"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    if candidate_id not in valid_ids:
                        continue
                    decision = str(item.get("decision", "uncertain")).strip().lower()
                    if decision not in {"relevant", "irrelevant", "uncertain"}:
                        decision = "uncertain"
                    decisions[candidate_id] = {
                        "decision": decision,
                        "reason": str(item.get("reason", "")).strip()}
        except Exception:
            pass

        for candidate in batch:
            candidate_id = int(candidate["candidate_id"])
            decisions.setdefault(
                candidate_id,
                {
                    "decision": "uncertain",
                    "reason": "Недостаточно данных для проверки.",
                },
            )
    return decisions
