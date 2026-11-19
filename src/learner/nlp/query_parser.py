from __future__ import annotations

from typing import Any

from learner.nlp.llm import QwenJSONClient

QUERY_SYSTEM_PROMPT = r"""
Ты NLP-компонент рекомендательной системы Learner.

Проанализируй запрос пользователя и верни параметры для семантического поиска.
Верни исключительно один корректный JSON-объект со следующими полями:

{
  "content_type": null,
  "year_from": null,
  "year_to": null,
  "reference_author": null,
  "reference_author_aliases": [],
  "reference_mode": null,
  "exclude_reference_author": false,
  "semantic_query": ""
}

ТИП КОНТЕНТА:
Допустимы books, movies, series или null. Если пользователь хочет несколько
типов, используй null.

ГОДЫ:
Распознавай конкретные годы, десятилетия, века и диапазоны. Границы включаются.
Неуказанные границы возвращай как null.

АВТОРЫ:
Определяй автора, если он является объектом поиска или ориентиром.
reference_author — имя автора. reference_author_aliases — только распространённые
и достоверные варианты написания на русском и английском.
reference_mode принимает author, style или null.
Для author exclude_reference_author=false.
Для style exclude_reference_author=true.
Если автор не указан, все авторские поля должны быть null/пустыми, а флаг false.

Если пользователь просит произведения конкретного автора, используй author.
Если просит произведения в стиле или похожие на творчество автора, используй style.
Для style не оставляй имя автора в semantic_query: опиши особенности творчества,
но не выдумывай неподтверждённые черты.
Для author не дублируй имя автора в semantic_query — фильтрация по автору
выполняется отдельно.

SEMANTIC_QUERY:
Это краткое содержательное описание для E5. Сохраняй смысл и предпочтения
пользователя. Не добавляй жанры, темы, атмосферу, сюжетные детали или другие
характеристики, которых пользователь не запрашивал и которые не следуют
однозначно из контекста. Убирай только тип контента и временные ограничения,
которые вынесены в отдельные поля. Не добавляй страну и национальность.
Если смысловых предпочтений нет, верни пустую строку.

СТРАНУ, национальность произведения и страну производства НЕ определяй.
Страна передаётся пользователем отдельно.
Все перечисленные поля обязательны. Годы — целые числа или null.
Не добавляй дополнительные поля, Markdown или пояснения.
"""

DEFAULTS: dict[str, Any] = {
    "content_type": None,
    "year_from": None,
    "year_to": None,
    "reference_author": None,
    "reference_author_aliases": [],
    "reference_mode": None,
    "exclude_reference_author": False,
    "semantic_query": "",}


def _as_optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_user_query(query: str, llm: QwenJSONClient) -> dict[str, Any]:
    """Parse a request with Qwen and normalize its returned schema."""
    if not str(query).strip():
        raise ValueError("Search query cannot be empty")

    raw = llm.ask_json(QUERY_SYSTEM_PROMPT, str(query).strip())
    result = {key: raw.get(key, default) for key, default in DEFAULTS.items()}

    if result["content_type"] not in {"books", "movies", "series", None}:
        result["content_type"] = None

    result["year_from"] = _as_optional_int(result["year_from"])
    result["year_to"] = _as_optional_int(result["year_to"])

    aliases = result["reference_author_aliases"]
    if isinstance(aliases, str):
        aliases = [aliases]
    if not isinstance(aliases, list):
        aliases = []
    result["reference_author_aliases"] = list(dict.fromkeys(str(item).strip() for item in aliases if str(item).strip()))

    author = result["reference_author"]
    result["reference_author"] = str(author).strip() if author else None

    if result["reference_mode"] not in {"author", "style", None}:
        result["reference_mode"] = None

    if result["reference_author"] is None:
        result["reference_author_aliases"] = []
        result["reference_mode"] = None
        result["exclude_reference_author"] = False
    else:
        if result["reference_mode"] is None:
            result["reference_mode"] = "author"
        result["reference_author_aliases"] = list(
            dict.fromkeys(
                result["reference_author_aliases"] + [result["reference_author"]]
            )
        )
        result["exclude_reference_author"] = result["reference_mode"] == "style"

    result["semantic_query"] = str(result["semantic_query"] or "").strip()
    return result
