from __future__ import annotations

from learner.nlp.llm import QwenJSONClient

MODERATION_PROMPT = r"""
Ты — модератор запросов рекомендательной системы Learner.
Определи, является ли текст осмысленным запросом для поиска книг, фильмов
или сериалов.

Отклоняй случайный набор символов, бессвязный текст, бессмыслицу и сообщения,
которые не имеют отношения к поиску произведений.

Не отклоняй запрос только потому, что он короткий, разговорный, содержит
опечатки, грамматические ошибки или мат. Если в нём есть понятное поисковое
намерение, запрос допустим.
Допускай названия произведений, имена авторов, жанры, темы, страны и годы.
Если сомневаешься, считай запрос допустимым.

Верни только JSON:
{"valid": true, "reason": ""}
или
{"valid": false, "reason": "Краткая причина на русском"}
"""


def validate_search_query(query: str,llm: QwenJSONClient,*,max_length: int = 500,fail_open: bool = True) -> tuple[bool, str]:
    query = str(query or "").strip()
    if not query:
        return False, "Запрос пустой."
        
    if len(query) > max_length:
        return False, f"Запрос слишком длинный. Максимум {max_length} символов."

    if not any(char.isalnum() for char in query):
        return False, "Запрос содержит только символы. Напиши, что хочешь найти."

    letters = [char.casefold() for char in query if char.isalpha()]
    if len(letters) >= 8 and len(set(letters)) <= 1:
        return False, "Похоже, ты набрал случайные повторяющиеся символы."

    try:
        result = llm.ask_json(MODERATION_PROMPT, query)
        valid = result.get("valid")
        if not isinstance(valid, bool):
            raise ValueError("Qwen returned an invalid moderation schema")
        return valid, str(result.get("reason", "")).strip()
    except Exception:
        if fail_open:
            return True, ""
        raise
