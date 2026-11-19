from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd

from learner.nlp.llm import QwenJSONClient


TRANSLATION_PROMPT = """
Ты — профессиональный литературный переводчик.
Тебе передают названия книг, авторов и годы публикации.
Твоя задача — определить правильное русское название
каждой книги.

ПРАВИЛА:

1. Если существует известное русское название произведения,
   обязательно используй именно его.

2. Учитывай автора и год публикации, чтобы не перепутать
   разные произведения с одинаковыми названиями.

3. Если известного русского названия нет,
   создай естественный литературный перевод.

4. Не оставляй английское название без перевода,
   если возможно подобрать русский вариант.

5. Не переводи имена авторов.

6. Не добавляй пояснений, альтернатив и комментариев.

7. Не пропускай ни одной книги.

8. row_key должен точно соответствовать входному объекту.

Верни только JSON:
{
    "translations": [
        {
            "row_key": 0,
            "title_ru": "Русское название"
        }
    ]
}
"""

CYRILLIC_PATTERN = re.compile(r"[А-Яа-яЁё]")
LATIN_PATTERN = re.compile(r"[A-Za-z]")

def _clean_value(value: Any) -> str:

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return str(value).strip()

def _contains_cyrillic(value: str) -> bool:
    return bool(CYRILLIC_PATTERN.search(value))

def _contains_latin(value: str) -> bool:
    return bool(LATIN_PATTERN.search(value))


def _valid_translation(original: str,translated: str,) -> bool:

    if not translated:
        return False

    if translated.casefold() in {"null", "none", "nan"}:
        return False

    if _contains_latin(original):
        return _contains_cyrillic(translated)

    return True

def translate_book_titles_ru(result: pd.DataFrame,llm: QwenJSONClient,*,batch_size: int = 5,verbose: bool = True,) -> pd.DataFrame:

    if result.empty:
        return result.copy()

    if "title" not in result.columns:
        return result.copy()

    if "content_type" not in result.columns:
        return result.copy()

    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    output = result.copy().reset_index(drop=True)

    if "original_title" not in output.columns:
        output["original_title"] = (output["title"].apply(_clean_value))

    else:

        output["original_title"] = [_clean_value(original) or _clean_value(title) for original, title in zip(output["original_title"],output["title"],)]
    output["title_ru"] = output["original_title"]
    
    records = []

    skipped = 0

    for row_key, row in output.iterrows():

        content_type = _clean_value(row.get("content_type")).casefold()

        if content_type != "books":
            continue

        title = _clean_value(row["original_title"])

        if not title:
            continue

        if (_contains_cyrillic(title)and not _contains_latin(title)):
            skipped += 1
            continue

        if not _contains_latin(title):
            skipped += 1
            continue

        records.append({
            "row_key": int(row_key),
            "title": title,
            "author": _clean_value(
                row.get("authors")
            ),
            "year": _clean_value(
                row.get("year")
            ),})

    if not records:

        if verbose:
            print("Нет иностранных названий для перевода.")

        output["title"] = output["title_ru"]

        return output

    translated_keys = set()

    failed_keys = set()

    def process_batch(batch):

        payload = json.dumps({"books": batch},ensure_ascii=False,)

        response = llm.ask_json(TRANSLATION_PROMPT,payload,)
        translations = response.get("translations",[],)

        if not isinstance(translations, list):
            raise ValueError("Qwen не вернула массив translations")

        batch_keys = {item["row_key"]for item in batch}

        processed = set()

        for item in translations:

            if not isinstance(item, dict):
                continue

            try:
                row_key = int(item["row_key"])

                translated = _clean_value(item["title_ru"])

            except (KeyError, TypeError, ValueError):
                continue

            if row_key not in batch_keys:
                continue

            original = output.at[row_key,"original_title"]

            if not _valid_translation(original,translated,):
                continue

            output.at[row_key,"title_ru"] = translated
            processed.add(row_key)

        return processed

    for start in range(0,len(records),batch_size,):

        batch = records[start:start + batch_size]

        if verbose:
            print(f"\nПеревод книг: "f"{start + 1}-{start + len(batch)} "f"из {len(records)}" )

        try:

            processed = process_batch(batch)
            translated_keys.update(processed)

        except Exception as error:

            if verbose:
                print(f"Ошибка перевода батча: {error}")

    remaining = [item for item in records if item["row_key"] not in translated_keys]

    if verbose and remaining:
        print(f"\nПовторная обработка: "f"{len(remaining)} названий.")

    for item in remaining:
        try:

            processed = process_batch([item])

            if item["row_key"] in processed:
                translated_keys.update(processed)

            else:
                failed_keys.add(item["row_key"])
                if verbose:
                    print("Не удалось перевести:",item["title"],)

        except Exception as error:

            failed_keys.add(item["row_key"])
            if verbose:
                print(f"Ошибка для '{item['title']}': {error}")

    if verbose:

        print("\n" + "=" * 45)
        print("СТАТИСТИКА ПЕРЕВОДА")
        print("=" * 45)

        print("Книг для перевода:", len(records))
        print("Успешно переведено:", len(translated_keys))
        print("Не удалось перевести:", len(failed_keys))
        print("Уже на русском:", skipped)

        print("=" * 45)

    return output
