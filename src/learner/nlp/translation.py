from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd

from learner.nlp.llm import QwenJSONClient

TRANSLATION_PROMPT = """
Ты — профессиональный переводчик названий художественных
и развлекательных произведений на русский язык.

Тебе передают книги, фильмы и сериалы.
Твоя задача — вернуть естественное русское название
каждого произведения.

ПРАВИЛА ПЕРЕВОДА:

1. КНИГИ:
   - Используй общеизвестное русское литературное название,
     если оно существует.
   - Учитывай автора и год публикации.
   - Если официального перевода нет, предложи
     естественный литературный перевод.

2. ФИЛЬМЫ:
   - Используй известное русское прокатное название,
     если оно существует.
   - Учитывай страну производства и год выхода.
   - Не путай разные фильмы с одинаковыми названиями.

3. СЕРИАЛЫ:
   - Используй известное русское локализованное название,
     если оно существует.
   - Учитывай страну производства и год выхода.
   - Если официальное название неизвестно,
     предложи естественный перевод.

ОБЩИЕ ТРЕБОВАНИЯ:

- Переводи названия с любого языка.
- Не оставляй иностранное название без перевода.
- Не делай механическую транслитерацию,
  если существует естественный русский перевод.
- Не выдумывай официальность перевода.
- Сохраняй смысл и особенности оригинального названия.
- Не переводи имена авторов, режиссёров и персонажей.
- Не добавляй альтернативные названия и пояснения.
- Возвращай исключительно русское название.
- Не пропускай входные произведения.
- row_key должен точно соответствовать входному объекту.

Верни только JSON:
{
    "translations": [
        {
            "row_key": 0,
            "title_ru": "Русское название"
        }
    ]
}

Не добавляй Markdown и пояснения.
"""

CYRILLIC_PATTERN = re.compile(r"[А-Яа-яЁё]")
LATIN_PATTERN = re.compile(r"[A-Za-z]")

def clean_value(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, (list, tuple, set)):
        return ", ".join(str(item) for item in value)

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    value = str(value).strip()

    if value.casefold() in {"none", "null", "nan", "[]"}:
        return ""

    return value


def contains_cyrillic(value: str) -> bool:
    return bool(CYRILLIC_PATTERN.search(value))

def contains_latin(value: str) -> bool:
    return bool(LATIN_PATTERN.search(value))

def needs_translation(title: str) -> bool:
    if not title:
        return False

    if not any(char.isalpha() for char in title):
        return False

    if contains_cyrillic(title) and not contains_latin(title):
        return False

    return True


def valid_translation(original: str,translated: str,) -> bool:
    if not translated:
        return False

    if translated.casefold() in {"none", "null", "nan"}:
        return False

    if not contains_cyrillic(translated):
        return False

    if translated.casefold() == original.casefold():
        return False

    return True



CONTENT_TYPES = {
    "books": "Книга",
    "book": "Книга",
    "movies": "Фильм",
    "movie": "Фильм",
    "series": "Сериал",
    "tv": "Сериал",
    "tv series": "Сериал",}

def translate_book_titles_ru(result: pd.DataFrame,llm: QwenJSONClient,*,batch_size: int = 5,verbose: bool = False,) -> pd.DataFrame:

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
        output["original_title"] = (output["title"].apply(clean_value))

    else:
        output["original_title"] = [clean_value(original) or clean_value(title)
            for original, title in zip(output["original_title"],output["title"],)]

    output.drop(columns=["title_ru"],errors="ignore",inplace=True,)

    records = []
    skipped = 0

    for row_key, row in output.iterrows():

        content_type = clean_value(row.get("content_type")).casefold()

        if content_type not in CONTENT_TYPES:
            continue

        title = clean_value(row["original_title"])

        if not title:
            continue

        if not needs_translation(title):
            skipped += 1
            continue

        year = (clean_value(row.get("year"))or clean_value(row.get("release_year"))or clean_value(row.get("first_publish_year")))
        country = (clean_value(row.get("country")) or clean_value(row.get("countries")) or clean_value(row.get("production_countries")))
        creator = (clean_value(row.get("authors")) or clean_value(row.get("directors")) or clean_value(row.get("director")))
       
        records.append({
            "row_key": int(row_key),
            "content_type": CONTENT_TYPES[content_type],
            "title": title,
            "author_or_director": creator,
            "year": year,
            "country": country,})

    if not records:
        if verbose:
            print("Нет иностранных названий для перевода.")

        return output
       
    translated_titles: dict[int, str] = {}

    def process_batch(batch):

        payload = json.dumps({"items": batch},ensure_ascii=False,)

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
                translated = clean_value(item["title_ru"])

            except (KeyError, TypeError, ValueError):
                continue

            if row_key not in batch_keys:
                continue

            original = clean_value(output.at[row_key, "original_title"])

            if not valid_translation(original,translated,):
                continue

            translated_titles[row_key] = translated
            processed.add(row_key)

        return processed

    for start in range(0,len(records),batch_size,):

        batch = records[start:start + batch_size]
        try:
            process_batch(batch)

        except Exception as error:
            if verbose:
                print(f"Ошибка перевода батча: {error}")

    remaining = [item for item in records if item["row_key"] not in translated_titles]
    failed = 0

    for item in remaining:
        try:
            processed = process_batch([item])

            if item["row_key"] not in processed:
                failed += 1

                if verbose:
                    print("Не удалось перевести:",item["title"],)

        except Exception as error:

            failed += 1

            if verbose:
                print(f"Ошибка для '{item['title']}': {error}")

    for row_key, translated in translated_titles.items():

        output.at[row_key, "title"] = translated

    if verbose:

        print("\nСтатистика перевода:")
        print("Передано на перевод:", len(records))
        print("Успешно переведено:", len(translated_titles))
        print("Не удалось перевести:", failed)
        print("Уже на русском:", skipped)

    return output
