from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from learner.filters.authors import author_matches
from learner.filters.countries import build_catalog_country_codes, resolve_country_codes
from learner.filters.years import extract_years
from learner.nlp.deduplication import check_candidate_duplicates
from learner.nlp.llm import QwenJSONClient
from learner.nlp.moderation import validate_search_query
from learner.nlp.query_parser import parse_user_query
from learner.nlp.relevance import build_candidate_text, check_candidate_relevance
from learner.nlp.translation import translate_book_titles_ru
from learner.recommendation.embeddings import cosine_scores, encode_query
from learner.recommendation.ranking import rank_indices
from learner.recommendation.results import build_results


class SearchEngine:
    def __init__(self,*,metadata: pd.DataFrame,embeddings: np.ndarray,e5_model: Any, llm: QwenJSONClient,catalog_country_codes: list[set[str]] | None = None) -> None:

        self.metadata = metadata.reset_index(drop=True).copy()
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.e5_model = e5_model
        self.llm = llm

        if self.metadata.empty:
            raise ValueError("metadata cannot be empty")

        if self.embeddings.ndim != 2:
            raise ValueError("embeddings must be a 2D matrix")

        if len(self.metadata) != len(self.embeddings):
            raise ValueError("metadata and embeddings must have the same row count")

        if catalog_country_codes is None:
            self.catalog_country_codes = build_catalog_country_codes(self.metadata)
        else:
            if len(catalog_country_codes) != len(self.metadata):
                raise ValueError("catalog_country_codes must align with metadata")
            self.catalog_country_codes = [set(codes) for codes in catalog_country_codes]

    def search(self,query: str,*,country: str | list[str] | tuple[str, ...] | set[str] | None = None,top_k: int = 5,candidate_pool: int = 120,batch_size: int = 6,deduplicate_across_types: bool = False,translate_titles: bool = True,verbose: bool = True,) -> pd.DataFrame:
        query = str(query or "").strip()
        if not query:
            return pd.DataFrame()

        if top_k < 1:
            raise ValueError("top_k must be positive")

        if candidate_pool < top_k:
            raise ValueError("candidate_pool must be greater than or equal to top_k")

        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        is_valid, rejection_reason = validate_search_query(query, self.llm)
        if not is_valid:
            if verbose:
                print(f"Запрос отклонён: {rejection_reason or 'бессмысленный запрос'}")
            return pd.DataFrame()

        parsed = parse_user_query(query, self.llm)
        if verbose:
            import json
            print("АНАЛИЗ ЗАПРОСА:")
            print(json.dumps(parsed, ensure_ascii=False, indent=2))

        country_codes = resolve_country_codes(country)
        content_type = parsed["content_type"]
        reference_author = parsed["reference_author"]
        author_aliases = parsed["reference_author_aliases"]
        reference_mode = parsed["reference_mode"]
        semantic_query = str(parsed["semantic_query"] or "").strip()

        if reference_mode != "author" and not semantic_query:
            semantic_query = query

        mask = np.ones(len(self.metadata), dtype=bool)

        if content_type:
            if "content_type" not in self.metadata.columns:
                raise ValueError("metadata has no content_type column")
            mask &= self.metadata["content_type"].eq(content_type).to_numpy()

        years = extract_years(self.metadata)
        if parsed["year_from"] is not None:
            mask &= years.ge(parsed["year_from"]).to_numpy()
        if parsed["year_to"] is not None:
            mask &= years.le(parsed["year_to"]).to_numpy()

        if country_codes:
            country_mask = np.fromiter((bool(codes & country_codes) for codes in self.catalog_country_codes),dtype=bool,count=len(self.catalog_country_codes))
            mask &= country_mask

        if reference_author:
            if "authors" not in self.metadata.columns:
                raise ValueError("metadata has no authors column")

            if "content_type" in self.metadata.columns:
                mask &= self.metadata["content_type"].eq("books").to_numpy()

            author_mask = self.metadata["authors"].apply(lambda value: author_matches(value, author_aliases or [reference_author])).to_numpy()
            if not author_mask.any() and reference_mode == "author":
                if verbose:
                    print("Произведения указанного автора не найдены в каталоге.")
                return pd.DataFrame()

            if reference_mode == "author":
                mask &= author_mask
            elif reference_mode == "style" and parsed["exclude_reference_author"]:
                mask &= ~author_mask

        candidate_indices = np.flatnonzero(mask)
        if not len(candidate_indices):
            if verbose:
                print("Подходящих произведений не найдено.")
            return pd.DataFrame()

        if reference_mode == "author" and not semantic_query:
            scores = np.zeros(len(self.metadata), dtype=np.float32)
        else:
            query_vector = encode_query(semantic_query, self.e5_model)
            scores = cosine_scores(self.embeddings, query_vector)

        ordered = rank_indices(candidate_indices, scores, self.metadata, reference_mode=reference_mode)
        pool = ordered[:candidate_pool]

        constraints = {
            "content_type": content_type,
            "country_codes": sorted(country_codes),
            "year_from": parsed["year_from"],
            "year_to": parsed["year_to"],
            "reference_author": reference_author,
            "reference_mode": reference_mode}

        accepted: list[dict[str, Any]] = []
        uncertain: list[dict[str, Any]] = []
        rejected_count = 0
        duplicate_count = 0
        checked_count = 0

        for start in range(0, len(pool), batch_size):
            batch_indices = [int(value) for value in pool[start : start + batch_size]]
            if reference_mode == "author":
                decisions = {
                    idx: {"decision": "relevant", "reason": "Совпадение автора."}
                    for idx in batch_indices}
            else:
                candidates = [
                    {
                        "candidate_id": idx,
                        "text": build_candidate_text(self.metadata.iloc[idx], idx),
                    }
                    for idx in batch_indices]
                decisions = check_candidate_relevance(query,candidates,self.llm,constraints=constraints,batch_size=batch_size,)

            checked_count += len(batch_indices)
            relevant_items: list[dict[str, Any]] = []
            uncertain_items: list[dict[str, Any]] = []
            for idx in batch_indices:
                verdict = decisions.get(
                    idx,
                    {"decision": "uncertain", "reason": "Нет оценки Qwen."})
                decision = str(verdict.get("decision", "uncertain")).lower().strip()
                item = {
                    "index": idx,
                    "decision": decision,
                    "reason": str(verdict.get("reason", ""))}

                if decision == "relevant":
                    relevant_items.append(item)

                elif decision == "irrelevant":
                    rejected_count += 1

                else:
                    uncertain_items.append(item)

            relevant_ids = [item["index"] for item in relevant_items]
            if relevant_ids:
                duplicate_map = check_candidate_duplicates(relevant_ids,[item["index"] for item in accepted],self.metadata,self.llm, cross_type=deduplicate_across_types)
            else:
                duplicate_map = {}
            duplicate_count += len(duplicate_map)

            for item in relevant_items:
                idx = item["index"]
                if idx in duplicate_map:
                    duplicate = duplicate_map[idx]
                    if verbose:
                        print(
                            f"Дубликат исключён: {self.metadata.iloc[idx].get('title', '')} "
                            f"-> {duplicate.get('duplicate_title', '')} "
                            f"({duplicate.get('reason', '')})")
                    continue
                accepted.append(item)

            uncertain.extend(uncertain_items)
            if verbose:
                print(
                    f"Проверено: {checked_count} | Подходит: {len(accepted)} "
                    f"| Отклонено: {rejected_count} | Дубликатов: {duplicate_count} "
                    f"| Неоднозначно: {len(uncertain)}")
            if len(accepted) >= top_k:
                break

        selected = accepted[:top_k]
        if len(selected) < top_k and uncertain:
            remaining_uncertain = [item for item in uncertain if item["index"] not in {x["index"] for x in selected}]
            uncertain_ids = [item["index"] for item in remaining_uncertain]
            uncertain_duplicates = check_candidate_duplicates(uncertain_ids,[item["index"] for item in selected],self.metadata,self.llm,cross_type=deduplicate_across_types,) if uncertain_ids else {}
            duplicate_count += len(uncertain_duplicates)
            selected.extend(item for item in remaining_uncertain if item["index"] not in uncertain_duplicates)
            selected = selected[:top_k]

        if not selected:
            if verbose:
                print("Релевантных уникальных произведений не найдено.")
            return pd.DataFrame()

        result = build_results(self.metadata, selected, years, scores)
        if translate_titles:
            result = translate_book_titles_ru(result,self.llm,batch_size=5,verbose=verbose,)

        if verbose:
            print("\n" + "=" * 48)
            print("ИТОГОВАЯ СТАТИСТИКА LEARNER")
            print("=" * 48)
            print(f"Проверено: {checked_count}")
            print(f"Релевантных: {len(accepted)}")
            print(f"Отклонено: {rejected_count}")
            print(f"Дубликатов исключено: {duplicate_count}")
            print(f"Неоднозначных: {len(uncertain)}")
            print(f"Итоговый размер выдачи: {len(result)}")

        return result


def semantic_search(query: str,*,metadata: pd.DataFrame,embeddings: np.ndarray,e5_model: Any,llm: QwenJSONClient,**search_options: Any,) -> pd.DataFrame:
    engine = SearchEngine(metadata=metadata,embeddings=embeddings,e5_model=e5_model, llm=llm)
    return engine.search(query, **search_options)
