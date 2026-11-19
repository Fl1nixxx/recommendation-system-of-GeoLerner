from __future__ import annotations

import numpy as np
import pandas as pd

from learner.filters.countries import country_display_values

BASE_COLUMNS = [
    "content_type", "id", "title", "original_title", "year",
    "country", "semantic_score", "authors", "genres",
    "qwen_verdict", "qwen_reason",]


def build_results(metadata: pd.DataFrame,selected: list[dict],years: pd.Series,scores: np.ndarray,) -> pd.DataFrame:
    if not selected:
        return pd.DataFrame()

    positions = [int(item["index"]) for item in selected]
    result = metadata.iloc[positions].copy().reset_index(drop=True)
    result["year"] = years.iloc[positions].to_numpy()
    result["semantic_score"] = scores[positions]
    result["qwen_verdict"] = [item["decision"] for item in selected]
    result["qwen_reason"] = [item["reason"] for item in selected]
    result["country"] = [country_display_values(metadata.iloc[position]) for position in positions]
    result["original_title"] = result["title"] if "title" in result else ""
    columns = [column for column in BASE_COLUMNS if column in result.columns]
    
    return result[columns].reset_index(drop=True)
