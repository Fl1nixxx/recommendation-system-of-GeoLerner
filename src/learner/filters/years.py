from __future__ import annotations

import pandas as pd

YEAR_COLUMNS = ("first_publish_year", "release_year", "first_air_year", "year")


def extract_years(metadata: pd.DataFrame) -> pd.Series:
    years = pd.Series(float("nan"), index=metadata.index, dtype="float64")
    for column in YEAR_COLUMNS:
        if column in metadata.columns:
            values = pd.to_numeric(metadata[column], errors="coerce")
            years = years.fillna(values)
    return years
