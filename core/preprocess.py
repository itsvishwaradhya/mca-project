"""Scaling / imputation shared by SSL and baselines. Labels never pass through here."""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

CLIP = 5.0


def numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    """Keep numeric columns, replace inf with NaN, drop all-NaN columns."""
    out = df.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    return out.dropna(axis=1, how="all")


class Preprocessor:
    """Median-impute -> standardise -> clip. Remembers the column list for inference."""

    def __init__(self):
        self.columns: list[str] = []
        self.pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
        self.keep_mask: np.ndarray | None = None

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        x = numeric_features(df)
        self.columns = list(x.columns)
        self.pipe.fit(x.values.astype(float))
        # drop constant columns (StandardScaler reports scale_=1 for them, so test the variance)
        self.keep_mask = self.pipe.named_steps["sc"].var_ > 1e-12
        self.feature_names = [c for c, k in zip(self.columns, self.keep_mask) if k]
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        x = df.reindex(columns=self.columns).replace([np.inf, -np.inf], np.nan)
        arr = self.pipe.transform(x.values.astype(float))
        return np.clip(arr[:, self.keep_mask], -CLIP, CLIP).astype(np.float32)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    def compatible(self, df: pd.DataFrame, min_overlap: float = 0.9) -> bool:
        return len(set(self.columns) & set(df.columns)) / max(len(self.columns), 1) >= min_overlap
