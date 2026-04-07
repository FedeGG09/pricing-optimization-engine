from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class AnomalyResult:
    score: float
    is_anomaly: bool
    reason: str


class PricingAnomalyDetector:
    def __init__(self, pipeline: Pipeline | None = None):
        self.pipeline = pipeline

    @classmethod
    def train(cls, numeric_df: pd.DataFrame) -> "PricingAnomalyDetector":
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("iso", IsolationForest(n_estimators=200, contamination=0.03, random_state=42)),
        ])
        pipeline.fit(numeric_df.replace([np.inf, -np.inf], np.nan).fillna(0.0))
        return cls(pipeline)

    def score(self, numeric_df: pd.DataFrame) -> AnomalyResult:
        if self.pipeline is None:
            return AnomalyResult(score=0.0, is_anomaly=False, reason="no_model")
        X = numeric_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        raw_score = float(self.pipeline.named_steps["iso"].decision_function(
            self.pipeline.named_steps["scaler"].transform(
                self.pipeline.named_steps["imputer"].transform(X)
            )
        )[0])
        pred = bool(self.pipeline.named_steps["iso"].predict(
            self.pipeline.named_steps["scaler"].transform(
                self.pipeline.named_steps["imputer"].transform(X)
            )
        )[0] == -1)
        reason = "outlier_feature_pattern" if pred else "normal"
        return AnomalyResult(score=raw_score, is_anomaly=pred, reason=reason)
