from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.core.config import settings
from app.pricing_engine.feature_builder import ArtifactBundle, PricingFeatureBuilder


@dataclass
class TrainingArtifacts:
    pipeline: Pipeline
    anomaly_pipeline: Pipeline
    bundle: ArtifactBundle


class PricingModelManager:
    def __init__(self, artifacts_dir: str | Path | None = None):
        self.artifacts_dir = Path(artifacts_dir or settings.artifacts_dir)
        self.model_path = Path(settings.model_path if artifacts_dir is None else self.artifacts_dir / "pricing_model.joblib")
        self.anomaly_path = Path(settings.anomaly_path if artifacts_dir is None else self.artifacts_dir / "anomaly_model.joblib")
        self.metadata_path = Path(settings.metadata_path if artifacts_dir is None else self.artifacts_dir / "pricing_metadata.json")
        self._artifacts: TrainingArtifacts | None = None

    @staticmethod
    def _one_hot() -> OneHotEncoder:
        try:
            return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        except TypeError:
            return OneHotEncoder(handle_unknown="ignore", sparse=True)

    def _build_preprocessor(self, df: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
        numeric_cols = [
            c for c in df.columns
            if pd.api.types.is_numeric_dtype(df[c]) and c not in {"price_ratio", "negotiated_price_usd"}
        ]
        numeric_cols = [c for c in numeric_cols if not df[c].isna().all()]
        categorical_cols = [
            c for c in df.columns
            if c not in numeric_cols
            and c not in {"price_ratio", "negotiated_price_usd", "transaction_id"}
            and (df[c].dtype == "object" or str(df[c].dtype).startswith("string") or str(df[c].dtype) == "category")
        ]

        num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler(with_mean=False))])
        cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", self._one_hot())])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", num_pipe, numeric_cols),
                ("cat", cat_pipe, categorical_cols),
            ],
            remainder="drop",
            sparse_threshold=0.0,
        )
        return preprocessor, numeric_cols, categorical_cols

    def train(self, data_dir: str | Path, random_state: int = 42, max_rows: int = 1000) -> TrainingArtifacts:
        builder = PricingFeatureBuilder.from_data_dir(data_dir)
        df = builder.build_training_frame()
        if df.empty:
            raise ValueError("No training data found in data_dir")

        df = df.replace([np.inf, -np.inf], np.nan).copy()
        if max_rows and len(df) > max_rows:
            df = df.sample(n=max_rows, random_state=random_state).reset_index(drop=True)
        target = "price_ratio"
        df[target] = df["price_ratio"].clip(0.50, 1.80)

        drop_cols = [
            "transaction_id", "company_name", "contact_id", "email", "phone", "website",
            "reason_code", "stage", "outcome", "source_system", "account_id", "product_id",
        ]
        feature_df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

        train_df, test_df = train_test_split(feature_df, test_size=0.2, random_state=random_state)

        preprocessor, numeric_cols, categorical_cols = self._build_preprocessor(train_df)

        model = RandomForestRegressor(
            n_estimators=320,
            max_depth=14,
            min_samples_split=8,
            min_samples_leaf=3,
            random_state=random_state,
            n_jobs=-1,
        )

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model),
        ])
        pipeline.fit(train_df.drop(columns=[target]), train_df[target])

        pred = pipeline.predict(test_df.drop(columns=[target]))
        rmse = float(np.sqrt(mean_squared_error(test_df[target], pred)))
        mae = float(mean_absolute_error(test_df[target], pred))
        r2 = float(r2_score(test_df[target], pred))

        anomaly_features = train_df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        anomaly_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("iso", IsolationForest(
                n_estimators=200,
                contamination=0.03,
                random_state=random_state,
            )),
        ])
        anomaly_pipeline.fit(anomaly_features)

        feature_names = list(numeric_cols) + categorical_cols
        feature_importance = self._feature_importance(pipeline, feature_names)

        product_stats = self._build_product_stats(df)
        account_stats = self._build_account_stats(df)

        bundle = ArtifactBundle(
            product_stats=product_stats,
            account_stats=account_stats,
            feature_columns=list(train_df.columns),
            numeric_columns=list(numeric_cols),
            categorical_columns=list(categorical_cols),
            feature_importance=feature_importance,
            model_version="rf-price-ratio-v1",
            metrics={"rmse": rmse, "mae": mae, "r2": r2},
        )

        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, self.model_path)
        joblib.dump(anomaly_pipeline, self.anomaly_path)
        self.metadata_path.write_text(json.dumps({
            "model_version": bundle.model_version,
            "metrics": bundle.metrics,
            "feature_importance": bundle.feature_importance[:40],
            "feature_columns": bundle.feature_columns,
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "product_stats": product_stats,
            "account_stats": account_stats,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

        self._artifacts = TrainingArtifacts(pipeline=pipeline, anomaly_pipeline=anomaly_pipeline, bundle=bundle)
        return self._artifacts

    def _build_product_stats(self, df: pd.DataFrame) -> dict[str, dict[str, Any]]:
        stats: dict[str, dict[str, Any]] = {}
        for key, grp in df.groupby("product_id"):
            base = grp["base_price_usd"].replace(0, np.nan)
            cost = grp["cost_usd"].replace(0, np.nan) if "cost_usd" in grp.columns else grp["base_price_usd"] * 0.58
            ratio = np.nanmean(cost / base)
            stats[str(key)] = {
                "cost_ratio": float(np.clip(ratio if np.isfinite(ratio) else 0.58, 0.25, 0.85)),
                "avg_margin_pct": float(np.nanmean(grp.get("gross_margin_pct", pd.Series([0.35]))) if "gross_margin_pct" in grp.columns else 0.35),
                "avg_discount_pct": float(np.nanmean(grp["list_discount_pct"]) if "list_discount_pct" in grp.columns else 0.08),
            }
        return stats

    def _build_account_stats(self, df: pd.DataFrame) -> dict[str, dict[str, Any]]:
        stats: dict[str, dict[str, Any]] = {}
        for key, grp in df.groupby("account_id"):
            stats[str(key)] = {
                "avg_price_sensitivity": float(grp["price_sensitivity"].mean()) if "price_sensitivity" in grp.columns else 0.45,
                "avg_service_attachment": float(grp["service_attachment"].mean()) if "service_attachment" in grp.columns else 0.5,
                "avg_win_probability": float(grp["win_probability"].mean()) if "win_probability" in grp.columns else 0.5,
                "avg_loyalty_score": float(grp["loyalty_score"].mean()) if "loyalty_score" in grp.columns else 0.5,
            }
        return stats

    def _feature_importance(self, pipeline: Pipeline, raw_feature_names: list[str]) -> list[tuple[str, float]]:
        preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
        model: RandomForestRegressor = pipeline.named_steps["model"]
        try:
            ohe = preprocessor.named_transformers_["cat"].named_steps["onehot"]
            cat_cols = preprocessor.transformers_[1][2]
            cat_names = list(ohe.get_feature_names_out(cat_cols))
            all_names = list(preprocessor.transformers_[0][2]) + cat_names
        except Exception:
            all_names = raw_feature_names

        importances = getattr(model, "feature_importances_", None)
        if importances is None:
            return [(name, 0.0) for name in all_names]
        pairs = sorted(zip(all_names, importances), key=lambda x: x[1], reverse=True)
        return [(str(k), float(v)) for k, v in pairs]

    def load(self) -> TrainingArtifacts:
        if self._artifacts is not None:
            return self._artifacts
        if not self.model_path.exists() or not self.anomaly_path.exists() or not self.metadata_path.exists():
            raise FileNotFoundError("Pricing artifacts not found. Run scripts/train_pricing_model.py first.")
        pipeline = joblib.load(self.model_path)
        anomaly_pipeline = joblib.load(self.anomaly_path)
        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        bundle = ArtifactBundle(
            product_stats=metadata.get("product_stats", {}),
            account_stats=metadata.get("account_stats", {}),
            feature_columns=metadata.get("feature_columns", []),
            numeric_columns=metadata.get("numeric_columns", []),
            categorical_columns=metadata.get("categorical_columns", []),
            feature_importance=[tuple(x) for x in metadata.get("feature_importance", [])],
            model_version=metadata.get("model_version", "unknown"),
            metrics=metadata.get("metrics", {}),
        )
        self._artifacts = TrainingArtifacts(pipeline=pipeline, anomaly_pipeline=anomaly_pipeline, bundle=bundle)
        return self._artifacts

    def predict_ratio(self, request_frame: pd.DataFrame) -> tuple[float, float]:
        artifacts = self.load()
        pred = float(artifacts.pipeline.predict(request_frame)[0])
        pred = float(np.clip(pred, 0.55, 1.80))
        return pred, float(artifacts.bundle.metrics.get("r2", 0.0))

    def detect_anomaly(self, request_frame: pd.DataFrame) -> tuple[float, bool]:
        artifacts = self.load()
        numeric_cols = artifacts.bundle.numeric_columns
        if numeric_cols:
            X = request_frame.reindex(columns=numeric_cols, fill_value=np.nan)
        else:
            X = request_frame.select_dtypes(include=[np.number]).fillna(0.0)
        score = float(artifacts.anomaly_pipeline.decision_function(X)[0])
        anomaly = bool(artifacts.anomaly_pipeline.predict(X)[0] == -1)
        return score, anomaly

    def explain_features(self, top_k: int = 8) -> list[tuple[str, float]]:
        artifacts = self.load()
        return artifacts.bundle.feature_importance[:top_k]
