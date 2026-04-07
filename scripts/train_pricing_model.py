from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "pricing"
SQLITE_PATH = DATA_DIR / "industrial_pricing_features.sqlite"
TABLE_NAME = "pricing_training_set"

TARGET_COL = "negotiated_price_usd"
DATE_COL = "date"

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def load_training_data(sqlite_path: Path, table_name: str) -> pd.DataFrame:
    if not sqlite_path.exists():
        raise FileNotFoundError(f"No existe la base SQLite: {sqlite_path}")

    with sqlite3.connect(str(sqlite_path)) as conn:
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)

    if df.empty:
        raise ValueError(f"La tabla {table_name} está vacía.")

    return df


def time_based_split(df: pd.DataFrame, date_col: str, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()

    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.sort_values(date_col).reset_index(drop=True)
        cut_idx = int(len(df) * (1.0 - test_size))
        train_df = df.iloc[:cut_idx].copy()
        valid_df = df.iloc[cut_idx:].copy()
    else:
        cut_idx = int(len(df) * (1.0 - test_size))
        train_df = df.iloc[:cut_idx].copy()
        valid_df = df.iloc[cut_idx:].copy()

    return train_df, valid_df


def infer_feature_columns(df: pd.DataFrame, target_col: str, date_col: str) -> Tuple[List[str], List[str]]:
    drop_cols = {
        target_col,
        "target_discount_pct",
        "target_margin_pct",
        "target_negotiated_price_usd",
        date_col,
    }

    candidate_features = [c for c in df.columns if c not in drop_cols]

    categorical_cols = []
    numeric_cols = []

    for col in candidate_features:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    return numeric_cols, categorical_cols


def sanitize_target(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    out = df.copy()
    out[target_col] = pd.to_numeric(out[target_col], errors="coerce")
    out = out.dropna(subset=[target_col])
    out = out[out[target_col] > 0].copy()
    return out


def build_pipeline(numeric_cols: List[str], categorical_cols: List[str]) -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.06,
        max_depth=8,
        max_iter=250,
        min_samples_leaf=25,
        l2_regularization=0.1,
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    return pipeline


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)

    mape = np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-6, None))) * 100.0

    return {
        "mae": float(mae),
        "rmse": rmse,
        "r2": float(r2),
        "mape_pct": float(mape),
    }


def get_feature_importance_like_summary(pipeline: Pipeline, X_train: pd.DataFrame) -> Dict[str, Any]:
    """
    HistGradientBoosting doesn't expose classic feature importances.
    We return a useful summary of the input schema and transformed feature count.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    try:
        transformed = preprocessor.fit_transform(X_train)
        transformed_shape = list(transformed.shape)
    except Exception:
        transformed_shape = [0, 0]

    return {
        "numeric_features": X_train.select_dtypes(include=[np.number]).columns.tolist(),
        "categorical_features": X_train.select_dtypes(exclude=[np.number]).columns.tolist(),
        "transformed_shape": transformed_shape,
    }


def train() -> None:
    df = load_training_data(SQLITE_PATH, TABLE_NAME)
    df = sanitize_target(df, TARGET_COL)

    if df.empty:
        raise ValueError("No hay datos válidos para entrenar.")

    train_df, valid_df = time_based_split(df, DATE_COL, test_size=0.2)

    numeric_cols, categorical_cols = infer_feature_columns(train_df, TARGET_COL, DATE_COL)

    if not numeric_cols and not categorical_cols:
        raise ValueError("No se detectaron features para entrenar.")

    X_train = train_df[numeric_cols + categorical_cols].copy()
    y_train = train_df[TARGET_COL].copy()

    X_valid = valid_df[numeric_cols + categorical_cols].copy()
    y_valid = valid_df[TARGET_COL].copy()

    pipeline = build_pipeline(numeric_cols, categorical_cols)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_valid)
    metrics = evaluate_model(y_valid.to_numpy(), y_pred)

    artifact_bundle = {
        "pipeline": pipeline,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "target_col": TARGET_COL,
        "date_col": DATE_COL,
        "training_rows": int(len(train_df)),
        "validation_rows": int(len(valid_df)),
        "metrics": metrics,
        "trained_at_utc": datetime.utcnow().isoformat(),
        "feature_summary": get_feature_importance_like_summary(pipeline, X_train),
    }

    model_path = ARTIFACTS_DIR / "pricing_model.joblib"
    meta_path = ARTIFACTS_DIR / "pricing_model_metadata.json"
    features_path = ARTIFACTS_DIR / "pricing_feature_columns.json"

    joblib.dump(pipeline, model_path)

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "target_col": TARGET_COL,
                "date_col": DATE_COL,
                "training_rows": artifact_bundle["training_rows"],
                "validation_rows": artifact_bundle["validation_rows"],
                "metrics": metrics,
                "trained_at_utc": artifact_bundle["trained_at_utc"],
                "numeric_cols": numeric_cols,
                "categorical_cols": categorical_cols,
                "sqlite_path": str(SQLITE_PATH),
                "table_name": TABLE_NAME,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    with features_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "numeric_cols": numeric_cols,
                "categorical_cols": categorical_cols,
                "all_features": numeric_cols + categorical_cols,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\nTraining completed")
    print(f"Model saved to: {model_path}")
    print(f"Metadata saved to: {meta_path}")
    print(f"Feature columns saved to: {features_path}")
    print("\nValidation metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")


if __name__ == "__main__":
    train()