from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "pricing"
SQLITE_PATH = DATA_DIR / "industrial_pricing_features.sqlite"
TABLE_NAME = "pricing_training_set"

TARGET_COL = "negotiated_price_usd"
DATE_COL = "date"

MODEL_PATH = ARTIFACTS_DIR / "pricing_model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "pricing_model_metrics.json"
FEATURES_PATH = ARTIFACTS_DIR / "pricing_feature_columns.json"


def load_data() -> pd.DataFrame:
    if not SQLITE_PATH.exists():
        raise FileNotFoundError(f"No existe la SQLite: {SQLITE_PATH}")

    with sqlite3.connect(str(SQLITE_PATH)) as conn:
        df = pd.read_sql_query(f"SELECT * FROM {TABLE_NAME}", conn)

    if df.empty:
        raise ValueError(f"La tabla {TABLE_NAME} está vacía")

    return df


def sanitize(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[TARGET_COL] = pd.to_numeric(out[TARGET_COL], errors="coerce")
    out = out.dropna(subset=[TARGET_COL])
    out = out[out[TARGET_COL] > 0].copy()
    if DATE_COL in out.columns:
        out[DATE_COL] = pd.to_datetime(out[DATE_COL], errors="coerce")
        out = out.sort_values(DATE_COL)
    return out


def time_split(df: pd.DataFrame, test_size: float = 0.2):
    cut = int(len(df) * (1 - test_size))
    train_df = df.iloc[:cut].copy()
    valid_df = df.iloc[cut:].copy()
    return train_df, valid_df


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No existe el modelo: {MODEL_PATH}")

    with FEATURES_PATH.open("r", encoding="utf-8") as f:
        features_meta = json.load(f)

    numeric_cols = features_meta["numeric_cols"]
    categorical_cols = features_meta["categorical_cols"]
    feature_cols = numeric_cols + categorical_cols

    df = sanitize(load_data())
    train_df, valid_df = time_split(df, test_size=0.2)

    X_valid = valid_df[feature_cols].copy()
    y_valid = valid_df[TARGET_COL].copy()

    pipeline = joblib.load(MODEL_PATH)
    y_pred = pipeline.predict(X_valid)

    mae = float(mean_absolute_error(y_valid, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_valid, y_pred)))
    r2 = float(r2_score(y_valid, y_pred))
    mape = float(np.mean(np.abs((y_valid - y_pred) / np.clip(np.abs(y_valid), 1e-6, None))) * 100.0)

    metrics = {
        "model_name": "pricing_model",
        "target_col": TARGET_COL,
        "date_col": DATE_COL,
        "training_rows": int(len(train_df)),
        "validation_rows": int(len(valid_df)),
        "feature_count": int(len(feature_cols)),
        "metrics": {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "mape_pct": mape,
        },
        "trained_at_utc": datetime.utcnow().isoformat(),
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("Evaluation completed")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()