from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(r"C:\Users\Usuario\Documents\Repos\sullair_pricing\data")
DEFAULT_SQLITE_PATH = BASE_DIR / "industrial_dataset.sqlite"
DEFAULT_OUTPUT_DB_PATH = BASE_DIR / "industrial_pricing_features.sqlite"


RAW_SOURCES = {
    "transactions": "transactions.csv",
    "account_master": "account_master.csv",
    "contact_master": "contact_master.csv",
    "product_master": "product_master.csv",
    "service_events": "service_events.csv",
    "lead_funnel": "lead_funnel.csv",
    "audit_log": "audit_log.csv",
    "catalog": "catalog.json",
    "data_dictionary": "data_dictionary.json",
    "sample_records": "sample_records.json",
}


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_json_if_exists(path: Path) -> Any:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df


def _safe_to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=False)


def _ensure_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default)


def _json_safe(value: Any) -> Any:
    """
    Converts nested objects into SQLite-safe scalars.
    """
    if isinstance(value, (list, dict, tuple, set, np.ndarray)):
        try:
            return json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            return str(value)
    return value


def serialize_dataframe_for_sqlite(df: pd.DataFrame) -> pd.DataFrame:
    """
    Makes a DataFrame safe for SQLite insertion.
    - serializes list/dict/array columns to JSON strings
    - normalizes datetime-like columns to string
    - converts unsupported object values to strings
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    for col in out.columns:
        # Serialize nested Python objects
        if out[col].dtype == "object":
            out[col] = out[col].apply(_json_safe)

        # Datetime columns to string
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = pd.to_datetime(out[col], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")

    # Replace NaN / NaT with None for SQLite friendliness
    out = out.where(pd.notnull(out), None)
    return out


def load_sources(base_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Load CSV/JSON files from the provided directory.
    If a JSON is a dict, it becomes a 1-row DataFrame.
    If a JSON is a list, it becomes a DataFrame with one row per item.
    """
    sources: Dict[str, pd.DataFrame] = {}

    for table, filename in RAW_SOURCES.items():
        path = base_dir / filename
        if filename.endswith(".csv"):
            sources[table] = _normalize_columns(_read_csv_if_exists(path))
        elif filename.endswith(".json"):
            payload = _read_json_if_exists(path)
            if isinstance(payload, dict):
                sources[table] = pd.DataFrame([payload])
            elif isinstance(payload, list):
                sources[table] = pd.DataFrame(payload)
            else:
                sources[table] = pd.DataFrame()
        else:
            sources[table] = pd.DataFrame()

    return sources


def load_from_sqlite(sqlite_path: Path, table_name: str) -> pd.DataFrame:
    if not sqlite_path.exists():
        return pd.DataFrame()

    with sqlite3.connect(str(sqlite_path)) as conn:
        try:
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        except Exception:
            return pd.DataFrame()
    return _normalize_columns(df)


def import_raw_to_sqlite(sqlite_path: Path, sources: Dict[str, pd.DataFrame]) -> None:
    """
    Persist raw sources into SQLite so the whole dataset is self-contained.
    Existing tables are replaced.
    """
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(str(sqlite_path)) as conn:
        for table_name, df in sources.items():
            if df is None or df.empty:
                continue

            df_to_write = df.copy()

            # Normalize date-like columns
            for col in df_to_write.columns:
                if col.lower() in {"date", "created_at", "updated_at", "timestamp", "event_date"}:
                    df_to_write[col] = pd.to_datetime(df_to_write[col], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")

            df_to_write = serialize_dataframe_for_sqlite(df_to_write)
            df_to_write.to_sql(table_name, conn, if_exists="replace", index=False)


def ensure_core_tables(sqlite_path: Path, sources: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Return a dictionary with the best available version of each table,
    falling back to SQLite if a CSV/JSON file is missing locally.
    """
    out = dict(sources)

    if out.get("audit_log", pd.DataFrame()).empty:
        out["audit_log"] = load_from_sqlite(sqlite_path, "audit_log")

    for table in ("transactions", "account_master", "contact_master", "product_master", "service_events", "lead_funnel"):
        if out.get(table, pd.DataFrame()).empty:
            out[table] = load_from_sqlite(sqlite_path, table)

    return out


def add_time_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    out = df.copy()
    out[date_col] = _safe_to_datetime(out[date_col])

    out["year"] = out[date_col].dt.year
    out["month"] = out[date_col].dt.month
    out["day"] = out[date_col].dt.day
    out["dayofweek"] = out[date_col].dt.dayofweek
    out["quarter"] = out[date_col].dt.quarter
    out["weekofyear"] = out[date_col].dt.isocalendar().week.astype("Int64")
    out["is_month_start"] = out[date_col].dt.is_month_start.astype("Int64")
    out["is_month_end"] = out[date_col].dt.is_month_end.astype("Int64")
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12.0)
    out["is_q4"] = (out["quarter"] == 4).astype("Int64")
    out["is_peak_season_proxy"] = out["month"].isin([11, 12, 1, 2]).astype("Int64")
    return out


def build_account_features(transactions: pd.DataFrame, accounts: pd.DataFrame, service_events: pd.DataFrame) -> pd.DataFrame:
    tx = transactions.copy()
    if tx.empty:
        return pd.DataFrame()

    tx["date"] = _safe_to_datetime(tx["date"])

    for col in [
        "list_price_usd",
        "negotiated_price_usd",
        "gross_margin_pct",
        "win_probability",
        "churn_risk",
        "urgency_score",
        "compliance_ok",
        "macro_index",
        "seasonality_index",
        "quantity",
        "revenue_usd",
        "base_price_usd",
    ]:
        if col in tx.columns:
            tx[col] = _ensure_numeric(tx[col])
        else:
            tx[col] = np.nan

    tx["discount_pct"] = np.where(
        tx["list_price_usd"].notna() & (tx["list_price_usd"] > 0),
        1.0 - (tx["negotiated_price_usd"] / tx["list_price_usd"]),
        np.nan,
    )
    tx["margin_pct"] = tx["gross_margin_pct"]

    cutoff = tx["date"].max()
    if pd.isna(cutoff):
        cutoff = pd.Timestamp.today()

    agg = tx.groupby("account_id").agg(
        transaction_count=("transaction_id", "count"),
        first_transaction_date=("date", "min"),
        last_transaction_date=("date", "max"),
        total_revenue_usd=("revenue_usd", "sum"),
        avg_revenue_usd=("revenue_usd", "mean"),
        avg_price_usd=("negotiated_price_usd", "mean"),
        avg_base_price_usd=("base_price_usd", "mean"),
        avg_list_price_usd=("list_price_usd", "mean"),
        avg_discount_pct=("discount_pct", "mean"),
        avg_margin_pct=("margin_pct", "mean"),
        avg_quantity=("quantity", "mean"),
        avg_win_probability=("win_probability", "mean"),
        avg_churn_risk=("churn_risk", "mean"),
        avg_urgency_score=("urgency_score", "mean"),
        avg_macro_index=("macro_index", "mean"),
        avg_seasonality_index=("seasonality_index", "mean"),
        compliance_rate=("compliance_ok", "mean"),
    ).reset_index()

    agg["tenure_days"] = (cutoff - agg["first_transaction_date"]).dt.days.clip(lower=0)
    agg["recency_days"] = (cutoff - agg["last_transaction_date"]).dt.days.clip(lower=0)

    freq_90 = (
        tx.loc[tx["date"] >= (cutoff - pd.Timedelta(days=90))]
        .groupby("account_id")["transaction_id"]
        .count()
        .reindex(agg["account_id"])
        .fillna(0)
        .values
    )
    freq_180 = (
        tx.loc[tx["date"] >= (cutoff - pd.Timedelta(days=180))]
        .groupby("account_id")["transaction_id"]
        .count()
        .reindex(agg["account_id"])
        .fillna(0)
        .values
    )

    agg["frequency_90d"] = freq_90
    agg["frequency_180d"] = freq_180

    if not service_events.empty:
        se = service_events.copy()
        if "date" in se.columns:
            se["date"] = _safe_to_datetime(se["date"])

        for col in ["resolved", "sla_hours", "labor_hours", "revenue_usd"]:
            if col in se.columns:
                se[col] = _ensure_numeric(se[col])
            else:
                se[col] = np.nan

        se_agg = se.groupby("account_id").agg(
            service_events_count=("service_id", "count"),
            service_resolved_rate=("resolved", "mean"),
            avg_sla_hours=("sla_hours", "mean"),
            avg_labor_hours=("labor_hours", "mean"),
            service_revenue_usd=("revenue_usd", "sum"),
        ).reset_index()

        agg = agg.merge(se_agg, on="account_id", how="left")
    else:
        agg["service_events_count"] = np.nan
        agg["service_resolved_rate"] = np.nan
        agg["avg_sla_hours"] = np.nan
        agg["avg_labor_hours"] = np.nan
        agg["service_revenue_usd"] = np.nan

    if not accounts.empty:
        accounts = accounts.copy().drop_duplicates("account_id")
        agg = agg.merge(accounts, on="account_id", how="left", suffixes=("", "_account"))

    risk_components = pd.DataFrame(index=agg.index)
    risk_components["churn_risk"] = agg["avg_churn_risk"].fillna(0.5)
    risk_components["inverted_compliance"] = 1.0 - agg["compliance_rate"].fillna(0.8)
    risk_components["recency_pressure"] = np.tanh(agg["recency_days"].fillna(365) / 180.0)
    risk_components["service_friction"] = 1.0 - agg["service_resolved_rate"].fillna(0.85)
    risk_components["sla_pressure"] = np.tanh(agg["avg_sla_hours"].fillna(24) / 24.0)
    risk_components["urgency_pressure"] = agg["avg_urgency_score"].fillna(0.5)

    agg["risk_score_0_100"] = (
        100.0
        * (
            0.30 * risk_components["churn_risk"]
            + 0.20 * risk_components["inverted_compliance"]
            + 0.15 * risk_components["recency_pressure"]
            + 0.15 * risk_components["service_friction"]
            + 0.10 * risk_components["sla_pressure"]
            + 0.10 * risk_components["urgency_pressure"]
        )
    ).clip(0, 100)

    def _score_quantile(series: pd.Series, reverse: bool = False) -> pd.Series:
        s = series.fillna(series.median() if series.notna().any() else 0)
        try:
            bins = pd.qcut(s.rank(method="first"), q=5, labels=[1, 2, 3, 4, 5])
            out = bins.astype(float)
        except Exception:
            out = pd.Series(np.full(len(s), 3.0), index=s.index)
        if reverse:
            out = 6.0 - out
        return out

    agg["recency_score"] = _score_quantile(agg["recency_days"], reverse=True)
    agg["frequency_score"] = _score_quantile(agg["transaction_count"], reverse=False)
    agg["monetary_score"] = _score_quantile(agg["total_revenue_usd"], reverse=False)
    agg["rfm_score"] = agg["recency_score"] * 0.2 + agg["frequency_score"] * 0.35 + agg["monetary_score"] * 0.45

    conditions = [
        (agg["rfm_score"] >= 4.2) & (agg["risk_score_0_100"] < 30),
        (agg["rfm_score"] >= 3.2) & (agg["risk_score_0_100"] < 55),
        (agg["risk_score_0_100"] >= 70),
    ]
    choices = ["vip", "recurrent", "at_risk"]
    agg["segment_rule_based"] = np.select(conditions, choices, default="new_or_low_activity")

    return agg


def build_product_features(transactions: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    tx = transactions.copy()
    if tx.empty:
        return pd.DataFrame()

    tx["date"] = _safe_to_datetime(tx["date"])

    for col in ["quantity", "negotiated_price_usd", "base_price_usd", "list_price_usd", "gross_margin_pct", "revenue_usd", "win_probability"]:
        if col in tx.columns:
            tx[col] = _ensure_numeric(tx[col])
        else:
            tx[col] = np.nan

    tx["discount_pct"] = np.where(
        tx["list_price_usd"].notna() & (tx["list_price_usd"] > 0),
        1.0 - (tx["negotiated_price_usd"] / tx["list_price_usd"]),
        np.nan,
    )
    tx["margin_pct"] = tx["gross_margin_pct"]
    tx["quantity"] = tx["quantity"].fillna(1.0)

    prod = tx.groupby("product_id").agg(
        txn_count=("transaction_id", "count"),
        avg_qty=("quantity", "mean"),
        avg_negotiated_price=("negotiated_price_usd", "mean"),
        avg_base_price=("base_price_usd", "mean"),
        avg_list_price=("list_price_usd", "mean"),
        avg_discount_pct=("discount_pct", "mean"),
        avg_margin_pct=("margin_pct", "mean"),
        price_std=("negotiated_price_usd", "std"),
        qty_std=("quantity", "std"),
    ).reset_index()

    if "family" in tx.columns:
        fam = tx.groupby(["family", "subfamily"], dropna=False).agg(
            family_txn_count=("transaction_id", "count"),
            family_revenue_usd=("revenue_usd", "sum"),
            family_avg_discount_pct=("discount_pct", "mean"),
            family_avg_margin_pct=("gross_margin_pct", "mean"),
            family_avg_win_probability=("win_probability", "mean"),
        ).reset_index()

        prod = prod.merge(
            tx[["product_id", "family", "subfamily"]].drop_duplicates("product_id"),
            on="product_id",
            how="left",
        )
        prod = prod.merge(fam, on=["family", "subfamily"], how="left")
    else:
        prod["family_txn_count"] = np.nan
        prod["family_revenue_usd"] = np.nan
        prod["family_avg_discount_pct"] = np.nan
        prod["family_avg_margin_pct"] = np.nan
        prod["family_avg_win_probability"] = np.nan

    elasticity_rows = []
    if "family" in tx.columns:
        grouped = tx.groupby("family")
        key_name = "family"
    else:
        grouped = tx.groupby("product_id")
        key_name = "product_id"

    for key, g in grouped:
        g = g.dropna(subset=["negotiated_price_usd", "quantity"])
        if len(g) < 8 or g["negotiated_price_usd"].nunique() < 2:
            elasticity = np.nan
            r2 = np.nan
        else:
            x = np.log(g["negotiated_price_usd"].clip(lower=1.0)).to_numpy().reshape(-1, 1)
            y = np.log(g["quantity"].clip(lower=0) + 1.0).to_numpy()
            model = LinearRegression()
            model.fit(x, y)
            elasticity = float(model.coef_[0])
            r2 = float(model.score(x, y))

        elasticity_rows.append(
            {
                "elasticity_key": key,
                "elasticity_estimate": elasticity,
                "elasticity_r2": r2,
                "elasticity_n": int(len(g)),
            }
        )

    elasticity_df = pd.DataFrame(elasticity_rows)

    if not elasticity_df.empty:
        if key_name == "family":
            elasticity_df = elasticity_df.rename(columns={"elasticity_key": "family"})
            prod = prod.merge(elasticity_df, on="family", how="left")
        else:
            elasticity_df = elasticity_df.rename(columns={"elasticity_key": "product_id"})
            prod = prod.merge(elasticity_df, on="product_id", how="left")
    else:
        prod["elasticity_estimate"] = np.nan
        prod["elasticity_r2"] = np.nan
        prod["elasticity_n"] = np.nan

    prod["elasticity_estimate"] = prod["elasticity_estimate"].fillna(-1.0)

    if "min_price_usd" in prod.columns:
        prod["price_floor_usd"] = prod["min_price_usd"]
    else:
        prod["price_floor_usd"] = prod["avg_negotiated_price"] * 0.85

    if "max_price_usd" in prod.columns:
        prod["price_ceiling_usd"] = prod["max_price_usd"]
    else:
        prod["price_ceiling_usd"] = prod["avg_list_price"] * 1.10

    prod["observed_discount_depth"] = (
        1.0 - prod["avg_negotiated_price"] / prod["avg_list_price"]
    ).replace([np.inf, -np.inf], np.nan)

    if not products.empty:
        products = products.copy().drop_duplicates("product_id")
        prod = prod.merge(products, on="product_id", how="left", suffixes=("", "_master"))

    return prod


def build_pricing_training_set(
    transactions: pd.DataFrame,
    account_features: pd.DataFrame,
    product_features: pd.DataFrame,
    lead_funnel: pd.DataFrame,
) -> pd.DataFrame:
    tx = transactions.copy()
    if tx.empty:
        return pd.DataFrame()

    tx = add_time_features(tx, "date")

    for col in ["list_price_usd", "negotiated_price_usd", "gross_margin_pct", "quantity", "revenue_usd"]:
        if col in tx.columns:
            tx[col] = _ensure_numeric(tx[col])
        else:
            tx[col] = np.nan

    tx["discount_pct"] = np.where(
        tx["list_price_usd"].notna() & (tx["list_price_usd"] > 0),
        1.0 - (tx["negotiated_price_usd"] / tx["list_price_usd"]),
        np.nan,
    )
    tx["margin_pct"] = tx["gross_margin_pct"]
    tx["quantity"] = tx["quantity"].fillna(1.0)

    account_cols = [
        "account_id",
        "transaction_count",
        "avg_discount_pct",
        "avg_margin_pct",
        "risk_score_0_100",
        "rfm_score",
        "segment_rule_based",
        "price_sensitivity",
        "annual_budget_multiplier",
        "account_tier",
        "province",
        "segment_kmeans_label",
        "segment_cluster_id",
    ]
    account_cols = [c for c in account_cols if c in account_features.columns]
    features = tx.merge(account_features[account_cols], on="account_id", how="left", suffixes=("", "_acct"))

    product_cols = [
        c
        for c in [
            "product_id",
            "avg_discount_pct",
            "avg_margin_pct",
            "elasticity_estimate",
            "price_floor_usd",
            "price_ceiling_usd",
            "criticality",
        ]
        if c in product_features.columns
    ]
    features = features.merge(product_features[product_cols], on="product_id", how="left", suffixes=("", "_prod"))

    if not lead_funnel.empty:
        lf = lead_funnel.copy().drop_duplicates("account_id")
        lf_cols = [c for c in ["account_id", "industry_fit_score", "funnel_stage", "next_best_action"] if c in lf.columns]
        if len(lf_cols) > 1:
            features = features.merge(lf[lf_cols], on="account_id", how="left")

    features["discount_depth"] = features["discount_pct"].fillna(0)

    if "margin_pct" in features.columns:
        margin_base = features["margin_pct"].median(skipna=True)
        if pd.isna(margin_base):
            margin_base = 0.3
    else:
        margin_base = 0.3

    features["margin_pressure"] = features["avg_margin_pct"].fillna(margin_base) - features["discount_depth"].fillna(0)

    if "price_floor_usd" in features.columns:
        features["price_position_vs_floor"] = np.where(
            features["price_floor_usd"].notna() & (features["price_floor_usd"] > 0),
            (features["negotiated_price_usd"] - features["price_floor_usd"]) / features["price_floor_usd"],
            np.nan,
        )
    else:
        features["price_position_vs_floor"] = np.nan

    if "price_ceiling_usd" in features.columns:
        features["price_position_vs_ceiling"] = np.where(
            features["price_ceiling_usd"].notna() & (features["price_ceiling_usd"] > 0),
            (features["price_ceiling_usd"] - features["negotiated_price_usd"]) / features["price_ceiling_usd"],
            np.nan,
        )
    else:
        features["price_position_vs_ceiling"] = np.nan

    features["target_negotiated_price_usd"] = features["negotiated_price_usd"]
    features["target_discount_pct"] = features["discount_pct"]
    features["target_margin_pct"] = features["margin_pct"]

    return features.replace([np.inf, -np.inf], np.nan)


def assign_kmeans_segments(account_features: pd.DataFrame, n_clusters: int = 4, random_state: int = 42) -> pd.DataFrame:
    if account_features.empty:
        return pd.DataFrame()

    df = account_features.copy()
    candidate_cols = [
        "transaction_count",
        "total_revenue_usd",
        "avg_margin_pct",
        "avg_discount_pct",
        "recency_days",
        "frequency_90d",
        "risk_score_0_100",
        "price_sensitivity",
        "annual_budget_multiplier",
        "service_attachment",
    ]
    use_cols = [c for c in candidate_cols if c in df.columns]

    if len(use_cols) < 2:
        df["segment_kmeans"] = "segment_0"
        df["segment_cluster_id"] = 0
        df["segment_kmeans_label"] = "new_or_low_activity"
        return df

    X = df[use_cols].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    n_clusters = int(min(n_clusters, max(2, len(df))))
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = km.fit_predict(Xs)

    df["segment_kmeans"] = [f"segment_{i}" for i in labels]
    df["segment_cluster_id"] = labels

    centroids = pd.DataFrame(km.cluster_centers_, columns=use_cols)

    revenue_col = "total_revenue_usd" if "total_revenue_usd" in centroids.columns else use_cols[0]
    freq_col = "frequency_90d" if "frequency_90d" in centroids.columns else use_cols[0]
    risk_col = "risk_score_0_100" if "risk_score_0_100" in centroids.columns else use_cols[0]

    ranking = (
        centroids[revenue_col].rank(method="dense")
        + centroids[freq_col].rank(method="dense")
        - centroids[risk_col].rank(method="dense")
    )

    premium_cluster = int(ranking.idxmax())
    at_risk_cluster = int(centroids[risk_col].idxmax())

    label_map = {}
    for cid in range(n_clusters):
        if cid == premium_cluster:
            label_map[cid] = "vip"
        elif cid == at_risk_cluster:
            label_map[cid] = "at_risk"
        else:
            label_map[cid] = "recurrent"

    df["segment_kmeans_label"] = df["segment_cluster_id"].map(label_map).fillna("new_or_low_activity")
    return df


def compute_seasonality_index(transactions: pd.DataFrame) -> pd.DataFrame:
    tx = transactions.copy()
    if tx.empty:
        return pd.DataFrame()

    tx["date"] = _safe_to_datetime(tx["date"])
    tx["month"] = tx["date"].dt.month

    for col in ["quantity", "revenue_usd"]:
        if col in tx.columns:
            tx[col] = _ensure_numeric(tx[col])
        else:
            tx[col] = np.nan

    tx["quantity"] = tx["quantity"].fillna(1.0)

    group_col = "family" if "family" in tx.columns else "product_id"
    monthly = tx.groupby([group_col, "month"], dropna=False).agg(
        monthly_revenue=("revenue_usd", "sum"),
        monthly_qty=("quantity", "sum"),
        txn_count=("transaction_id", "count"),
    ).reset_index()

    baseline = monthly.groupby(group_col).agg(
        avg_monthly_revenue=("monthly_revenue", "mean"),
        avg_monthly_qty=("monthly_qty", "mean"),
    ).reset_index()

    seasonality = monthly.merge(baseline, on=group_col, how="left")

    seasonality["seasonality_revenue_index"] = np.where(
        seasonality["avg_monthly_revenue"] > 0,
        seasonality["monthly_revenue"] / seasonality["avg_monthly_revenue"] - 1.0,
        np.nan,
    )
    seasonality["seasonality_qty_index"] = np.where(
        seasonality["avg_monthly_qty"] > 0,
        seasonality["monthly_qty"] / seasonality["avg_monthly_qty"] - 1.0,
        np.nan,
    )

    return seasonality


def write_outputs_to_sqlite(
    sqlite_path: Path,
    sources: Dict[str, pd.DataFrame],
    account_features: pd.DataFrame,
    product_features: pd.DataFrame,
    pricing_training_set: pd.DataFrame,
    seasonality_index: pd.DataFrame,
    account_segments: pd.DataFrame,
) -> None:
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(str(sqlite_path)) as conn:
        for table_name, df in sources.items():
            if df is None or df.empty:
                continue
            out = serialize_dataframe_for_sqlite(df)
            out.to_sql(table_name, conn, if_exists="replace", index=False)

        if not account_features.empty:
            serialize_dataframe_for_sqlite(account_features).to_sql("account_features", conn, if_exists="replace", index=False)
        if not product_features.empty:
            serialize_dataframe_for_sqlite(product_features).to_sql("product_features", conn, if_exists="replace", index=False)
        if not pricing_training_set.empty:
            serialize_dataframe_for_sqlite(pricing_training_set).to_sql("pricing_training_set", conn, if_exists="replace", index=False)
        if not seasonality_index.empty:
            serialize_dataframe_for_sqlite(seasonality_index).to_sql("seasonality_index", conn, if_exists="replace", index=False)
        if not account_segments.empty:
            serialize_dataframe_for_sqlite(account_segments).to_sql("account_segments", conn, if_exists="replace", index=False)

        conn.execute("DROP VIEW IF EXISTS v_pricing_training_set")
        try:
            conn.execute("CREATE VIEW v_pricing_training_set AS SELECT * FROM pricing_training_set")
        except Exception:
            pass

        conn.commit()


def build_all(base_dir: Path, sqlite_path: Path, output_sqlite_path: Optional[Path] = None) -> Dict[str, pd.DataFrame]:
    output_sqlite_path = output_sqlite_path or sqlite_path

    sources = load_sources(base_dir)
    import_raw_to_sqlite(output_sqlite_path, sources)
    sources = ensure_core_tables(output_sqlite_path, sources)

    transactions = sources.get("transactions", pd.DataFrame())
    accounts = sources.get("account_master", pd.DataFrame())
    products = sources.get("product_master", pd.DataFrame())
    services = sources.get("service_events", pd.DataFrame())
    leads = sources.get("lead_funnel", pd.DataFrame())

    account_features = build_account_features(transactions, accounts, services)
    account_features = assign_kmeans_segments(account_features, n_clusters=4)
    product_features = build_product_features(transactions, products)
    pricing_training_set = build_pricing_training_set(transactions, account_features, product_features, leads)
    seasonality_index = compute_seasonality_index(transactions)

    write_outputs_to_sqlite(
        output_sqlite_path,
        sources,
        account_features,
        product_features,
        pricing_training_set,
        seasonality_index,
        account_features,
    )

    return {
        "transactions": transactions,
        "account_features": account_features,
        "product_features": product_features,
        "pricing_training_set": pricing_training_set,
        "seasonality_index": seasonality_index,
        "account_segments": account_features,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build pricing dataset, feature tables and SQLite database.")
    parser.add_argument("--base-dir", type=str, default=str(BASE_DIR), help="Folder containing CSV/JSON sources.")
    parser.add_argument("--sqlite-path", type=str, default=str(DEFAULT_SQLITE_PATH), help="Base SQLite database path.")
    parser.add_argument("--output-sqlite-path", type=str, default=str(DEFAULT_OUTPUT_DB_PATH), help="Output SQLite path to write raw + feature tables.")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    sqlite_path = Path(args.sqlite_path)
    output_sqlite_path = Path(args.output_sqlite_path)

    result = build_all(base_dir, sqlite_path, output_sqlite_path)

    print("\nBuild completed.")
    print(f"Base directory: {base_dir}")
    print(f"Source SQLite:   {sqlite_path}")
    print(f"Output SQLite:   {output_sqlite_path}")

    for key, value in result.items():
        if isinstance(value, pd.DataFrame):
            print(f"{key}: {len(value):,} rows")
        else:
            print(f"{key}: {type(value)}")


if __name__ == "__main__":
    main()