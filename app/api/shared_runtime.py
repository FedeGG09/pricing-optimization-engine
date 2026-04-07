from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "pricing"
SQLITE_PATH = DATA_DIR / "industrial_pricing_features.sqlite"
METRICS_PATH = ARTIFACTS_DIR / "pricing_model_metrics.json"
AUDIT_TABLE = "pricing_decisions_log"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def to_mapping(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump()
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return dict(value.__dict__)
        except Exception:
            pass
    try:
        return dict(value)
    except Exception:
        return {}


def json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [json_safe(v) for v in value]
    return value


def json_dump(value: Any) -> str:
    return json.dumps(json_safe(value), ensure_ascii=False, default=str)


def merge_non_null(target: Dict[str, Any], source: Dict[str, Any]) -> None:
    for key, value in source.items():
        if not is_missing(value):
            target[key] = value


def to_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    if is_missing(value):
        return default
    try:
        return int(float(value))
    except Exception:
        return default


def to_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if is_missing(value):
        return default
    try:
        return float(value)
    except Exception:
        return default


def model_dump_compat(model: Any) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    if hasattr(model, "dict"):
        return model.dict()
    return to_mapping(model)


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
    return cur.fetchone() is not None


def read_table(table_name: str) -> pd.DataFrame:
    if not SQLITE_PATH.exists():
        return pd.DataFrame()
    with sqlite3.connect(str(SQLITE_PATH)) as conn:
        if not table_exists(conn, table_name):
            return pd.DataFrame()
        try:
            return pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        except Exception:
            return pd.DataFrame()


def ensure_audit_schema() -> None:
    SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(SQLITE_PATH)) as conn:
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {AUDIT_TABLE} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at_utc TEXT NOT NULL,
                endpoint TEXT NOT NULL,
                account_id TEXT NOT NULL,
                product_id TEXT NOT NULL,
                context_source TEXT NOT NULL,
                request_json TEXT NOT NULL,
                enriched_payload_json TEXT NOT NULL,
                response_json TEXT NOT NULL,
                status_code INTEGER NOT NULL,
                error TEXT,
                model_version TEXT
            )
            """
        )
        conn.commit()


def load_metrics() -> Dict[str, Any]:
    if not METRICS_PATH.exists():
        return {
            "status": "pending",
            "detail": "Todavía no se generaron métricas. Ejecutá scripts/evaluate_pricing_model.py primero.",
        }
    with METRICS_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def global_defaults() -> Dict[str, Any]:
    defaults: Dict[str, Any] = {
        "month": int(datetime.now().month),
        "dayofweek": int(datetime.now().weekday()),
        "is_peak_season_proxy": 0,
        "month_sin": 0.0,
        "month_cos": 1.0,
        "risk_score_0_100": 50.0,
        "rfm_score": 3.0,
        "urgency_score": 0.5,
        "churn_risk": 0.5,
        "compliance_rate": 0.8,
        "price_sensitivity": 0.5,
        "annual_budget_multiplier": 1.0,
        "service_attachment": 0.5,
        "quantity": 1.0,
        "stock_units": 10.0,
        "margin_target_pct": 0.25,
        "elasticity_estimate": -1.0,
        "avg_discount_pct": 0.10,
        "avg_margin_pct": 0.30,
        "price_floor_usd": None,
        "price_ceiling_usd": None,
        "negotiated_price_usd": None,
        "list_price_usd": None,
        "base_price_usd": None,
        "province": None,
        "zone": None,
        "segment_rule_based": "new_or_low_activity",
        "segment_kmeans_label": "new_or_low_activity",
    }

    df = read_table("pricing_training_set")
    if not df.empty:
        numeric_candidates = [
            "list_price_usd",
            "base_price_usd",
            "negotiated_price_usd",
            "avg_discount_pct",
            "avg_margin_pct",
            "risk_score_0_100",
            "rfm_score",
            "price_floor_usd",
            "price_ceiling_usd",
            "seasonality_index",
            "urgency_score",
            "churn_risk",
            "compliance_rate",
            "price_sensitivity",
            "annual_budget_multiplier",
            "service_attachment",
            "quantity",
            "stock_units",
            "margin_target_pct",
            "elasticity_estimate",
        ]
        for col in numeric_candidates:
            if col in df.columns:
                s = pd.to_numeric(df[col], errors="coerce")
                med = s.median()
                if not pd.isna(med):
                    defaults[col] = float(med)

        categorical_candidates = [
            "province",
            "zone",
            "segment_rule_based",
            "segment_kmeans_label",
            "funnel_stage",
            "next_best_action",
        ]
        for col in categorical_candidates:
            if col in df.columns:
                mode = df[col].dropna().astype(str).mode()
                if not mode.empty:
                    defaults[col] = mode.iloc[0]

    return defaults


def read_exact_training_row(account_id: str, product_id: str) -> Dict[str, Any]:
    if not SQLITE_PATH.exists():
        return {}

    with sqlite3.connect(str(SQLITE_PATH)) as conn:
        if not table_exists(conn, "pricing_training_set"):
            return {}

        try:
            df = pd.read_sql_query(
                """
                SELECT *
                FROM pricing_training_set
                WHERE CAST(account_id AS TEXT) = CAST(? AS TEXT)
                  AND CAST(product_id AS TEXT) = CAST(? AS TEXT)
                ORDER BY date DESC
                LIMIT 1
                """,
                conn,
                params=(str(account_id), str(product_id)),
            )
        except Exception:
            return {}

    if df.empty:
        return {}

    return {k: v for k, v in df.iloc[0].to_dict().items() if not is_missing(v)}


def read_single_row_by_key(table_name: str, key_col: str, key_value: Any) -> Dict[str, Any]:
    if not SQLITE_PATH.exists():
        return {}

    with sqlite3.connect(str(SQLITE_PATH)) as conn:
        if not table_exists(conn, table_name):
            return {}

        try:
            df = pd.read_sql_query(
                f"""
                SELECT *
                FROM {table_name}
                WHERE CAST({key_col} AS TEXT) = CAST(? AS TEXT)
                LIMIT 1
                """,
                conn,
                params=(str(key_value),),
            )
        except Exception:
            return {}

    if df.empty:
        return {}

    return {k: v for k, v in df.iloc[0].to_dict().items() if not is_missing(v)}


def load_seasonality_context(product_id: str, month: int, family: Optional[str]) -> Dict[str, Any]:
    if not SQLITE_PATH.exists():
        return {}

    with sqlite3.connect(str(SQLITE_PATH)) as conn:
        if not table_exists(conn, "seasonality_index"):
            return {}

        queries: List[Tuple[str, Tuple[Any, ...]]] = []

        if family is not None:
            queries.extend(
                [
                    (
                        "SELECT * FROM seasonality_index WHERE family = ? AND month = ? LIMIT 1",
                        (family, month),
                    ),
                    (
                        """
                        SELECT AVG(seasonality_revenue_index) AS seasonality_revenue_index,
                               AVG(seasonality_qty_index) AS seasonality_qty_index
                        FROM seasonality_index
                        WHERE family = ?
                        """,
                        (family,),
                    ),
                ]
            )

        queries.extend(
            [
                (
                    "SELECT * FROM seasonality_index WHERE CAST(product_id AS TEXT) = CAST(? AS TEXT) AND month = ? LIMIT 1",
                    (str(product_id), month),
                ),
                (
                    """
                    SELECT AVG(seasonality_revenue_index) AS seasonality_revenue_index,
                           AVG(seasonality_qty_index) AS seasonality_qty_index
                    FROM seasonality_index
                    WHERE CAST(product_id AS TEXT) = CAST(? AS TEXT)
                    """,
                    (str(product_id),),
                ),
            ]
        )

        for sql, params in queries:
            try:
                df = pd.read_sql_query(sql, conn, params=params)
            except Exception:
                continue
            if not df.empty:
                row = {k: v for k, v in df.iloc[0].to_dict().items() if not is_missing(v)}
                if row:
                    return row

    return {}


def build_enriched_payload(
    *,
    account_id: str,
    product_id: str,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    overrides = overrides or {}

    exact_row = read_exact_training_row(account_id, product_id)
    account_row = read_single_row_by_key("account_features", "account_id", account_id)
    product_row = read_single_row_by_key("product_features", "product_id", product_id)

    month = (
        to_int(overrides.get("month"))
        or to_int(exact_row.get("month"))
        or to_int(product_row.get("month"))
        or int(global_defaults()["month"])
    )
    dayofweek = (
        to_int(overrides.get("dayofweek"))
        or to_int(exact_row.get("dayofweek"))
        or to_int(product_row.get("dayofweek"))
        or int(global_defaults()["dayofweek"])
    )

    family = None
    if "family" in product_row and not is_missing(product_row["family"]):
        family = str(product_row["family"])
    elif "family" in exact_row and not is_missing(exact_row["family"]):
        family = str(exact_row["family"])

    seasonality_row = load_seasonality_context(product_id, int(month), family)

    payload: Dict[str, Any] = dict(global_defaults())
    payload.update(
        {
            "account_id": account_id,
            "product_id": product_id,
            "month": int(month),
            "dayofweek": int(dayofweek),
            "is_peak_season_proxy": 1 if int(month) in {11, 12, 1, 2} else 0,
            "month_sin": float(np.sin(2 * np.pi * int(month) / 12.0)),
            "month_cos": float(np.cos(2 * np.pi * int(month) / 12.0)),
        }
    )

    context_source = "fallback_enriched"
    used_sources = ["global_defaults"]

    if exact_row:
        merge_non_null(payload, exact_row)
        context_source = "historical_exact_match"
        used_sources.append("pricing_training_set_exact")
    if account_row:
        merge_non_null(payload, account_row)
        used_sources.append("account_features")
    if product_row:
        merge_non_null(payload, product_row)
        used_sources.append("product_features")
    if seasonality_row:
        merge_non_null(payload, seasonality_row)
        used_sources.append("seasonality_index")

    merge_non_null(payload, {k: v for k, v in overrides.items() if not is_missing(v)})

    if is_missing(payload.get("list_price_usd")):
        if not is_missing(payload.get("avg_list_price_usd")):
            payload["list_price_usd"] = payload["avg_list_price_usd"]
        elif not is_missing(payload.get("avg_price_usd")):
            payload["list_price_usd"] = float(payload["avg_price_usd"]) * 1.15

    if is_missing(payload.get("base_price_usd")):
        if not is_missing(payload.get("avg_base_price_usd")):
            payload["base_price_usd"] = payload["avg_base_price_usd"]
        elif not is_missing(payload.get("list_price_usd")):
            payload["base_price_usd"] = float(payload["list_price_usd"]) * 0.72

    if is_missing(payload.get("negotiated_price_usd")):
        if not is_missing(payload.get("avg_price_usd")):
            payload["negotiated_price_usd"] = payload["avg_price_usd"]
        elif not is_missing(payload.get("list_price_usd")):
            payload["negotiated_price_usd"] = float(payload["list_price_usd"]) * 0.92

    if is_missing(payload.get("price_floor_usd")):
        if not is_missing(payload.get("min_price_usd")):
            payload["price_floor_usd"] = payload["min_price_usd"]
        elif not is_missing(payload.get("negotiated_price_usd")):
            payload["price_floor_usd"] = float(payload["negotiated_price_usd"]) * 0.85
        elif not is_missing(payload.get("list_price_usd")):
            payload["price_floor_usd"] = float(payload["list_price_usd"]) * 0.80

    if is_missing(payload.get("price_ceiling_usd")):
        if not is_missing(payload.get("max_price_usd")):
            payload["price_ceiling_usd"] = payload["max_price_usd"]
        elif not is_missing(payload.get("list_price_usd")):
            payload["price_ceiling_usd"] = float(payload["list_price_usd"]) * 1.05
        elif not is_missing(payload.get("negotiated_price_usd")):
            payload["price_ceiling_usd"] = float(payload["negotiated_price_usd"]) * 1.20

    if is_missing(payload.get("margin_target_pct")):
        payload["margin_target_pct"] = payload.get("avg_margin_pct") or global_defaults()["margin_target_pct"]
    if is_missing(payload.get("risk_score_0_100")):
        payload["risk_score_0_100"] = 50.0
    if is_missing(payload.get("rfm_score")):
        payload["rfm_score"] = 3.0
    if is_missing(payload.get("segment_rule_based")):
        payload["segment_rule_based"] = payload.get("segment_kmeans_label") or "new_or_low_activity"

    payload = {k: v for k, v in payload.items() if not is_missing(v) or k in {"price_floor_usd", "price_ceiling_usd"}}

    return {
        "context_source": context_source,
        "used_sources": used_sources,
        "exact_row": exact_row,
        "account_row": account_row,
        "product_row": product_row,
        "seasonality_row": seasonality_row,
        "defaults": global_defaults(),
        "payload": payload,
        "month_used": int(month),
        "dayofweek_used": int(dayofweek),
        "matched_exact_row": bool(exact_row),
        "missing_after_enrichment": sorted([k for k, v in payload.items() if is_missing(v)]),
    }


def log_decision(
    *,
    endpoint: str,
    account_id: str,
    product_id: str,
    context_source: str,
    request_json: Dict[str, Any],
    enriched_payload_json: Dict[str, Any],
    response_json: Dict[str, Any],
    status_code: int,
    error: Optional[str] = None,
    model_version: Optional[str] = None,
) -> Optional[int]:
    try:
        ensure_audit_schema()
        with sqlite3.connect(str(SQLITE_PATH)) as conn:
            cur = conn.cursor()
            cur.execute(
                f"""
                INSERT INTO {AUDIT_TABLE} (
                    created_at_utc,
                    endpoint,
                    account_id,
                    product_id,
                    context_source,
                    request_json,
                    enriched_payload_json,
                    response_json,
                    status_code,
                    error,
                    model_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now_utc(),
                    endpoint,
                    str(account_id),
                    str(product_id),
                    context_source,
                    json_dump(request_json),
                    json_dump(enriched_payload_json),
                    json_dump(response_json),
                    int(status_code),
                    error,
                    model_version,
                ),
            )
            conn.commit()
            return int(cur.lastrowid)
    except Exception:
        return None


def read_audit_decision(decision_id: int) -> Dict[str, Any]:
    if not SQLITE_PATH.exists():
        return {}

    with sqlite3.connect(str(SQLITE_PATH)) as conn:
        if not table_exists(conn, AUDIT_TABLE):
            return {}

        try:
            df = pd.read_sql_query(
                f"SELECT * FROM {AUDIT_TABLE} WHERE id = ? LIMIT 1",
                conn,
                params=(int(decision_id),),
            )
        except Exception:
            return {}

    if df.empty:
        return {}

    return df.iloc[0].to_dict()


def safe_json_load_text(value: Any) -> Any:
    if value is None:
        return None
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except Exception:
        return value


def pricing_response(
    recommendation: Any,
    context: Dict[str, Any],
    decision_id: Optional[int],
    fallback_used: bool,
) -> Dict[str, Any]:
    rec = to_mapping(recommendation)

    recommended_price = to_float(rec.get("recommended_price_usd"))
    model_price = to_float(rec.get("model_price_usd"))
    floor_price = to_float(rec.get("floor_price_usd"))
    ceiling_price = to_float(rec.get("ceiling_price_usd"))
    list_price = to_float(global_defaults().get("list_price_usd"))
    if list_price is None:
        list_price = ceiling_price or model_price

    discount = to_float(rec.get("recommended_discount_pct"))
    if discount is None and list_price is not None and list_price > 0 and recommended_price is not None:
        discount = max(0.0, min(0.75, 1.0 - (recommended_price / list_price)))

    margin = to_float(rec.get("final_margin_estimate_pct"))

    ui_payload = {
        "status": "ok",
        "decision_id": decision_id,
        "fallback_used": fallback_used,
        "context_source": context["context_source"],
        "price": {
            "recommended": round(recommended_price, 4) if recommended_price is not None else None,
            "model_price": round(model_price, 4) if model_price is not None else None,
            "floor": round(floor_price, 4) if floor_price is not None else None,
            "ceiling": round(ceiling_price, 4) if ceiling_price is not None else None,
        },
        "discount": {
            "recommended_pct": round(discount, 4) if discount is not None else None,
        },
        "margin": {
            "expected_pct": round(margin, 4) if margin is not None else None,
        },
        "reason": rec.get("explanation", ""),
        "factor_scores": rec.get("factor_scores", {}),
        "rule_adjustments": rec.get("rule_adjustments", {}),
        "model_version": rec.get("model_version"),
        "context": context,
    }

    metrics = load_metrics()
    ui_payload["model_metrics"] = metrics.get("metrics", metrics) if isinstance(metrics, dict) else metrics
    ui_payload["trained_at_utc"] = metrics.get("trained_at_utc") if isinstance(metrics, dict) else None
    ui_payload["feature_count"] = metrics.get("feature_count") if isinstance(metrics, dict) else None
    ui_payload["training_rows"] = metrics.get("training_rows") if isinstance(metrics, dict) else None
    ui_payload["validation_rows"] = metrics.get("validation_rows") if isinstance(metrics, dict) else None
    return ui_payload


def load_reference_list(table_name: str, kind: str, limit: int = 5000) -> List[Dict[str, Any]]:
    df = read_table(table_name)
    if df.empty:
        return []

    id_candidates = [
        f"{kind}_id",
        "id",
        "account_id",
        "product_id",
        "client_id",
        "customer_id",
        "province_id",
        "zone_id",
        "region_id",
        "code",
    ]
    name_candidates = [
        "name",
        f"{kind}_name",
        "account_name",
        "customer_name",
        "client_name",
        "product_name",
        "description",
        "title",
        "legal_name",
        "display_name",
        "province_name",
        "zone_name",
        "region_name",
    ]

    id_col = next((c for c in id_candidates if c in df.columns), df.columns[0])
    name_col = next((c for c in name_candidates if c in df.columns), "__fallback__")

    cols = [id_col]
    if name_col != "__fallback__" and name_col not in cols:
        cols.append(name_col)

    for extra in ["province", "region", "zone", "segment", "category", "family", "subfamily", "state", "city"]:
        if extra in df.columns and extra not in cols:
            cols.append(extra)

    out = df[cols].drop_duplicates().head(limit).copy()
    if name_col != "__fallback__":
        out = out.rename(columns={id_col: "id", name_col: "name"})
    else:
        out = out.rename(columns={id_col: "id"})
    return out.to_dict(orient="records")