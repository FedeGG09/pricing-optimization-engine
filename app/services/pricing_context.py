from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SQLITE_PATH = PROJECT_ROOT / "data" / "industrial_pricing_features.sqlite"


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _clean_dict(d: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        if _is_missing(v):
            continue
        if isinstance(v, (np.integer, np.floating)):
            out[k] = v.item()
        elif isinstance(v, pd.Timestamp):
            out[k] = v.isoformat()
        else:
            out[k] = v
    return out


@dataclass
class PricingContextService:
    sqlite_path: Path = DEFAULT_SQLITE_PATH

    def _table_exists(self, conn: sqlite3.Connection, table_name: str) -> bool:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        )
        return cur.fetchone() is not None

    def _read_table(self, table_name: str) -> pd.DataFrame:
        if not self.sqlite_path.exists():
            return pd.DataFrame()
        with sqlite3.connect(str(self.sqlite_path)) as conn:
            if not self._table_exists(conn, table_name):
                return pd.DataFrame()
            try:
                return pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            except Exception:
                return pd.DataFrame()

    def _read_single_row(self, table_name: str, key_col: str, key_value: Any) -> dict[str, Any]:
        if not self.sqlite_path.exists():
            return {}
        with sqlite3.connect(str(self.sqlite_path)) as conn:
            if not self._table_exists(conn, table_name):
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
        return _clean_dict(df.iloc[0].to_dict())

    def _read_exact_training_row(self, account_id: str, product_id: str) -> dict[str, Any]:
        if not self.sqlite_path.exists():
            return {}
        with sqlite3.connect(str(self.sqlite_path)) as conn:
            if not self._table_exists(conn, "pricing_training_set"):
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
        return _clean_dict(df.iloc[0].to_dict())

    def _load_seasonality_context(self, product_id: str, month: int, family: Optional[str]) -> dict[str, Any]:
        if not self.sqlite_path.exists():
            return {}

        with sqlite3.connect(str(self.sqlite_path)) as conn:
            if not self._table_exists(conn, "seasonality_index"):
                return {}

            queries = []
            if family is not None:
                queries.extend(
                    [
                        (
                            """
                            SELECT *
                            FROM seasonality_index
                            WHERE family = ? AND month = ?
                            LIMIT 1
                            """,
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
                        """
                        SELECT *
                        FROM seasonality_index
                        WHERE CAST(product_id AS TEXT) = CAST(? AS TEXT) AND month = ?
                        LIMIT 1
                        """,
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
                    row = _clean_dict(df.iloc[0].to_dict())
                    if row:
                        return row

        return {}

    def _global_defaults(self) -> dict[str, Any]:
        df = self._read_table("pricing_training_set")
        defaults: dict[str, Any] = {
            "month": int(pd.Timestamp.now().month),
            "dayofweek": int(pd.Timestamp.now().dayofweek),
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

    def build_context(
        self,
        *,
        account_id: str,
        product_id: str,
        overrides: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        overrides = overrides or {}

        exact_row = self._read_exact_training_row(account_id, product_id)
        account_row = self._read_single_row("account_features", "account_id", account_id)
        product_row = self._read_single_row("product_features", "product_id", product_id)

        month = (
            int(overrides.get("month"))
            if overrides.get("month") is not None
            else int(exact_row.get("month") or product_row.get("month") or pd.Timestamp.now().month)
        )
        dayofweek = (
            int(overrides.get("dayofweek"))
            if overrides.get("dayofweek") is not None
            else int(exact_row.get("dayofweek") or product_row.get("dayofweek") or pd.Timestamp.now().dayofweek)
        )

        family = None
        if product_row.get("family") is not None:
            family = str(product_row.get("family"))
        elif exact_row.get("family") is not None:
            family = str(exact_row.get("family"))

        seasonality_row = self._load_seasonality_context(product_id, month, family)
        defaults = self._global_defaults()

        payload: dict[str, Any] = dict(defaults)
        payload.update(
            {
                "account_id": account_id,
                "product_id": product_id,
                "month": month,
                "dayofweek": dayofweek,
                "is_peak_season_proxy": 1 if month in {11, 12, 1, 2} else 0,
                "month_sin": float(np.sin(2 * np.pi * month / 12.0)),
                "month_cos": float(np.cos(2 * np.pi * month / 12.0)),
            }
        )

        used_sources = ["global_defaults"]
        context_source = "fallback_enriched"

        if exact_row:
            payload.update(exact_row)
            used_sources.append("pricing_training_set_exact")
            context_source = "historical_exact_match"
        if account_row:
            payload.update(account_row)
            used_sources.append("account_features")
        if product_row:
            payload.update(product_row)
            used_sources.append("product_features")
        if seasonality_row:
            payload.update(seasonality_row)
            used_sources.append("seasonality_index")

        payload.update({k: v for k, v in overrides.items() if v is not None})

        if payload.get("list_price_usd") is None:
            if payload.get("avg_list_price_usd") is not None:
                payload["list_price_usd"] = payload["avg_list_price_usd"]
            elif payload.get("avg_price_usd") is not None:
                payload["list_price_usd"] = float(payload["avg_price_usd"]) * 1.15

        if payload.get("base_price_usd") is None:
            if payload.get("avg_base_price_usd") is not None:
                payload["base_price_usd"] = payload["avg_base_price_usd"]
            elif payload.get("list_price_usd") is not None:
                payload["base_price_usd"] = float(payload["list_price_usd"]) * 0.72

        if payload.get("negotiated_price_usd") is None:
            if payload.get("avg_price_usd") is not None:
                payload["negotiated_price_usd"] = payload["avg_price_usd"]
            elif payload.get("list_price_usd") is not None:
                payload["negotiated_price_usd"] = float(payload["list_price_usd"]) * 0.92

        if payload.get("price_floor_usd") is None:
            if payload.get("min_price_usd") is not None:
                payload["price_floor_usd"] = payload["min_price_usd"]
            elif payload.get("negotiated_price_usd") is not None:
                payload["price_floor_usd"] = float(payload["negotiated_price_usd"]) * 0.85
            elif payload.get("list_price_usd") is not None:
                payload["price_floor_usd"] = float(payload["list_price_usd"]) * 0.80

        if payload.get("price_ceiling_usd") is None:
            if payload.get("max_price_usd") is not None:
                payload["price_ceiling_usd"] = payload["max_price_usd"]
            elif payload.get("list_price_usd") is not None:
                payload["price_ceiling_usd"] = float(payload["list_price_usd"]) * 1.05
            elif payload.get("negotiated_price_usd") is not None:
                payload["price_ceiling_usd"] = float(payload["negotiated_price_usd"]) * 1.20

        if payload.get("margin_target_pct") is None:
            payload["margin_target_pct"] = payload.get("avg_margin_pct") or defaults["margin_target_pct"]

        if payload.get("risk_score_0_100") is None:
            payload["risk_score_0_100"] = defaults["risk_score_0_100"]
        if payload.get("rfm_score") is None:
            payload["rfm_score"] = defaults["rfm_score"]
        if payload.get("segment_rule_based") is None:
            payload["segment_rule_based"] = payload.get("segment_kmeans_label") or defaults["segment_rule_based"]

        payload = _clean_dict(payload)
        return {
            "context_source": context_source,
            "used_sources": used_sources,
            "exact_row": exact_row,
            "account_row": account_row,
            "product_row": product_row,
            "seasonality_row": seasonality_row,
            "defaults": defaults,
            "payload": payload,
        }