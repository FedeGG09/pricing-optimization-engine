from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _safe_lower(value: Any) -> str:
    return str(value).strip().lower() if value is not None and str(value).strip() else ""


def _seasonality_from_month(month: int) -> float:
    # Seno estacional suave en [-1, 1]
    return float(np.sin((month - 1) / 12.0 * 2.0 * np.pi))


def _province_competition_index(province: str | None) -> float:
    mapping = {
        "buenos aires": 0.85,
        "ciudad autónoma de buenos aires": 0.82,
        "cordoba": 0.79,
        "córdoba": 0.79,
        "santa fe": 0.77,
        "mendoza": 0.74,
        "neuquén": 0.72,
        "neuquen": 0.72,
        "santa cruz": 0.63,
        "chubut": 0.65,
        "tierra del fuego": 0.60,
    }
    return mapping.get(_safe_lower(province), 0.70)


def _province_logistics_index(province: str | None) -> float:
    mapping = {
        "buenos aires": 0.85,
        "ciudad autónoma de buenos aires": 0.82,
        "cordoba": 0.77,
        "córdoba": 0.77,
        "santa fe": 0.79,
        "mendoza": 0.73,
        "neuquén": 0.68,
        "neuquen": 0.68,
        "santa cruz": 0.58,
        "chubut": 0.60,
        "tierra del fuego": 0.52,
    }
    return mapping.get(_safe_lower(province), 0.70)


def _province_purchasing_power_index(province: str | None) -> float:
    mapping = {
        "buenos aires": 0.92,
        "ciudad autónoma de buenos aires": 0.95,
        "cordoba": 0.78,
        "córdoba": 0.78,
        "santa fe": 0.79,
        "mendoza": 0.76,
        "neuquén": 0.74,
        "neuquen": 0.74,
        "santa cruz": 0.69,
        "chubut": 0.70,
    }
    return mapping.get(_safe_lower(province), 0.73)


def _segment_loyalty(segment: str) -> float:
    return {"new": 0.30, "recurrent": 0.55, "vip": 0.90, "at_risk": 0.20}.get(_safe_lower(segment), 0.50)


def _segment_payment_score(segment: str) -> float:
    return {"new": 0.45, "recurrent": 0.62, "vip": 0.85, "at_risk": 0.25}.get(_safe_lower(segment), 0.55)


@dataclass
class ArtifactBundle:
    product_stats: dict[str, dict[str, Any]]
    account_stats: dict[str, dict[str, Any]]
    feature_columns: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]
    feature_importance: list[tuple[str, float]]
    model_version: str
    metrics: dict[str, float]


class PricingFeatureBuilder:
    def __init__(
        self,
        product_master: pd.DataFrame | None = None,
        account_master: pd.DataFrame | None = None,
        lead_funnel: pd.DataFrame | None = None,
        service_events: pd.DataFrame | None = None,
        transactions: pd.DataFrame | None = None,
        artifact_bundle: ArtifactBundle | None = None,
    ):
        self.product_master = product_master.copy() if product_master is not None else pd.DataFrame()
        self.account_master = account_master.copy() if account_master is not None else pd.DataFrame()
        self.lead_funnel = lead_funnel.copy() if lead_funnel is not None else pd.DataFrame()
        self.service_events = service_events.copy() if service_events is not None else pd.DataFrame()
        self.transactions = transactions.copy() if transactions is not None else pd.DataFrame()
        self.artifact_bundle = artifact_bundle

        self.product_stats = artifact_bundle.product_stats if artifact_bundle else {}
        self.account_stats = artifact_bundle.account_stats if artifact_bundle else {}

        self._normalize_frames()

    @classmethod
    def from_data_dir(cls, data_dir: str | Path) -> "PricingFeatureBuilder":
        data_dir = Path(data_dir)
        def load(name: str) -> pd.DataFrame:
            path = data_dir / name
            return pd.read_csv(path) if path.exists() else pd.DataFrame()

        return cls(
            product_master=load("product_master.csv"),
            account_master=load("account_master.csv"),
            lead_funnel=load("lead_funnel.csv"),
            service_events=load("service_events.csv"),
            transactions=load("transactions.csv"),
        )

    @classmethod
    def from_artifacts(
        cls,
        data_dir: str | Path,
        artifact_bundle: ArtifactBundle,
    ) -> "PricingFeatureBuilder":
        obj = cls.from_data_dir(data_dir)
        obj.artifact_bundle = artifact_bundle
        obj.product_stats = artifact_bundle.product_stats
        obj.account_stats = artifact_bundle.account_stats
        return obj

    def _normalize_frames(self) -> None:
        if not self.product_master.empty and "product_id" in self.product_master:
            self.product_master["product_id"] = self.product_master["product_id"].astype(str)

        if not self.account_master.empty and "account_id" in self.account_master:
            self.account_master["account_id"] = self.account_master["account_id"].astype(str)

        if not self.transactions.empty and "date" in self.transactions:
            self.transactions["date"] = pd.to_datetime(self.transactions["date"], errors="coerce")

    def build_training_frame(self) -> pd.DataFrame:
        if self.transactions.empty:
            return pd.DataFrame()

        df = self.transactions.copy()
        if not self.product_master.empty:
            df = df.merge(self.product_master, on="product_id", how="left", suffixes=("", "_prod"))
        if not self.account_master.empty:
            df = df.merge(self.account_master, on="account_id", how="left", suffixes=("", "_acct"))
        if not self.lead_funnel.empty and "account_id" in self.lead_funnel:
            lead_cols = [c for c in ["account_id", "industry_fit_score", "funnel_stage", "next_best_action"] if c in self.lead_funnel.columns]
            df = df.merge(self.lead_funnel[lead_cols].drop_duplicates("account_id"), on="account_id", how="left")

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["month"] = df["date"].dt.month.fillna(6).astype(int)
        df["quarter"] = df["date"].dt.quarter.fillna(2).astype(int)
        df["dayofweek"] = df["date"].dt.dayofweek.fillna(2).astype(int)
        df["is_month_end"] = df["date"].dt.is_month_end.fillna(False).astype(int)

        df["price_ratio"] = np.where(df["base_price_usd"] > 0, df["negotiated_price_usd"] / df["base_price_usd"], 1.0)
        df["list_discount_pct"] = np.where(df["list_price_usd"] > 0, 1 - (df["negotiated_price_usd"] / df["list_price_usd"]), 0.0)
        df["margin_pct"] = df.get("gross_margin_pct", np.where(df["revenue_usd"] > 0, df["gross_margin_usd"] / df["revenue_usd"], 0.0))
        df["margin_buffer_pct"] = df["margin_pct"] - df.get("min_margin_required", 0.2)

        df["price_sensitivity"] = df.get("price_sensitivity", 0.45)
        df["annual_budget_multiplier"] = df.get("annual_budget_multiplier", 1.0)
        df["service_attachment"] = df.get("service_attachment", 0.5)
        df["industry_fit_score"] = df.get("industry_fit_score", 0.5)
        df["customer_segment"] = df.get("account_tier", df.get("customer_segment", "recurrent"))
        df["province"] = df.get("province", "unknown")
        df["sector"] = df.get("sector", "unknown")
        df["industry"] = df.get("industry", "unknown")
        df["section"] = df.get("section", "unknown")
        df["family"] = df.get("family", "unknown")
        df["subfamily"] = df.get("subfamily", "unknown")
        df["asset_type"] = df.get("asset_type", "unknown")
        df["business_model"] = df.get("business_model", "unknown")
        df["service_intensity"] = df.get("service_intensity", "medium")
        df["lead_intent"] = df.get("lead_intent", "quote")
        df["contact_role"] = df.get("contact_role", "unknown")

        df["seasonality_index"] = df.get("seasonality_index", np.sin((df["month"] - 1) / 12.0 * 2.0 * np.pi))
        df["macro_index"] = df.get("macro_index", 1.0)
        df["urgency_score"] = df.get("urgency_score", 0.5)
        df["churn_risk"] = df.get("churn_risk", 0.4)
        df["win_probability"] = df.get("win_probability", 0.5)

        df["competitor_price_index"] = df["province"].map(_province_competition_index)
        df["logistics_cost_index"] = df["province"].map(_province_logistics_index)
        df["purchasing_power_index"] = df["province"].map(_province_purchasing_power_index)
        df["loyalty_score"] = df["customer_segment"].map(_segment_loyalty)
        df["payment_behavior_score"] = df["customer_segment"].map(_segment_payment_score)
        df["event_factor"] = np.where(df["month"].isin([5, 6, 7, 11, 12]), 0.08, 0.0)

        if "stock_available" not in df.columns:
            df["stock_available"] = np.nan
        if "stock_rotations_90d" not in df.columns:
            df["stock_rotations_90d"] = np.nan

        df["stock_pressure"] = np.where(
            df["stock_available"].fillna(9999) <= 3, 1.0,
            np.where(df["stock_available"].fillna(9999) <= 10, 0.65, 0.25)
        )
        df["elasticity_estimate"] = np.clip(
            1.8 * df["price_sensitivity"].fillna(0.5) + 0.8 * (1 - df["service_attachment"].fillna(0.5)),
            0.2,
            2.5,
        )
        df["payment_risk"] = np.clip(1 - df["payment_behavior_score"], 0.0, 1.0)

        return df

    def _lookup_product(self, product_id: str) -> dict[str, Any]:
        if self.product_master.empty:
            return {}
        row = self.product_master[self.product_master["product_id"].astype(str) == str(product_id)]
        if row.empty:
            return {}
        return row.iloc[0].to_dict()

    def _lookup_account(self, account_id: str | None) -> dict[str, Any]:
        if not account_id or self.account_master.empty:
            return {}
        row = self.account_master[self.account_master["account_id"].astype(str) == str(account_id)]
        if row.empty:
            return {}
        return row.iloc[0].to_dict()

    def _historical_cost_ratio(self, product_id: str | None, category: str | None = None) -> float:
        if not self.product_stats:
            return 0.58
        if product_id and product_id in self.product_stats:
            return float(self.product_stats[product_id].get("cost_ratio", 0.58))
        if category and category in self.product_stats:
            return float(self.product_stats[category].get("cost_ratio", 0.58))
        return 0.58

    def build_request_frame(self, payload: dict[str, Any]) -> pd.DataFrame:
        product = self._lookup_product(payload.get("product_id", ""))
        account = self._lookup_account(payload.get("account_id"))

        transaction_date = pd.to_datetime(payload.get("transaction_date") or datetime.utcnow(), errors="coerce")
        month = int(transaction_date.month if pd.notna(transaction_date) else datetime.utcnow().month)

        base_price = float(payload.get("base_price") or product.get("base_price_usd") or payload.get("current_price") or 1000.0)
        current_price = float(payload.get("current_price") or payload.get("list_price") or product.get("max_price_usd") or base_price)
        list_price = float(payload.get("list_price") or product.get("max_price_usd") or current_price)
        cost_price = float(payload.get("cost_price") or (base_price * self._historical_cost_ratio(payload.get("product_id"), payload.get("category"))))
        quantity = float(payload.get("quantity", 1.0))

        province = payload.get("province") or account.get("province") or product.get("province") or "unknown"
        sector = payload.get("sector") or account.get("sector") or "unknown"
        industry = payload.get("industry") or account.get("industry") or "unknown"
        customer_segment = payload.get("customer_segment") or account.get("account_tier") or "recurrent"

        seasonal_hint = payload.get("seasonality_index")
        if seasonal_hint is None:
            seasonal_hint = _seasonality_from_month(month)

        stock_available = payload.get("stock_available")
        if stock_available is None:
            stock_available = payload.get("inventory_available")
        if stock_available is None:
            stock_available = 999.0

        frequency_12m = payload.get("frequency_12m")
        if frequency_12m is None:
            frequency_12m = 4.0 if _safe_lower(customer_segment) == "vip" else 2.0

        days_since_last_purchase = payload.get("days_since_last_purchase")
        if days_since_last_purchase is None:
            days_since_last_purchase = 45 if _safe_lower(customer_segment) == "recurrent" else 120

        price_sensitivity = payload.get("price_sensitivity")
        if price_sensitivity is None:
            price_sensitivity = account.get("price_sensitivity", 0.45)

        loyalty_score = payload.get("loyalty_score")
        if loyalty_score is None:
            loyalty_score = _segment_loyalty(str(customer_segment))

        payment_behavior_score = payload.get("payment_behavior_score")
        if payment_behavior_score is None:
            payment_behavior_score = _segment_payment_score(str(customer_segment))

        data = {
            "product_id": str(payload.get("product_id")),
            "account_id": payload.get("account_id"),
            "transaction_date": transaction_date,
            "current_price": current_price,
            "list_price": list_price,
            "base_price_usd": base_price,
            "cost_price": cost_price,
            "quantity": quantity,
            "province": province,
            "sector": sector,
            "industry": industry,
            "customer_segment": customer_segment,
            "loyalty_score": loyalty_score,
            "frequency_12m": float(frequency_12m),
            "days_since_last_purchase": int(days_since_last_purchase),
            "payment_behavior_score": float(payment_behavior_score),
            "overdue_days": int(payload.get("overdue_days", 0)),
            "on_time_payment_ratio": float(payload.get("on_time_payment_ratio", 0.85)),
            "payment_type": payload.get("payment_type", "transfer"),
            "stock_available": float(stock_available),
            "stock_rotations_90d": float(payload.get("stock_rotations_90d", 1.0)),
            "stock_out_days_90d": float(payload.get("stock_out_days_90d", 0.0)),
            "seasonality_index": float(seasonal_hint),
            "macro_index": float(payload.get("macro_index", 1.0)),
            "demand_history_index": float(payload.get("demand_history_index", 0.5)),
            "event_factor": float(payload.get("event_factor", 0.0)),
            "competitor_price_index": float(payload.get("competitor_price_index", _province_competition_index(province))),
            "logistics_cost_index": float(payload.get("logistics_cost_index", _province_logistics_index(province))),
            "purchasing_power_index": float(payload.get("purchasing_power_index", _province_purchasing_power_index(province))),
            "discount_history_avg": float(payload.get("discount_history_avg", 0.08)),
            "price_sensitivity": float(price_sensitivity),
            "elasticity_estimate": float(payload.get("elasticity_estimate", np.clip(1.8 * float(price_sensitivity) + 0.8 * (1 - float(account.get("service_attachment", 0.5))), 0.2, 2.5))),
            "category": payload.get("category") or product.get("section") or "unknown",
            "family": payload.get("family") or product.get("family") or "unknown",
            "subfamily": payload.get("subfamily") or product.get("subfamily") or "unknown",
            "business_model": payload.get("business_model") or product.get("business_model") or "unknown",
            "asset_type": payload.get("asset_type") or product.get("asset_type") or "unknown",
            "service_intensity": payload.get("service_intensity") or product.get("service_intensity") or "medium",
            "criticality": int(payload.get("criticality", product.get("criticality", 3) or 3)),
            "section": product.get("section", payload.get("category") or "unknown"),
            "min_price_usd": float(product.get("min_price_usd", max(cost_price * 1.1, 0.0))),
            "max_price_usd": float(product.get("max_price_usd", current_price * 1.2)),
            "base_price_catalog": float(product.get("base_price_usd", base_price)),
            "lead_intent": payload.get("lead_intent") or product.get("lead_intent") or "quote",
            "service_attachment": float(account.get("service_attachment", payload.get("service_attachment", 0.5))),
            "annual_budget_multiplier": float(account.get("annual_budget_multiplier", payload.get("annual_budget_multiplier", 1.0))),
            "account_tier": account.get("account_tier", payload.get("account_tier", "B")),
            "industry_fit_score": float(payload.get("industry_fit_score", 0.55)),
            "urgency_score": float(payload.get("urgency_score", 0.55)),
            "churn_risk": float(payload.get("churn_risk", 0.35)),
            "win_probability": float(payload.get("win_probability", 0.50)),
            "lead_time_days": float(payload.get("lead_time_days", product.get("avg_lead_time_days", 25))),
            "avg_lifespan_months": float(product.get("avg_lifespan_months", payload.get("avg_lifespan_months", 84))),
            "avg_lead_time_days": float(product.get("avg_lead_time_days", payload.get("avg_lead_time_days", 25))),
        }

        df = pd.DataFrame([data])
        df["month"] = month
        df["quarter"] = int((month - 1) // 3 + 1)
        df["dayofweek"] = int(transaction_date.dayofweek if pd.notna(transaction_date) else 2)
        df["is_month_end"] = int(transaction_date.is_month_end if pd.notna(transaction_date) else False)
        df["seasonality_bucket"] = pd.cut(
            df["seasonality_index"],
            bins=[-1.1, -0.4, 0.2, 1.1],
            labels=["low", "medium", "high"],
            include_lowest=True,
        ).astype(str)

        df["demand_pressure"] = (
            0.30 * df["seasonality_index"]
            + 0.22 * df["urgency_score"]
            + 0.18 * df["win_probability"]
            + 0.15 * df["competitor_price_index"]
            + 0.10 * df["purchasing_power_index"]
            - 0.20 * (df["stock_available"].clip(lower=0) <= 3).astype(float)
        )

        df["loyalty_band"] = pd.cut(
            df["loyalty_score"],
            bins=[-0.1, 0.35, 0.7, 1.1],
            labels=["low", "medium", "high"],
            include_lowest=True,
        ).astype(str)

        df["payment_band"] = pd.cut(
            df["payment_behavior_score"],
            bins=[-0.1, 0.35, 0.7, 1.1],
            labels=["risky", "normal", "strong"],
            include_lowest=True,
        ).astype(str)

        df["inventory_pressure"] = np.where(df["stock_available"] <= 3, 1.0, np.where(df["stock_available"] <= 10, 0.7, 0.25))
        df["payment_risk"] = 1 - df["payment_behavior_score"].clip(0, 1)
        df["margin_buffer_pct"] = np.where(df["current_price"] > 0, (df["current_price"] - df["cost_price"]) / df["current_price"], 0.0)

        return df

    def request_to_feature_row(self, payload: dict[str, Any]) -> pd.DataFrame:
        return self.build_request_frame(payload)

    def factor_scores(self, row: pd.Series) -> dict[str, float]:
        seasonality = float((row.get("seasonality_index", 0.0) + 1) / 2 * 100)
        loyalty = float(row.get("loyalty_score", 0.5) * 100)
        payment = float(row.get("payment_behavior_score", 0.5) * 100)
        location = float(((row.get("purchasing_power_index", 0.7) + (1 - row.get("logistics_cost_index", 0.7)) + (1 - row.get("competitor_price_index", 0.7))) / 3) * 100)
        stock = float((1 - min(max(row.get("inventory_pressure", 0.5), 0), 1)) * 100)
        elasticity = float((1 - min(max(row.get("elasticity_estimate", 1.0) / 2.5, 0), 1)) * 100)
        margin = float(min(max(row.get("margin_buffer_pct", 0.25) / 0.5, 0), 1) * 100)
        return {
            "seasonality": round(seasonality, 2),
            "loyalty": round(loyalty, 2),
            "payment_behavior": round(payment, 2),
            "location": round(location, 2),
            "stock": round(stock, 2),
            "elasticity": round(elasticity, 2),
            "margin_buffer": round(margin, 2),
        }

    def factor_impact(self, row: pd.Series) -> dict[str, float]:
        return {
            "seasonality": round(float(row.get("seasonality_index", 0.0)) * 0.04, 4),
            "loyalty": round((float(row.get("loyalty_score", 0.5)) - 0.5) * -0.05, 4),
            "payment_behavior": round((float(row.get("payment_behavior_score", 0.5)) - 0.5) * 0.04, 4),
            "location": round(
                (
                    (float(row.get("purchasing_power_index", 0.7)) - 0.7) * 0.03
                    + (0.7 - float(row.get("logistics_cost_index", 0.7))) * 0.02
                    + (0.7 - float(row.get("competitor_price_index", 0.7))) * 0.02
                ),
                4,
            ),
            "stock": round((0.8 - float(row.get("inventory_pressure", 0.5))) * 0.06, 4),
            "elasticity": round(-(float(row.get("elasticity_estimate", 1.0)) - 1.0) * 0.03, 4),
            "margin": round((float(row.get("margin_buffer_pct", 0.2)) - 0.2) * 0.05, 4),
        }
