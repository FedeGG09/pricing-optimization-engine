from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from tqdm.auto import tqdm


# Enable pandas progress helpers when available
try:
    tqdm.pandas(desc="Processing rows")
except Exception:
    pass


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts" / "pricing"
DEFAULT_SQLITE_PATH = PROJECT_ROOT / "data" / "industrial_pricing_features.sqlite"
DEFAULT_FEATURES_TABLE = "pricing_training_set"


# -----------------------------
# Data structures
# -----------------------------


@dataclass
class PricingRequest:
    account_id: Optional[str] = None
    product_id: Optional[str] = None
    province: Optional[str] = None
    segment_rule_based: Optional[str] = None
    segment_kmeans_label: Optional[str] = None
    risk_score_0_100: Optional[float] = None
    rfm_score: Optional[float] = None
    price_floor_usd: Optional[float] = None
    price_ceiling_usd: Optional[float] = None
    negotiated_price_usd: Optional[float] = None
    list_price_usd: Optional[float] = None
    base_price_usd: Optional[float] = None
    avg_discount_pct: Optional[float] = None
    avg_margin_pct: Optional[float] = None
    elasticity_estimate: Optional[float] = None
    urgency_score: Optional[float] = None
    churn_risk: Optional[float] = None
    compliance_rate: Optional[float] = None
    seasonality_index: Optional[float] = None
    month: Optional[int] = None
    dayofweek: Optional[int] = None
    quantity: Optional[float] = None
    stock_units: Optional[float] = None
    margin_target_pct: Optional[float] = None
    price_sensitivity: Optional[float] = None
    annual_budget_multiplier: Optional[float] = None
    service_attachment: Optional[float] = None
    funnel_stage: Optional[str] = None
    next_best_action: Optional[str] = None
    industry_fit_score: Optional[float] = None
    is_peak_season_proxy: Optional[int] = None
    month_sin: Optional[float] = None
    month_cos: Optional[float] = None


@dataclass
class PricingRecommendation:
    recommended_price_usd: float
    recommended_discount_pct: float
    model_price_usd: float
    floor_price_usd: float
    ceiling_price_usd: float
    final_margin_estimate_pct: float
    rule_adjustments: Dict[str, Any]
    factor_scores: Dict[str, float]
    explanation: str
    model_version: str


@dataclass
class ScenarioResult:
    candidate_multiplier: float
    candidate_price_usd: float
    expected_margin_pct: float
    expected_revenue_score: float
    feasible: bool
    notes: str


# -----------------------------
# Helpers
# -----------------------------


def _safe_json_load(path: Path, default: Optional[dict] = None) -> dict:
    if not path.exists():
        return default or {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _clip(v: float, low: float, high: float) -> float:
    return float(max(low, min(high, v)))


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, str) and value.strip() == "":
            return default
        if isinstance(value, (list, dict, tuple, set)):
            return default
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if value is None:
            return default
        return int(float(value))
    except Exception:
        return default


def _compute_discount(list_price: Optional[float], negotiated_price: Optional[float]) -> float:
    lp = _safe_float(list_price)
    np_ = _safe_float(negotiated_price)
    if not np.isfinite(lp) or lp <= 0 or not np.isfinite(np_):
        return np.nan
    return float(1.0 - (np_ / lp))


# -----------------------------
# Main engine
# -----------------------------


class DynamicPricingEngine:
    def __init__(
        self,
        artifacts_dir: Union[str, Path] = DEFAULT_ARTIFACTS_DIR,
        sqlite_path: Union[str, Path] = DEFAULT_SQLITE_PATH,
        feature_table: str = DEFAULT_FEATURES_TABLE,
    ) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.sqlite_path = Path(sqlite_path)
        self.feature_table = feature_table

        self.model_path = self.artifacts_dir / "pricing_model.joblib"
        self.metadata_path = self.artifacts_dir / "pricing_model_metadata.json"
        self.features_path = self.artifacts_dir / "pricing_feature_columns.json"

        self.pipeline = None
        self.metadata: Dict[str, Any] = {}
        self.numeric_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.all_features: List[str] = []
        self.model_version: str = "v1"

        self._load_artifacts()

    def _load_artifacts(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(f"No se encontró el modelo: {self.model_path}")

        self.pipeline = joblib.load(self.model_path)
        self.metadata = _safe_json_load(self.metadata_path, default={})
        features_meta = _safe_json_load(self.features_path, default={})

        self.numeric_cols = list(features_meta.get("numeric_cols", self.metadata.get("numeric_cols", [])))
        self.categorical_cols = list(features_meta.get("categorical_cols", self.metadata.get("categorical_cols", [])))
        self.all_features = list(features_meta.get("all_features", self.numeric_cols + self.categorical_cols))
        self.model_version = str(self.metadata.get("trained_at_utc", "v1"))

    def _load_reference_tables(self) -> Dict[str, pd.DataFrame]:
        if not self.sqlite_path.exists():
            return {}

        tables = ["account_features", "product_features", "seasonality_index", "account_segments", self.feature_table]
        out: Dict[str, pd.DataFrame] = {}
        with sqlite3.connect(str(self.sqlite_path)) as conn:
            for table in tables:
                try:
                    out[table] = pd.read_sql_query(f"SELECT * FROM {table}", conn)
                except Exception:
                    out[table] = pd.DataFrame()
        return out

    def _align_payload(self, payload: Dict[str, Any]) -> pd.DataFrame:
        row: Dict[str, Any] = {}

        # Fill every known feature explicitly so the model gets the same schema as training.
        for col in self.all_features:
            row[col] = payload.get(col, np.nan)

        # Also map common aliases from the request layer.
        alias_map = {
            "recommended_discount_pct": "target_discount_pct",
            "discount_pct": "target_discount_pct",
            "price": "negotiated_price_usd",
            "base_price": "base_price_usd",
            "list_price": "list_price_usd",
            "segment": "segment_rule_based",
        }
        for alias, canonical in alias_map.items():
            if alias in payload and canonical in self.all_features:
                row[canonical] = payload.get(alias)

        # Ensure date-derived features default gracefully.
        defaults = {
            "month": 1,
            "dayofweek": 0,
            "is_peak_season_proxy": 0,
            "month_sin": 0.0,
            "month_cos": 1.0,
            "seasonality_index": 0.0,
            "risk_score_0_100": 50.0,
            "rfm_score": 3.0,
            "urgency_score": 0.5,
            "churn_risk": 0.5,
            "compliance_rate": 0.8,
            "price_sensitivity": 0.5,
            "annual_budget_multiplier": 1.0,
            "service_attachment": 0.5,
            "quantity": 1.0,
            "stock_units": 1.0,
            "margin_target_pct": 0.25,
            "elasticity_estimate": -1.0,
            "avg_discount_pct": 0.1,
            "avg_margin_pct": 0.3,
            "price_floor_usd": np.nan,
            "price_ceiling_usd": np.nan,
        }
        for k, v in defaults.items():
            if k in self.all_features and (row.get(k) is None or (isinstance(row.get(k), float) and np.isnan(row.get(k)))):
                row[k] = v

        df = pd.DataFrame([row])
        return df.reindex(columns=self.all_features, fill_value=np.nan)

    def _preprocess_request(self, request: Union[PricingRequest, Dict[str, Any]]) -> Dict[str, Any]:
        if isinstance(request, PricingRequest):
            payload = asdict(request)
        else:
            payload = dict(request)

        # Normalize numeric fields
        numeric_fields = set(self.numeric_cols) | {
            "risk_score_0_100",
            "rfm_score",
            "price_floor_usd",
            "price_ceiling_usd",
            "negotiated_price_usd",
            "list_price_usd",
            "base_price_usd",
            "avg_discount_pct",
            "avg_margin_pct",
            "elasticity_estimate",
            "urgency_score",
            "churn_risk",
            "compliance_rate",
            "seasonality_index",
            "month_sin",
            "month_cos",
            "quantity",
            "stock_units",
            "margin_target_pct",
            "price_sensitivity",
            "annual_budget_multiplier",
            "service_attachment",
            "industry_fit_score",
        }
        for k in list(payload.keys()):
            if k in numeric_fields:
                payload[k] = _safe_float(payload.get(k))

        for k in ["month", "dayofweek", "is_peak_season_proxy"]:
            if k in payload:
                payload[k] = _safe_int(payload.get(k))

        return payload

    def _rule_engine(self, payload: Dict[str, Any], model_price_usd: float) -> Tuple[float, Dict[str, Any], Dict[str, float], str]:
        list_price_usd = _safe_float(payload.get("list_price_usd"))
        negotiated_price_usd = _safe_float(payload.get("negotiated_price_usd"))
        floor_price_usd = _safe_float(payload.get("price_floor_usd"))
        ceiling_price_usd = _safe_float(payload.get("price_ceiling_usd"))
        margin_target_pct = _safe_float(payload.get("margin_target_pct"), 0.25)
        risk_score = _safe_float(payload.get("risk_score_0_100"), 50.0)
        rfm_score = _safe_float(payload.get("rfm_score"), 3.0)
        elasticity = _safe_float(payload.get("elasticity_estimate"), -1.0)
        seasonality = _safe_float(payload.get("seasonality_index"), 0.0)
        stock_units = _safe_float(payload.get("stock_units"), 1.0)
        urgency = _safe_float(payload.get("urgency_score"), 0.5)
        compliance_rate = _safe_float(payload.get("compliance_rate"), 0.8)
        segment = str(payload.get("segment_rule_based") or payload.get("segment_kmeans_label") or "new_or_low_activity")

        adjustments: Dict[str, Any] = {}
        factor_scores: Dict[str, float] = {}

        # Floors and ceilings
        if not np.isfinite(floor_price_usd) or floor_price_usd <= 0:
            if np.isfinite(negotiated_price_usd) and negotiated_price_usd > 0:
                floor_price_usd = negotiated_price_usd * 0.85
            elif np.isfinite(list_price_usd) and list_price_usd > 0:
                floor_price_usd = list_price_usd * 0.80
            else:
                floor_price_usd = max(model_price_usd * 0.75, 1.0)

        if not np.isfinite(ceiling_price_usd) or ceiling_price_usd <= 0:
            if np.isfinite(list_price_usd) and list_price_usd > 0:
                ceiling_price_usd = list_price_usd * 1.05
            else:
                ceiling_price_usd = model_price_usd * 1.20

        base_price = _clip(model_price_usd, floor_price_usd, ceiling_price_usd)

        # Segment strategy
        if segment in {"vip", "segment_0"}:
            segment_multiplier = 1.02
            discount_cap = 0.08
        elif segment in {"recurrent", "segment_1"}:
            segment_multiplier = 1.00
            discount_cap = 0.12
        elif segment in {"at_risk", "segment_2"}:
            segment_multiplier = 0.96
            discount_cap = 0.16
        else:
            segment_multiplier = 0.98
            discount_cap = 0.10
        adjustments["segment_multiplier"] = segment_multiplier
        adjustments["discount_cap"] = discount_cap

        # Risk / urgency / compliance based adjustments
        risk_multiplier = 1.0 - _clip((risk_score - 50.0) / 500.0, -0.06, 0.08)
        urgency_multiplier = 1.0 + _clip((urgency - 0.5) * 0.08, -0.04, 0.05)
        compliance_multiplier = 1.0 + _clip((compliance_rate - 0.8) * 0.04, -0.02, 0.03)
        seasonality_multiplier = 1.0 + _clip(seasonality * 0.06, -0.08, 0.10)

        # Elasticity-based pricing pressure
        if np.isfinite(elasticity):
            elasticity_adjustment = 1.0 + _clip((-elasticity - 1.0) * 0.03, -0.05, 0.05)
        else:
            elasticity_adjustment = 1.0

        # Stock-based logic
        if np.isfinite(stock_units) and stock_units <= 5:
            scarcity_multiplier = 1.03
        elif np.isfinite(stock_units) and stock_units > 100:
            scarcity_multiplier = 0.99
        else:
            scarcity_multiplier = 1.0

        # Margin target guardrail
        if np.isfinite(list_price_usd) and list_price_usd > 0:
            implied_margin_pct = (base_price - _safe_float(payload.get("base_price_usd"), base_price * 0.7)) / max(base_price, 1e-6)
        else:
            implied_margin_pct = margin_target_pct

        margin_multiplier = 1.0
        if implied_margin_pct < margin_target_pct:
            margin_multiplier = 1.0 + min((margin_target_pct - implied_margin_pct) * 0.5, 0.08)
        else:
            margin_multiplier = 1.0 - min((implied_margin_pct - margin_target_pct) * 0.2, 0.04)

        final_price = base_price
        for m in [segment_multiplier, risk_multiplier, urgency_multiplier, compliance_multiplier, seasonality_multiplier, elasticity_adjustment, scarcity_multiplier, margin_multiplier]:
            final_price *= m

        final_price = _clip(final_price, floor_price_usd, ceiling_price_usd)

        recommended_discount_pct = np.nan
        if np.isfinite(list_price_usd) and list_price_usd > 0:
            recommended_discount_pct = max(0.0, min(0.75, 1.0 - (final_price / list_price_usd)))
            recommended_discount_pct = min(recommended_discount_pct, discount_cap)
            final_price = list_price_usd * (1.0 - recommended_discount_pct)
            final_price = _clip(final_price, floor_price_usd, ceiling_price_usd)

        # factor scores (0-100, higher means stronger positive effect on price)
        factor_scores["risk"] = _clip(100.0 - risk_score, 0, 100)
        factor_scores["urgency"] = _clip(urgency * 100.0, 0, 100)
        factor_scores["seasonality"] = _clip((seasonality + 1.0) * 50.0, 0, 100)
        factor_scores["compliance"] = _clip(compliance_rate * 100.0, 0, 100)
        factor_scores["elasticity"] = _clip((1.0 / (1.0 + abs(elasticity))) * 100.0 if np.isfinite(elasticity) else 50.0, 0, 100)
        factor_scores["segment"] = 90.0 if segment == "vip" else 70.0 if segment == "recurrent" else 45.0 if segment == "new_or_low_activity" else 35.0
        factor_scores["stock"] = 85.0 if stock_units <= 5 else 55.0 if stock_units <= 20 else 35.0

        explanation_parts = []
        if segment == "vip":
            explanation_parts.append("Cliente premium con política de protección de margen.")
        elif segment == "at_risk":
            explanation_parts.append("Cliente con mayor sensibilidad: se limita el aumento y se conserva competitividad.")
        else:
            explanation_parts.append("Cliente estándar con política balanceada entre conversión y margen.")

        if risk_score >= 70:
            explanation_parts.append("Riesgo alto: se evita sobre-descontar y se protege el precio.")
        if urgency >= 0.7:
            explanation_parts.append("Urgencia alta: se prioriza cierre rápido.")
        if seasonality > 0.15:
            explanation_parts.append("Estacionalidad favorable: se permite mayor precio.")
        if np.isfinite(elasticity) and elasticity < -1.5:
            explanation_parts.append("Elasticidad alta: el precio fue moderado para no destruir demanda.")
        if stock_units <= 5:
            explanation_parts.append("Stock bajo: se evita descuento agresivo.")

        explanation = " ".join(explanation_parts)

        return final_price, adjustments, factor_scores, explanation

    def recommend(self, request: Union[PricingRequest, Dict[str, Any]]) -> PricingRecommendation:
        payload = self._preprocess_request(request)
        model_input = self._align_payload(payload)
        model_price = float(self.pipeline.predict(model_input)[0])

        final_price, adjustments, factor_scores, explanation = self._rule_engine(payload, model_price)

        list_price_usd = _safe_float(payload.get("list_price_usd"))
        if np.isfinite(list_price_usd) and list_price_usd > 0:
            discount_pct = max(0.0, min(0.75, 1.0 - final_price / list_price_usd))
        else:
            discount_pct = np.nan

        base_price_usd = _safe_float(payload.get("base_price_usd"), np.nan)
        if not np.isfinite(base_price_usd):
            base_price_usd = final_price * 0.72

        final_margin_estimate_pct = float((final_price - base_price_usd) / max(final_price, 1e-6))

        return PricingRecommendation(
            recommended_price_usd=float(round(final_price, 4)),
            recommended_discount_pct=float(round(discount_pct, 4)) if np.isfinite(discount_pct) else np.nan,
            model_price_usd=float(round(model_price, 4)),
            floor_price_usd=float(round(_safe_float(payload.get("price_floor_usd"), final_price * 0.85), 4)),
            ceiling_price_usd=float(round(_safe_float(payload.get("price_ceiling_usd"), final_price * 1.15), 4)),
            final_margin_estimate_pct=float(round(final_margin_estimate_pct, 4)),
            rule_adjustments=adjustments,
            factor_scores=factor_scores,
            explanation=explanation,
            model_version=self.model_version,
        )

    def explain(self, request: Union[PricingRequest, Dict[str, Any]]) -> Dict[str, Any]:
        rec = self.recommend(request)
        return {
            "model_version": rec.model_version,
            "recommended_price_usd": rec.recommended_price_usd,
            "recommended_discount_pct": rec.recommended_discount_pct,
            "model_price_usd": rec.model_price_usd,
            "floor_price_usd": rec.floor_price_usd,
            "ceiling_price_usd": rec.ceiling_price_usd,
            "final_margin_estimate_pct": rec.final_margin_estimate_pct,
            "rule_adjustments": rec.rule_adjustments,
            "factor_scores": rec.factor_scores,
            "explanation": rec.explanation,
        }

    def simulate(
        self,
        request: Union[PricingRequest, Dict[str, Any]],
        candidate_multipliers: Optional[Sequence[float]] = None,
    ) -> List[ScenarioResult]:
        payload = self._preprocess_request(request)
        model_input = self._align_payload(payload)
        model_price = float(self.pipeline.predict(model_input)[0])

        if candidate_multipliers is None:
            candidate_multipliers = [0.85, 0.90, 0.95, 1.00, 1.03, 1.05, 1.08, 1.10]

        list_price_usd = _safe_float(payload.get("list_price_usd"), model_price * 1.15)
        base_price_usd = _safe_float(payload.get("base_price_usd"), model_price * 0.72)
        margin_target_pct = _safe_float(payload.get("margin_target_pct"), 0.25)
        floor_price_usd = _safe_float(payload.get("price_floor_usd"), model_price * 0.85)
        ceiling_price_usd = _safe_float(payload.get("price_ceiling_usd"), model_price * 1.15)
        elasticity = _safe_float(payload.get("elasticity_estimate"), -1.0)

        results: List[ScenarioResult] = []
        for mult in tqdm(candidate_multipliers, desc="Simulating scenarios", unit="scenario"):
            candidate_price = _clip(model_price * float(mult), floor_price_usd, ceiling_price_usd)
            if np.isfinite(list_price_usd) and list_price_usd > 0:
                candidate_discount = max(0.0, min(0.75, 1.0 - candidate_price / list_price_usd))
            else:
                candidate_discount = np.nan

            margin_pct = float((candidate_price - base_price_usd) / max(candidate_price, 1e-6))

            # A simple expected revenue score proxy, used for scenario ranking.
            demand_factor = 1.0
            if np.isfinite(elasticity):
                demand_factor = max(0.1, 1.0 + elasticity * (candidate_price / max(model_price, 1e-6) - 1.0))

            expected_revenue_score = float(candidate_price * demand_factor)
            feasible = margin_pct >= margin_target_pct and candidate_price >= floor_price_usd and candidate_price <= ceiling_price_usd
            notes = []
            if candidate_price <= floor_price_usd:
                notes.append("Touched floor")
            if candidate_price >= ceiling_price_usd:
                notes.append("Touched ceiling")
            if margin_pct < margin_target_pct:
                notes.append("Below target margin")
            if np.isfinite(candidate_discount) and candidate_discount > 0.12:
                notes.append("Deep discount")

            results.append(
                ScenarioResult(
                    candidate_multiplier=float(mult),
                    candidate_price_usd=float(round(candidate_price, 4)),
                    expected_margin_pct=float(round(margin_pct, 4)),
                    expected_revenue_score=float(round(expected_revenue_score, 4)),
                    feasible=feasible,
                    notes="; ".join(notes) if notes else "OK",
                )
            )

        return results

    def batch_recommend(self, df: pd.DataFrame, limit: Optional[int] = None) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()

        work_df = df.copy()
        if limit is not None:
            work_df = work_df.head(limit).copy()

        rows: List[Dict[str, Any]] = []
        iterator = work_df.to_dict(orient="records")
        for record in tqdm(iterator, desc="Scoring batch recommendations", total=len(iterator), unit="row"):
            rec = self.recommend(record)
            output = dict(record)
            output.update(
                {
                    "recommended_price_usd": rec.recommended_price_usd,
                    "recommended_discount_pct": rec.recommended_discount_pct,
                    "final_margin_estimate_pct": rec.final_margin_estimate_pct,
                    "model_price_usd": rec.model_price_usd,
                    "floor_price_usd": rec.floor_price_usd,
                    "ceiling_price_usd": rec.ceiling_price_usd,
                    "model_version": rec.model_version,
                }
            )
            rows.append(output)

        return pd.DataFrame(rows)


# -----------------------------
# Convenience API
# -----------------------------


_ENGINE: Optional[DynamicPricingEngine] = None


def get_engine() -> DynamicPricingEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = DynamicPricingEngine()
    return _ENGINE


def recommend_price(payload: Union[PricingRequest, Dict[str, Any]]) -> Dict[str, Any]:
    engine = get_engine()
    return engine.explain(payload)


def simulate_price(payload: Union[PricingRequest, Dict[str, Any]], multipliers: Optional[Sequence[float]] = None) -> List[Dict[str, Any]]:
    engine = get_engine()
    scenarios = engine.simulate(payload, candidate_multipliers=multipliers)
    return [asdict(s) for s in scenarios]


def batch_recommend(df: pd.DataFrame, limit: Optional[int] = None) -> pd.DataFrame:
    engine = get_engine()
    return engine.batch_recommend(df, limit=limit)
