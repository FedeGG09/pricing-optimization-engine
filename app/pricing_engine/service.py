from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.agents.pricing_agent import PricingAgent
from app.core.config import settings
from app.pricing_engine.feature_builder import PricingFeatureBuilder
from app.pricing_engine.model_manager import PricingModelManager
from app.pricing_engine.rules import PricingRuleEngine


class PricingService:
    def __init__(
        self,
        data_dir: str | Path | None = None,
        artifacts_dir: str | Path | None = None,
    ):
        self.data_dir = Path(data_dir or settings.data_dir)
        self.artifacts_dir = Path(artifacts_dir or settings.artifacts_dir)
        self.model_manager = PricingModelManager(self.artifacts_dir)
        self.rule_engine = PricingRuleEngine()
        self.agent = PricingAgent()

    def _builder(self) -> PricingFeatureBuilder:
        artifacts = self.model_manager.load()
        return PricingFeatureBuilder.from_artifacts(self.data_dir, artifacts.bundle)

    def recommend(self, payload: dict[str, Any]) -> dict[str, Any]:
        builder = self._builder()
        row_df = builder.request_to_feature_row(payload)
        row = row_df.iloc[0]

        current_price = float(row["current_price"])
        model_ratio, model_quality = self.model_manager.predict_ratio(row_df)
        model_price = current_price * model_ratio

        rule_result = self.rule_engine.apply(
            row=row,
            model_price=model_price,
            current_price=current_price,
        )

        factor_scores = builder.factor_scores(row)
        factor_impact = builder.factor_impact(row)

        anomaly_score, anomaly_detected = self.model_manager.detect_anomaly(row_df)
        risk_flags = list(rule_result.risk_flags)
        if anomaly_detected:
            risk_flags.append("anomaly_detected")

        explanation_context = {
            "product_id": payload.get("product_id"),
            "account_id": payload.get("account_id"),
            "current_price": current_price,
            "model_price": rule_result.model_price,
            "final_price": rule_result.final_price,
            "factor_scores": factor_scores,
            "factor_impact": factor_impact,
            "risk_flags": risk_flags,
            "rule_trace": rule_result.rule_trace,
            "model_quality": model_quality,
            "anomaly_score": anomaly_score,
            "payload": payload,
        }

        llm_explanation = self.agent.explain(explanation_context) if payload.get("explain", True) else None
        llm_text = None
        if llm_explanation is not None:
            llm_text = llm_explanation.get("summary") or llm_explanation.get("raw") or str(llm_explanation)

        cost_price = float(row.get("cost_price", current_price * 0.65))
        margin_pct = max(0.0, (rule_result.final_price - cost_price) / rule_result.final_price) if rule_result.final_price > 0 else 0.0

        response = {
            "product_id": str(payload.get("product_id")),
            "account_id": payload.get("account_id"),
            "timestamp": datetime.now(timezone.utc),
            "current_price": round(current_price, 2),
            "model_price": rule_result.model_price,
            "final_price": rule_result.final_price,
            "discount_pct": rule_result.discount_pct,
            "margin_pct": round(float(margin_pct), 4),
            "min_allowed_price": rule_result.min_allowed_price,
            "max_allowed_price": rule_result.max_allowed_price,
            "suggested_action": rule_result.suggested_action,
            "confidence": round(float(np.clip(model_quality if model_quality else 0.65, 0.15, 0.95)), 2),
            "risk_flags": risk_flags,
            "factor_scores": factor_scores,
            "factor_impact": factor_impact,
            "explanation": rule_result.explanation,
            "llm_explanation": llm_text,
            "anomaly_score": anomaly_score,
            "anomaly_detected": anomaly_detected,
            "model_version": self.model_manager.load().bundle.model_version,
            "metadata": {
                "rule_trace": rule_result.rule_trace,
                "raw_model_ratio": model_ratio,
                "model_quality": model_quality,
            },
        }
        return response

    def explain(self, payload: dict[str, Any]) -> dict[str, Any]:
        recommendation = self.recommend({**payload, "explain": False})
        builder = self._builder()
        row = builder.request_to_feature_row(payload).iloc[0]
        context = {
            "recommendation": recommendation,
            "feature_scores": builder.factor_scores(row),
            "feature_impact": builder.factor_impact(row),
        }
        llm = self.agent.explain(context)
        return {
            "product_id": recommendation["product_id"],
            "explanation": recommendation["explanation"],
            "factor_scores": recommendation["factor_scores"],
            "factor_impact": recommendation["factor_impact"],
            "rule_trace": recommendation["metadata"]["rule_trace"],
            "llm_explanation": llm.get("summary") or llm.get("raw"),
            "metadata": recommendation["metadata"],
        }

    def simulate(self, payload: dict[str, Any]) -> dict[str, Any]:
        builder = self._builder()
        row_df = builder.request_to_feature_row(payload)
        row = row_df.iloc[0]

        current_price = float(row["current_price"])
        base_units = float(payload.get("base_units", row.get("quantity", 1.0)))
        cost_price = float(row.get("cost_price", current_price * 0.65))
        elasticity = float(row.get("elasticity_estimate", 1.0))
        loyalty = float(row.get("loyalty_score", 0.5))
        seasonality = float(row.get("seasonality_index", 0.0))
        demand_boost = 1.0 + 0.10 * max(0.0, seasonality) + 0.08 * max(0.0, loyalty - 0.5)

        scenarios = []
        for delta in payload.get("price_deltas_pct", [-0.10, -0.05, 0.0, 0.05, 0.10]):
            multiplier = 1.0 + float(delta)
            simulated_price = current_price * multiplier
            demand_change_pct = -elasticity * float(delta) * demand_boost
            expected_units = max(0.0, base_units * (1.0 + demand_change_pct))
            expected_revenue = simulated_price * expected_units
            expected_margin = (simulated_price - cost_price) * expected_units
            expected_margin_pct = (simulated_price - cost_price) / simulated_price if simulated_price > 0 else 0.0

            scenarios.append({
                "price_delta_pct": float(delta),
                "price_multiplier": round(multiplier, 4),
                "expected_units": round(expected_units, 2),
                "expected_revenue": round(expected_revenue, 2),
                "expected_margin": round(expected_margin, 2),
                "expected_margin_pct": round(expected_margin_pct, 4),
                "scenario_label": "discount" if delta < 0 else "increase" if delta > 0 else "baseline",
            })

        llm = self.agent.scenario({
            "product_id": payload.get("product_id"),
            "current_price": current_price,
            "base_units": base_units,
            "elasticity": elasticity,
            "scenarios": scenarios,
        })

        return {
            "product_id": payload.get("product_id"),
            "current_price": round(current_price, 2),
            "base_units": round(base_units, 2),
            "baseline_revenue": round(current_price * base_units, 2),
            "baseline_margin": round((current_price - cost_price) * base_units, 2),
            "scenarios": scenarios,
            "explanation": "Simulación basada en elasticidad estimada, presión de demanda, stock y reglas de margen.",
            "llm_explanation": llm.get("summary") or llm.get("raw"),
        }

    def train(self) -> dict[str, Any]:
        artifacts = self.model_manager.train(self.data_dir, max_rows=1000)
        return {
            "model_version": artifacts.bundle.model_version,
            "metrics": artifacts.bundle.metrics,
            "top_features": artifacts.bundle.feature_importance[:15],
        }
