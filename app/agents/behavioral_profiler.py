from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.agents.schemas import BehavioralProfile


@dataclass
class BehavioralProfilerAgent:
    """
    Behavioral Profiler:
    - Segmentation del cliente por recurrencia, riesgo y sensibilidad al precio.
    - Basado en features reales ya construidas en la SQLite.
    """

    def analyze(self, context: dict[str, Any]) -> BehavioralProfile:
        payload = context.get("payload", {})
        account_row = context.get("account_row", {})

        account_id = str(payload.get("account_id") or account_row.get("account_id") or "unknown")
        segment_rule = str(payload.get("segment_rule_based") or account_row.get("segment_rule_based") or "new_or_low_activity")
        rfm_score = self._to_float(payload.get("rfm_score") or account_row.get("rfm_score"), 3.0)
        risk_score = self._to_float(payload.get("risk_score_0_100") or account_row.get("risk_score_0_100"), 50.0)
        recency_days = self._to_float(payload.get("recency_days") or account_row.get("recency_days"))
        frequency_90d = self._to_float(payload.get("frequency_90d") or account_row.get("frequency_90d"))
        churn_risk = self._to_float(payload.get("churn_risk") or account_row.get("avg_churn_risk"))
        compliance_rate = self._to_float(payload.get("compliance_rate") or account_row.get("compliance_rate"), 0.8)
        price_sensitivity = self._to_float(payload.get("price_sensitivity") or account_row.get("price_sensitivity"), 0.5)
        avg_discount_pct = self._to_float(payload.get("avg_discount_pct") or account_row.get("avg_discount_pct"), 0.10)

        if price_sensitivity >= 0.75 or (avg_discount_pct >= 0.14 and churn_risk and churn_risk >= 0.6):
            sensitivity_bucket = "high"
        elif price_sensitivity <= 0.35 and (rfm_score >= 4.0 or frequency_90d and frequency_90d >= 6):
            sensitivity_bucket = "low"
        else:
            sensitivity_bucket = "medium"

        if rfm_score >= 4.2 and risk_score < 30:
            segment = "vip"
        elif rfm_score >= 3.2 and risk_score < 55:
            segment = "recurrent"
        elif risk_score >= 70:
            segment = "at_risk"
        else:
            segment = segment_rule

        summary = (
            f"Cuenta {account_id}: segmento={segment}, sensibilidad={sensitivity_bucket}, "
            f"RFM={rfm_score:.2f}, riesgo={risk_score:.0f}/100, "
            f"cumplimiento={compliance_rate:.2f}."
        )

        return BehavioralProfile(
            account_id=account_id,
            segment=segment,
            sensitivity_bucket=sensitivity_bucket,
            recency_days=recency_days,
            frequency_90d=frequency_90d,
            rfm_score=rfm_score,
            churn_risk=churn_risk,
            compliance_rate=compliance_rate,
            price_sensitivity=price_sensitivity,
            summary=summary,
        )

    @staticmethod
    def _to_float(value: Any, default: float | None = None) -> float | None:
        if value is None:
            return default
        try:
            return float(value)
        except Exception:
            return default