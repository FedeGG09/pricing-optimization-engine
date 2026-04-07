from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.agents.schemas import DiscountGuardrail


@dataclass
class DiscountLogicAgent:
    """
    Discount Logic:
    - Capa determinística de seguridad.
    - Protege margen, techo, piso y topes por segmento.
    """

    def evaluate(self, context: dict[str, Any], recommendation: dict[str, Any] | None = None) -> DiscountGuardrail:
        payload = context.get("payload", {})
        rec = recommendation or {}

        price_floor = self._to_float(rec.get("floor_price_usd") or payload.get("price_floor_usd"))
        price_ceiling = self._to_float(rec.get("ceiling_price_usd") or payload.get("price_ceiling_usd"))
        recommended_price = self._to_float(rec.get("recommended_price_usd") or payload.get("negotiated_price_usd"))
        list_price = self._to_float(payload.get("list_price_usd"))
        base_price = self._to_float(payload.get("base_price_usd"))
        margin_target = self._to_float(payload.get("margin_target_pct"), 0.25) or 0.25
        segment = str(payload.get("segment_rule_based") or payload.get("segment_kmeans_label") or "new_or_low_activity")
        risk_score = self._to_float(payload.get("risk_score_0_100"), 50.0) or 50.0
        stock_units = self._to_float(payload.get("stock_units"), 10.0) or 10.0
        elasticity = self._to_float(payload.get("elasticity_estimate"), -1.0) or -1.0

        rule_hits: list[str] = []
        max_discount = 0.12
        min_margin = margin_target
        allowed = True
        blocked_reason = None

        if segment == "vip":
            max_discount = 0.08
            rule_hits.append("vip_discount_cap")
        elif segment == "recurrent":
            max_discount = 0.12
            rule_hits.append("recurrent_discount_cap")
        elif segment == "at_risk":
            max_discount = 0.16
            rule_hits.append("at_risk_discount_cap")
        else:
            max_discount = 0.10
            rule_hits.append("default_discount_cap")

        if risk_score >= 70:
            max_discount = min(max_discount, 0.08)
            rule_hits.append("high_risk_protection")

        if stock_units <= 5:
            max_discount = min(max_discount, 0.06)
            rule_hits.append("low_stock_protection")
        elif stock_units >= 100:
            max_discount = min(max_discount + 0.02, 0.18)
            rule_hits.append("overstock_relief")

        if elasticity < -1.5:
            max_discount = min(max_discount, 0.10)
            rule_hits.append("elasticity_sensitive")

        if recommended_price is not None and list_price and list_price > 0:
            implied_discount = max(0.0, min(0.99, 1.0 - (recommended_price / list_price)))
            if implied_discount > max_discount:
                allowed = False
                blocked_reason = f"discount {implied_discount:.2f} exceeds cap {max_discount:.2f}"
                rule_hits.append("discount_blocked")

        if base_price and recommended_price:
            margin = (recommended_price - base_price) / max(recommended_price, 1e-6)
            if margin < margin_target:
                allowed = False
                min_margin = margin_target
                blocked_reason = blocked_reason or f"margin {margin:.2f} below target {margin_target:.2f}"
                rule_hits.append("margin_blocked")

        return DiscountGuardrail(
            allowed=allowed,
            floor_price_usd=price_floor,
            ceiling_price_usd=price_ceiling,
            max_discount_pct=max_discount,
            min_margin_pct=min_margin,
            blocked_reason=blocked_reason,
            rule_hits=rule_hits,
        )

    @staticmethod
    def _to_float(value: Any, default: float | None = None) -> float | None:
        if value is None:
            return default
        try:
            return float(value)
        except Exception:
            return default